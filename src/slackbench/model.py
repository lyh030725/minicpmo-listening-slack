from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel


@dataclass
class ModelConfig:
    model_id: str = "openbmb/MiniCPM-o-4_5"
    revision: str = "main"
    seed: int = 42


@dataclass
class SlackGenerationResult:
    tokens: int
    wall_ms: float
    tokens_per_s: float
    remaining_ms: float
    deadline_miss: bool
    max_token_ms: float
    mean_token_ms: float


class MiniCPMODuplexRunner:
    def __init__(self, config: ModelConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this benchmark.")

        self.config = config
        self._seed_all(config.seed)

        # Keep model/duplex generation behavior at upstream defaults. ``torch_dtype='auto'``
        # follows the model config (bf16 for MiniCPM-o 4.5) without overriding generation
        # parameters. Vision/audio/TTS initialization also follows the model config defaults.
        self.model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=True,
            torch_dtype="auto",
        ).eval().cuda()

        # No duplex kwargs: generate_audio, listen/speak decoding, TTS, windowing, etc.
        # all come from MiniCPM-o 4.5's upstream defaults.
        self.duplex = self.model.as_duplex()

        self._slack_cache = None
        self._slack_logits: torch.Tensor | None = None
        self._slack_token_ids: list[int] = []
        self._slack_last_token_ms = 0.0

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_sample(self) -> str:
        self._seed_all(self.config.seed)
        # No custom prompt. Upstream prepare() currently defaults to
        # "Streaming Omni Conversation."
        prompt = self.duplex.prepare()
        self._reset_slack_text_worker()
        return prompt

    def prefill(self, audio_chunk: np.ndarray) -> dict[str, Any]:
        return self.duplex.streaming_prefill(audio_waveform=audio_chunk)

    def generate(self) -> dict[str, Any]:
        # No decode kwargs: use the upstream streaming_generate defaults.
        return self.duplex.streaming_generate()

    def _generation_config(self):
        cfg = getattr(self.model, "generation_config", None)
        if cfg is None:
            cfg = getattr(self.model.llm, "generation_config", None)
        if cfg is None:
            raise RuntimeError("MiniCPM-o generation_config is unavailable.")
        return cfg

    @staticmethod
    def _first_token_id(value: Any) -> int:
        if isinstance(value, (list, tuple)):
            if not value:
                raise RuntimeError("Empty token-id list in generation config.")
            return int(value[0])
        if value is None:
            raise RuntimeError("generation_config.bos_token_id is required for slack token capacity.")
        return int(value)

    def _reset_slack_text_worker(self) -> None:
        """Create an isolated text-generation KV cache using only upstream defaults.

        The worker shares MiniCPM-o's Qwen3 weights but never touches the duplex decoder
        cache. It starts from the model generation config's BOS token, so no benchmark-
        specific text prompt is injected. One unmeasured calibration token is generated
        to obtain a conservative per-token deadline guard for the first measured unit.
        """
        cfg = self._generation_config()
        bos_id = self._first_token_id(getattr(cfg, "bos_token_id", None))
        bos = torch.tensor([[bos_id]], dtype=torch.long, device=self.model.llm.device)

        with torch.inference_mode():
            out = self.model.llm(input_ids=bos, use_cache=True, return_dict=True)
        self._slack_cache = out.past_key_values
        self._slack_logits = out.logits[:, -1, :]
        self._slack_token_ids = [bos_id]
        torch.cuda.synchronize()

        # Calibration is outside the measured real-time timeline and is not counted.
        t0 = time.perf_counter()
        self._slack_step()
        torch.cuda.synchronize()
        self._slack_last_token_ms = (time.perf_counter() - t0) * 1000.0

    def _sample_from_default_config(self, logits: torch.Tensor) -> torch.Tensor:
        cfg = self._generation_config()
        scores = logits.clone()
        do_sample = bool(getattr(cfg, "do_sample", False))

        if not do_sample:
            return torch.argmax(scores, dim=-1)

        temperature = getattr(cfg, "temperature", None)
        if temperature is not None and float(temperature) != 1.0:
            scores = scores / float(temperature)

        top_k = getattr(cfg, "top_k", None)
        if top_k is not None and int(top_k) > 0 and int(top_k) < scores.shape[-1]:
            kth = torch.topk(scores, int(top_k), dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < kth, float("-inf"))

        top_p = getattr(cfg, "top_p", None)
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
            sorted_remove = cumulative_probs > float(top_p)
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
            scores = scores.masked_fill(remove, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def _slack_step(self) -> int:
        if self._slack_logits is None:
            raise RuntimeError("Slack text worker has not been initialized.")

        next_id = self._sample_from_default_config(self._slack_logits)
        input_ids = next_id.reshape(1, 1)
        with torch.inference_mode():
            out = self.model.llm(
                input_ids=input_ids,
                past_key_values=self._slack_cache,
                use_cache=True,
                return_dict=True,
            )
        self._slack_cache = out.past_key_values
        self._slack_logits = out.logits[:, -1, :]
        token_id = int(next_id.item())
        self._slack_token_ids.append(token_id)
        return token_id

    def generate_slack_tokens_until(self, deadline_monotonic_s: float) -> SlackGenerationResult:
        """Consume the remaining real-time budget with isolated text decoding.

        A token is counted only after its decode forward pass has completed. The loop uses
        the last observed token latency as a deadline guard so the benchmark does not
        intentionally steal time from the next 1-second duplex unit.
        """
        start = time.perf_counter()
        token_times_ms: list[float] = []
        tokens = 0

        while True:
            now = time.perf_counter()
            remaining_ms = (deadline_monotonic_s - now) * 1000.0
            predicted_ms = self._slack_last_token_ms if self._slack_last_token_ms > 0 else 1.0
            guard_ms = max(1.0, predicted_ms * 1.10)
            if remaining_ms <= guard_ms:
                break

            token_start = time.perf_counter()
            self._slack_step()
            torch.cuda.synchronize()
            token_end = time.perf_counter()
            token_ms = (token_end - token_start) * 1000.0
            self._slack_last_token_ms = token_ms

            # Only count work completed inside the slack deadline.
            if token_end > deadline_monotonic_s:
                break
            token_times_ms.append(token_ms)
            tokens += 1

        finish = time.perf_counter()
        wall_ms = (finish - start) * 1000.0
        remaining_ms = (deadline_monotonic_s - finish) * 1000.0
        return SlackGenerationResult(
            tokens=tokens,
            wall_ms=wall_ms,
            tokens_per_s=(tokens / (wall_ms / 1000.0)) if wall_ms > 0 and tokens > 0 else 0.0,
            remaining_ms=remaining_ms,
            deadline_miss=remaining_ms < 0.0,
            max_token_ms=max(token_times_ms) if token_times_ms else 0.0,
            mean_token_ms=(sum(token_times_ms) / len(token_times_ms)) if token_times_ms else 0.0,
        )

    def slack_worker_kv_length(self) -> int:
        cache = self._slack_cache
        if cache is None:
            return 0
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                pass
        try:
            return int(cache[0][0].shape[2])
        except Exception:
            return -1

    def upstream_defaults(self) -> dict[str, Any]:
        duplex_defaults = dict(getattr(self.duplex, "_default_duplex_params", {}))
        generation_cfg = self._generation_config()
        return {
            "duplex_prepare_prompt": "upstream default (prepare called with no prompt override)",
            "duplex": duplex_defaults,
            "text_generation": {
                "bos_token_id": getattr(generation_cfg, "bos_token_id", None),
                "eos_token_id": getattr(generation_cfg, "eos_token_id", None),
                "do_sample": getattr(generation_cfg, "do_sample", None),
                "temperature": getattr(generation_cfg, "temperature", None),
                "top_k": getattr(generation_cfg, "top_k", None),
                "top_p": getattr(generation_cfg, "top_p", None),
            },
        }

    def kv_cache_length(self) -> int:
        decoder = getattr(self.duplex, "decoder", None)
        if decoder is None:
            return -1

        for getter_name in ("get_cache_length", "get_kv_cache_length", "_get_kv_cache_length"):
            getter = getattr(decoder, getter_name, None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:
                    pass

        for cache_name in ("cache", "past_key_values", "llm_past_key_values"):
            cache = getattr(decoder, cache_name, None)
            if cache is None:
                continue
            getter = getattr(cache, "get_seq_length", None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:
                    pass

        stats = self.window_stats()
        for key in ("cache_length", "kv_cache_length", "seq_length"):
            if key in stats:
                try:
                    return int(stats[key])
                except (TypeError, ValueError):
                    pass
        return -1

    def window_stats(self) -> dict[str, Any]:
        decoder = getattr(self.duplex, "decoder", None)
        getter = getattr(decoder, "get_window_stats", None) if decoder is not None else None
        if callable(getter):
            try:
                stats = getter()
                return dict(stats) if isinstance(stats, dict) else {"window_stats": stats}
            except Exception:
                return {}
        return {}
