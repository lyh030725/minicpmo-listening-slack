from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from transformers import AutoModel


DEFAULT_SYSTEM_PROMPT = (
    "You are in a real-time full-duplex spoken conversation. "
    "Listen while the user is speaking and do not interrupt. "
    "Respond only after the user has clearly finished speaking."
)


@dataclass
class ModelConfig:
    model_id: str = "openbmb/MiniCPM-o-4_5"
    revision: str = "main"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    decode_mode: str = "sampling"
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.8
    listen_prob_scale: float = 1.0
    seed: int = 42
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


def _torch_dtype(name: str) -> torch.dtype:
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


class MiniCPMODuplexRunner:
    def __init__(self, config: ModelConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this benchmark.")

        self.config = config
        self._seed_all(config.seed)

        self.model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=True,
            attn_implementation=config.attn_implementation,
            torch_dtype=_torch_dtype(config.dtype),
            init_vision=False,
            init_audio=True,
            # as_duplex() initializes the TTS stack internally in the current official implementation.
            # Keeping this False avoids eagerly initializing it twice during from_pretrained.
            init_tts=True,
            low_cpu_mem_usage=True,
        ).eval().cuda()

        self.duplex = self.model.as_duplex(
            generate_audio=False,
            force_listen_count=0,
            sliding_window_mode="off",
        )

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_sample(self) -> str:
        self._seed_all(self.config.seed)
        return self.duplex.prepare(prefix_system_prompt=self.config.system_prompt)

    def prefill(self, audio_chunk: np.ndarray) -> dict[str, Any]:
        return self.duplex.streaming_prefill(audio_waveform=audio_chunk)

    def generate(self) -> dict[str, Any]:
        return self.duplex.streaming_generate(
            decode_mode=self.config.decode_mode,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            listen_prob_scale=self.config.listen_prob_scale,
        )

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
