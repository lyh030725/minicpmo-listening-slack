from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

from .dataset import AudioSample, discover_librispeech, full_second_chunks, load_audio
from .manifest import write_manifest
from .metrics import summarize
from .model import DEFAULT_SYSTEM_PROMPT, MiniCPMODuplexRunner, ModelConfig
from .timing import sleep_until, timed_duplex_unit


UNIT_BUDGET_MS = 1000.0

CSV_FIELDS = [
    "sample_id",
    "audio_duration_s",
    "unit_idx",
    "audio_start_s",
    "audio_end_s",
    "state",
    "prefill_success",
    "prefill_reason",
    "start_lag_ms",
    "prefill_host_ms",
    "generate_host_ms",
    "total_wall_ms",
    "prefill_gpu_ms",
    "generate_gpu_ms",
    "total_gpu_ms",
    "compute_slack_ms",
    "deadline_slack_ms",
    "slack_ratio",
    "deadline_miss",
    "cost_audio_process_ms",
    "cost_audio_embed_ms",
    "cost_audio_feed_ms",
    "cost_prefill_all_ms",
    "cost_llm_ms",
    "cost_generate_all_ms",
    "n_tokens",
    "n_tts_tokens",
    "kv_cache_length",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "gpu_peak_memory_allocated_mb",
]


def _sec_to_ms(value: Any) -> float:
    try:
        return float(value) * 1000.0
    except (TypeError, ValueError):
        return float("nan")


def _finite_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    env: dict[str, Any] = {
        "created_unix_s": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "benchmark_args": vars(args).copy(),
        "target_runpod_image": "runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404",
    }
    for key, value in list(env["benchmark_args"].items()):
        if isinstance(value, Path):
            env["benchmark_args"][key] = str(value)

    try:
        import transformers

        env["transformers"] = transformers.__version__
    except Exception:
        pass

    try:
        from huggingface_hub import model_info

        info = model_info(args.model_id, revision=args.model_revision)
        env["model_resolved_sha"] = info.sha
    except Exception as exc:
        env["model_resolved_sha_error"] = repr(exc)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
    return env


def warmup(runner: MiniCPMODuplexRunner, sample: AudioSample, units: int) -> None:
    if units <= 0:
        return
    print(f"[warmup] sample={sample.sample_id} units={units}")
    audio = load_audio(sample.path)
    runner.prepare_sample()
    for unit_idx, chunk in full_second_chunks(audio):
        if unit_idx >= units:
            break
        prefill = runner.prefill(chunk)
        if not prefill.get("success", False):
            raise RuntimeError(f"Warm-up prefill failed: {prefill.get('reason', '')}")
        runner.generate()
    torch.cuda.synchronize()


def measure_sample(
    runner: MiniCPMODuplexRunner,
    sample: AudioSample,
    realtime: bool,
    stop_on_speak: bool,
    writer: csv.DictWriter,
    row_sink: list[dict[str, Any]],
) -> None:
    audio = load_audio(sample.path)
    runner.prepare_sample()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    origin = time.perf_counter()

    for unit_idx, chunk in full_second_chunks(audio):
        scheduled_start = origin + float(unit_idx)
        if realtime:
            actual_start = sleep_until(scheduled_start)
        else:
            actual_start = time.perf_counter()
        start_lag_ms = (actual_start - scheduled_start) * 1000.0 if realtime else 0.0

        timed = timed_duplex_unit(
            prefill_fn=lambda: runner.prefill(chunk),
            generate_fn=runner.generate,
        )
        finish = time.perf_counter()
        prefill = timed.prefill
        generated = timed.generated or {}

        total_wall_ms = timed.total_wall_ms
        compute_slack_ms = UNIT_BUDGET_MS - total_wall_ms
        next_deadline = origin + float(unit_idx + 1)
        deadline_slack_ms = (next_deadline - finish) * 1000.0 if realtime else float("nan")

        if not prefill.get("success", False):
            state = "PREFILL_ERROR"
            is_listen = False
        else:
            is_listen = bool(generated.get("is_listen", True))
            state = "LISTEN" if is_listen else "SPEAK"

        row = {
            "sample_id": sample.sample_id,
            "audio_duration_s": sample.duration_s,
            "unit_idx": unit_idx,
            "audio_start_s": float(unit_idx),
            "audio_end_s": float(unit_idx + 1),
            "state": state,
            "prefill_success": bool(prefill.get("success", False)),
            "prefill_reason": prefill.get("reason", ""),
            "start_lag_ms": start_lag_ms,
            "prefill_host_ms": timed.prefill_host_ms,
            "generate_host_ms": timed.generate_host_ms,
            "total_wall_ms": total_wall_ms,
            "prefill_gpu_ms": timed.prefill_gpu_ms,
            "generate_gpu_ms": timed.generate_gpu_ms,
            "total_gpu_ms": timed.total_gpu_ms,
            "compute_slack_ms": compute_slack_ms,
            "deadline_slack_ms": deadline_slack_ms,
            "slack_ratio": compute_slack_ms / UNIT_BUDGET_MS,
            "deadline_miss": bool(realtime and deadline_slack_ms < 0),
            "cost_audio_process_ms": _sec_to_ms(prefill.get("cost_audio_process")),
            "cost_audio_embed_ms": _sec_to_ms(prefill.get("cost_audio_embed")),
            "cost_audio_feed_ms": _sec_to_ms(prefill.get("cost_audio_feed")),
            "cost_prefill_all_ms": _sec_to_ms(prefill.get("cost_all")),
            "cost_llm_ms": _sec_to_ms(generated.get("cost_llm")),
            "cost_generate_all_ms": _sec_to_ms(generated.get("cost_all")),
            "n_tokens": int(generated.get("n_tokens", 0) or 0),
            "n_tts_tokens": int(generated.get("n_tts_tokens", 0) or 0),
            "kv_cache_length": runner.kv_cache_length(),
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "gpu_peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
        writer.writerow(row)
        row_sink.append(row)

        if state == "PREFILL_ERROR":
            break
        if stop_on_speak and state == "SPEAK":
            break


def _subset(rows: Iterable[dict[str, Any]], state: str | None = None) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if state is not None and row.get("state") != state:
            continue
        result.append(row)
    return result


def build_summary(rows: list[dict[str, Any]], realtime: bool) -> dict[str, Any]:
    def stats_for(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "units": len(subset),
            "compute_slack_ms": summarize(row["compute_slack_ms"] for row in subset),
            "deadline_slack_ms": summarize(row["deadline_slack_ms"] for row in subset),
            "total_wall_ms": summarize(row["total_wall_ms"] for row in subset),
            "total_gpu_ms": summarize(row["total_gpu_ms"] for row in subset),
            "kv_cache_length": summarize(row["kv_cache_length"] for row in subset),
            "deadline_miss_rate": (
                sum(bool(row["deadline_miss"]) for row in subset) / len(subset) if realtime and subset else None
            ),
        }

    valid = [row for row in rows if row.get("state") in {"LISTEN", "SPEAK"}]
    listen = _subset(valid, "LISTEN")
    speak = _subset(valid, "SPEAK")
    errors = [row for row in rows if row.get("state") == "PREFILL_ERROR"]
    return {
        "primary_population": "natural LISTEN units (force_listen_count=0)",
        "all": stats_for(valid),
        "listen": stats_for(listen),
        "speak": stats_for(speak),
        "prefill_errors": len(errors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure MiniCPM-o 4.5 natural LISTEN-state slack.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/LibriSpeech/test-clean"))
    parser.add_argument("--min-duration", type=float, default=10.0, help="Strict lower bound in seconds.")
    parser.add_argument("--max-samples", type=int, default=100, help="0 means all qualifying samples.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-clean-gt10"))
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-units", type=int, default=2)
    parser.add_argument(
        "--stop-on-speak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop a sample after its first SPEAK unit so later units are not contaminated by model output.",
    )

    parser.add_argument("--model-id", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument(
        "--model-revision",
        default="main",
        help="Hugging Face model revision. The resolved commit SHA is saved in environment.json.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--decode-mode", default="sampling", choices=["sampling", "greedy"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_librispeech(
        args.dataset_root,
        min_duration_s=args.min_duration,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not samples:
        raise SystemExit(
            f"No FLAC samples found with duration > {args.min_duration}s under {args.dataset_root}. "
            "Run scripts/download_librispeech.sh first."
        )

    write_manifest(samples, args.output_dir / "manifest.csv")
    with (args.output_dir / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(collect_environment(args), f, indent=2, ensure_ascii=False)

    print(f"[benchmark] selected_samples={len(samples)}")
    print(f"[benchmark] realtime={args.realtime} stop_on_speak={args.stop_on_speak}")
    print(f"[benchmark] loading {args.model_id}")

    runner = MiniCPMODuplexRunner(
        ModelConfig(
            model_id=args.model_id,
            revision=args.model_revision,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            decode_mode=args.decode_mode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            listen_prob_scale=args.listen_prob_scale,
            seed=args.seed,
            system_prompt=args.system_prompt,
        )
    )

    warmup(runner, samples[0], args.warmup_units)

    rows: list[dict[str, Any]] = []
    units_csv = args.output_dir / "units.csv"
    with units_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sample in tqdm(samples, desc="LibriSpeech samples"):
            measure_sample(
                runner,
                sample,
                realtime=args.realtime,
                stop_on_speak=args.stop_on_speak,
                writer=writer,
                row_sink=rows,
            )
            f.flush()

    summary = build_summary(rows, realtime=args.realtime)
    summary["selected_samples"] = len(samples)
    summary["completed_rows"] = len(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    listen_stats = summary["listen"]["deadline_slack_ms" if args.realtime else "compute_slack_ms"]
    print(f"[benchmark] rows={len(rows)} listen_units={summary['listen']['units']} speak_units={summary['speak']['units']}")
    print(f"[benchmark] LISTEN slack stats: {json.dumps(listen_stats, ensure_ascii=False)}")
    print(f"[benchmark] wrote {units_csv}")


if __name__ == "__main__":
    main()
