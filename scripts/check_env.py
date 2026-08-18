#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-install", action="store_true")
    args = parser.parse_args()

    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pre_install": args.pre_install,
    }

    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "gpu_total_memory_gib": round(props.total_memory / 1024**3, 2),
                    "gpu_capability": list(torch.cuda.get_device_capability(0)),
                }
            )
        if not str(torch.__version__).startswith("2.9.1"):
            print(
                f"[warning] Expected RunPod image PyTorch 2.9.1, found {torch.__version__}.",
                file=sys.stderr,
            )
    except Exception as exc:  # pragma: no cover - environment diagnostic
        info["torch_error"] = repr(exc)

    if not args.pre_install:
        for package in ("transformers", "soundfile", "numpy"):
            try:
                module = __import__(package)
                info[package] = getattr(module, "__version__", "unknown")
            except Exception as exc:
                info[f"{package}_error"] = repr(exc)

    print(json.dumps(info, indent=2, ensure_ascii=False))
    if info.get("torch") and str(info["torch"]).startswith("2.9.1"):
        print(
            "[note] MiniCPM-o upstream currently documents the Transformers path as tested "
            "through torch<=2.8.0; this repository intentionally keeps RunPod torch 2.9.1."
        )


if __name__ == "__main__":
    main()
