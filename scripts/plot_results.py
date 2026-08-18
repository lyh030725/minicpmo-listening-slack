#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    csv_path = args.run_dir / "units.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    listen = df[df["state"] == "LISTEN"].copy()
    if listen.empty:
        print("[plot] No LISTEN rows; nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(listen["kv_cache_length"], listen["compute_slack_ms"], s=12, alpha=0.5)
    ax.set_xlabel("KV cache length")
    ax.set_ylabel("Compute slack (ms)")
    ax.set_title("MiniCPM-o 4.5 LISTEN slack vs KV cache length")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.run_dir / "slack_vs_kv.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(listen["deadline_slack_ms"].dropna(), bins=40)
    ax.set_xlabel("Deadline slack (ms)")
    ax.set_ylabel("LISTEN units")
    ax.set_title("MiniCPM-o 4.5 LISTEN deadline slack")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.run_dir / "listen_slack_histogram.png", dpi=180)
    plt.close(fig)

    print(f"[plot] Wrote plots to {args.run_dir}")


if __name__ == "__main__":
    main()
