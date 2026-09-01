#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


VALID_STATES = ("LISTEN", "SPEAK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        help="Optional path for an example-style two-panel PDF summary.",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Write the PDF summary without regenerating the legacy PNG plots.",
    )
    parser.add_argument(
        "--pdf-layout",
        choices=("vertical", "horizontal"),
        default="vertical",
        help="Arrange the two PDF panels vertically or horizontally.",
    )
    args = parser.parse_args()

    if args.pdf_only and args.pdf_output is None:
        parser.error("--pdf-only requires --pdf-output")

    csv_path = args.run_dir / "units.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    valid = df[df["state"].isin(VALID_STATES)].copy()
    if valid.empty:
        print("[plot] No LISTEN/SPEAK rows; nothing to plot.")
        return

    if args.pdf_output is not None:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        if args.pdf_layout == "horizontal":
            fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.05))
            # Match the apparent marker size of the narrower vertical panels.
            marker_size = 12
            label_fontsize = 15
            tick_fontsize = 13
            legend_fontsize = 14
        else:
            fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.5))
            marker_size = 7
            label_fontsize = None
            tick_fontsize = None
            legend_fontsize = None
        colors = {"LISTEN": "tab:blue", "SPEAK": "tab:orange"}

        for state in VALID_STATES:
            subset = valid[valid["state"] == state]
            if subset.empty:
                continue
            axes[0].scatter(
                subset["slack_worker_kv_length"],
                subset["compute_slack_ms"],
                s=marker_size,
                alpha=0.72,
                color=colors[state],
                edgecolors="none",
                label=state,
            )
            axes[1].scatter(
                subset["compute_slack_ms"],
                subset["slack_text_tokens"],
                s=marker_size,
                alpha=0.72,
                color=colors[state],
                edgecolors="none",
                label=state,
            )

        axes[0].set_xlabel("KV Cache Length", fontsize=label_fontsize)
        axes[0].set_ylabel("Slack (ms)", fontsize=label_fontsize)
        axes[1].set_xlabel("Slack (ms)", fontsize=label_fontsize)
        axes[1].set_ylabel("Generated Text Tokens", fontsize=label_fontsize)
        for ax in axes:
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.tick_params(labelsize=tick_fontsize)
            ax.legend(loc="lower right", framealpha=0.9, fontsize=legend_fontsize)

        fig.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
        fig.savefig(args.pdf_output, format="pdf", metadata={"Title": "MiniCPM-o Test Results"})
        plt.close(fig)
        print(f"[plot] Wrote PDF summary to {args.pdf_output}")

        if args.pdf_only:
            return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for state in VALID_STATES:
        subset = valid[valid["state"] == state]
        if subset.empty:
            continue
        ax.scatter(
            subset["deadline_slack_ms"],
            subset["slack_text_tokens"],
            s=12,
            alpha=0.5,
            label=state,
        )
    ax.set_xlabel("Slack budget before next deadline (ms)")
    ax.set_ylabel("Completed slack text tokens")
    ax.set_title("MiniCPM-o 4.5 slack token capacity")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.run_dir / "slack_tokens_vs_budget.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for state in VALID_STATES:
        subset = valid[valid["state"] == state]
        if subset.empty:
            continue
        ax.scatter(
            subset["slack_worker_kv_length"],
            subset["slack_tokens_per_s"],
            s=12,
            alpha=0.5,
            label=state,
        )
    ax.set_xlabel("Slack-worker KV cache length")
    ax.set_ylabel("Slack text throughput (tokens/s)")
    ax.set_title("Slack text throughput vs worker KV length")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.run_dir / "slack_throughput_vs_worker_kv.png", dpi=180)
    plt.close(fig)

    listen = valid[valid["state"] == "LISTEN"].copy()
    if not listen.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(listen["kv_cache_length"], listen["compute_slack_ms"], s=12, alpha=0.5)
        ax.set_xlabel("Duplex KV cache length")
        ax.set_ylabel("Compute slack (ms)")
        ax.set_title("MiniCPM-o 4.5 LISTEN slack vs duplex KV cache length")
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
