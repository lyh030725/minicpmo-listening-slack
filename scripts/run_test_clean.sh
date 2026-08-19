#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MAX_SAMPLES="${MAX_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-results/test-clean-slack-token-capacity}"

python -m slackbench.benchmark \
  --dataset-root data/LibriSpeech/test-clean \
  --min-duration 10 \
  --max-samples "${MAX_SAMPLES}" \
  --realtime \
  --warmup-units 2 \
  --output-dir "${OUTPUT_DIR}"

python scripts/plot_results.py --run-dir "${OUTPUT_DIR}"
