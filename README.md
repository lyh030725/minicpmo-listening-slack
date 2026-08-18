# MiniCPM-o 4.5 Listening Slack Benchmark

Measure how much real-time compute slack MiniCPM-o 4.5 leaves while it is naturally in the `LISTEN` state.

The first benchmark uses LibriSpeech ASR `test-clean` utterances longer than 10 seconds and feeds audio in complete 1-second chunks to the official MiniCPM-o duplex API.

## Target RunPod environment

This repository is built for the following RunPod base image:

```text
runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404
```

That image provides CUDA 12.9, PyTorch 2.9.1, and Ubuntu 24.04. The setup script intentionally **does not reinstall PyTorch or torchvision/torchaudio**, so the image's CUDA/PyTorch stack remains intact.

> Upstream MiniCPM-o 4.5 documentation currently lists PyTorch up to 2.8.0 for its tested Transformers path. This benchmark intentionally targets the requested PyTorch 2.9.1 RunPod image and prints a startup compatibility check so failures are obvious rather than silently changing the base environment.

## What is measured

For every complete 1-second audio unit:

1. Wait until the unit's real-time arrival deadline.
2. Call `streaming_prefill(audio_waveform=chunk)`.
3. Call `streaming_generate()` and let MiniCPM-o make its normal LISTEN/SPEAK decision.
4. Measure the whole prefill→generate unit with a single CUDA synchronization at the end, while CUDA events provide per-stage GPU timings.
5. Record the KV-cache length and CUDA memory.
6. Keep `LISTEN` units as the primary slack population; `SPEAK` units are retained for diagnostics.

No forced-listen shortcut is used (`force_listen_count=0`). Speech waveform generation is disabled (`generate_audio=False`) because the target metric is listening slack, not TTS latency.

The current official `as_duplex()` implementation still initializes the TTS stack even when `generate_audio=False`. This repository preserves that behavior so the benchmark represents the official duplex model footprint; `init_tts=False` is used only during the initial `from_pretrained()` call to avoid initializing TTS twice.

### Slack definitions

`compute_slack_ms` isolates the amount of a nominal 1-second budget left after the measured model calls:

```text
compute_slack_ms = 1000 - total_wall_ms
```

`deadline_slack_ms` is the more realistic real-time metric. It includes any accumulated scheduling delay from earlier units:

```text
deadline_slack_ms = next_scheduled_arrival - actual_finish_time
```

A negative `deadline_slack_ms` is a real-time deadline miss.

## Quick start on RunPod

Start a pod using:

```text
runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404
```

Then inside the pod:

```bash
bash scripts/setup_runpod.sh
bash scripts/download_librispeech.sh
bash scripts/run_test_clean.sh
```

By default the benchmark selects the first 100 `test-clean` utterances with duration strictly greater than 10 seconds.

To run every qualifying utterance:

```bash
python -m slackbench.benchmark \
  --dataset-root data/LibriSpeech/test-clean \
  --min-duration 10 \
  --max-samples 0 \
  --realtime \
  --output-dir results/test-clean-gt10-all
```

## Outputs

A run creates:

```text
results/test-clean-gt10/
├── manifest.csv
├── units.csv
├── summary.json
├── environment.json
├── slack_vs_kv.png
└── listen_slack_histogram.png
```

Important `units.csv` columns include:

- `sample_id`, `unit_idx`, `state`
- `prefill_host_ms`, `generate_host_ms`, `total_wall_ms` (`total_wall_ms` is the primary synchronized end-to-end unit latency)
- `prefill_gpu_ms`, `generate_gpu_ms`, `total_gpu_ms`
- `compute_slack_ms`, `deadline_slack_ms`, `deadline_miss`
- MiniCPM internal timing fields such as `cost_audio_process_ms`, `cost_audio_embed_ms`, `cost_audio_feed_ms`, `cost_llm_ms`
- `kv_cache_length`
- CUDA allocated/reserved/peak memory

## Why complete 1-second chunks only?

MiniCPM-o's duplex interface is designed around roughly one call per second. For the initial controlled benchmark, the trailing partial second of each LibriSpeech utterance is dropped so each measured unit has the same 1000 ms real-time budget.

The official implementation has special first-chunk handling internally (roughly 1035 ms of audio context); the benchmark still supplies a 1-second first chunk and leaves that model-specific padding/handling to the upstream implementation.

## Warm-up

Before recording the dataset, the benchmark runs two units from the first selected sample as an unrecorded warm-up by default. Change it with `--warmup-units` or disable with `--warmup-units 0`.

## Reproducibility

The default model revision is `main`. Each run records the requested revision in `environment.json`; use `--model-revision <commit-sha>` to pin an exact upstream revision for a final reproducible experiment.

The default random seed is 42. Samples are sorted deterministically unless `--shuffle` is passed. Sampling/greedy behavior for the LISTEN/SPEAK decision can be selected with `--decode-mode`.

## Repository layout

```text
src/slackbench/
├── benchmark.py   # real-time benchmark loop
├── dataset.py     # LibriSpeech discovery/loading/chunking
├── manifest.py    # duration-filtered manifest generation
├── metrics.py     # summary statistics
├── model.py       # MiniCPM-o 4.5 duplex wrapper
└── timing.py      # synchronized wall/CUDA timers
scripts/
├── setup_runpod.sh
├── download_librispeech.sh
├── run_test_clean.sh
├── check_env.py
└── plot_results.py
```

## Upstream references

- MiniCPM-o 4.5 model: https://huggingface.co/openbmb/MiniCPM-o-4_5
- MiniCPM-o demo: https://github.com/OpenBMB/MiniCPM-o-Demo
- LibriSpeech / OpenSLR 12: https://www.openslr.org/12
