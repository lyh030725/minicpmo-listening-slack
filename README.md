# MiniCPM-o 4.5 Real-Time Slack Token Capacity Benchmark

Measure both **LISTEN** and **SPEAK** real-time compute slack in MiniCPM-o 4.5, then directly measure how many additional text-decoding tokens fit inside that slack before the next 1-second audio deadline.

The benchmark uses LibriSpeech ASR `test-clean` utterances longer than 10 seconds. Audio is streamed to the official MiniCPM-o duplex API in 1-second units. The last partial speech unit is zero-padded instead of dropped, and three 1-second silence units are appended by default so the model gets a natural opportunity to transition from LISTEN to SPEAK.

## Target RunPod environment

```text
runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404
```

The setup script intentionally does not reinstall the base image's PyTorch/CUDA stack.

## Upstream-default model behavior

This benchmark deliberately removes the previous prompt and duplex-decoding overrides.

The measured duplex path now does the following:

```python
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-4_5",
    trust_remote_code=True,
    torch_dtype="auto",
)
model = model.eval().cuda()

duplex = model.as_duplex()
duplex.prepare()
duplex.streaming_prefill(audio_waveform=chunk)
duplex.streaming_generate()
```

In particular:

- No custom system prompt is supplied. `duplex.prepare()` therefore uses MiniCPM-o's own default duplex prompt.
- No LISTEN/SPEAK sampling parameters are supplied to `streaming_generate()`.
- No `generate_audio=False`, forced-listen, or sliding-window override is supplied to `as_duplex()`.
- `torch_dtype="auto"` only asks Transformers to honor the dtype stored in the model config; it does not change generation behavior.
- The requested model revision and its resolved commit SHA are saved in `environment.json`.

### Default-audio caveat

`as_duplex()` enables speech generation by default. However, the upstream Token2Wav path is initialized from a prompt/reference WAV. This benchmark intentionally does **not** invent a reference voice or prompt WAV, because doing so would no longer be a default setup. As a result, SPEAK units still include the model's default text/TTS-token generation path, while the waveform/Token2Wav portion may remain inactive unless upstream itself provides an initialized prompt audio path. `cost_tts_ms`, `cost_token2wav_ms`, and `n_tts_tokens` are retained so this is visible in the results.

## What is measured

For every 1-second real-time unit:

1. Wait for the unit's scheduled arrival time.
2. Run the normal `streaming_prefill()` + `streaming_generate()` duplex work.
3. Record whether MiniCPM-o naturally chose LISTEN or SPEAK.
4. Compute the time remaining until the next 1-second deadline.
5. Use that remaining time for isolated autoregressive text decoding with the **same MiniCPM-o language-model weights**.
6. Stop before the next audio deadline and count only text tokens whose GPU forward pass actually completed in time.
7. Feed the next audio unit to the original duplex state.

### Slack definitions

The original compute metric is retained:

```text
compute_slack_ms = 1000 - duplex_total_wall_ms
```

The primary real-time budget is:

```text
deadline_slack_ms = next_audio_deadline - duplex_finish_time
```

A negative `deadline_slack_ms` means the normal duplex path already missed its deadline.

The new capacity metric is:

```text
slack_text_tokens = number of extra text-decode tokens completed before next_audio_deadline
```

and:

```text
slack_tokens_per_s = slack_text_tokens / slack_generation_time
```

## Why the slack text worker has a separate KV cache

The capacity worker shares the already-loaded MiniCPM-o language-model **weights**, but it does not append its tokens to the duplex conversation cache.

```text
                     MiniCPM-o weights
                            |
             +--------------+--------------+
             |                             |
       Duplex KV cache               Slack-worker KV cache
             |                             |
     audio -> LISTEN/SPEAK          extra text decoding
             |                             |
             +---------- same GPU ---------+
```

This isolation is important. If benchmark-only tokens were appended to the duplex cache, the next LISTEN/SPEAK decision and conversation state would be changed by the measurement itself.

The slack worker starts once per LibriSpeech sample from the BOS token defined by the model's own generation config and samples using that generation config. It is intentionally a **compute-capacity probe**, not a semantic summarization task. EOS does not terminate the probe; the goal is to measure sustained incremental text-decoding capacity.

A single calibration decode is performed outside the timed audio loop. During each slack interval the worker synchronizes CUDA after every completed token and uses a small guard based on the previous token latency so it does not intentionally start a token that is expected to cross the next audio deadline. `deadline_miss_after_slack` still records any actual overrun.

## Speech-end protocol

The old benchmark dropped the final partial second of every utterance and stopped a sample at the first SPEAK unit. That made natural SPEAK observations rare.

The new default protocol is:

```text
full speech units -> zero-padded final partial unit -> 3 x 1-second silence units
```

and it does **not** stop after SPEAK. Change only the number of appended silence units with:

```bash
--trailing-silence-units N
```

This is an input-protocol option, not a model hyperparameter.

## Quick start on RunPod

```bash
bash scripts/setup_runpod.sh
bash scripts/download_librispeech.sh
bash scripts/run_test_clean.sh
```

By default, the benchmark selects the first 100 `test-clean` utterances with duration strictly greater than 10 seconds and writes to:

```text
results/test-clean-slack-token-capacity/
```

Run every qualifying utterance with:

```bash
python -m slackbench.benchmark \
  --dataset-root data/LibriSpeech/test-clean \
  --min-duration 10 \
  --max-samples 0 \
  --realtime \
  --output-dir results/test-clean-slack-token-capacity-all
```

## Outputs

A run creates:

```text
results/test-clean-slack-token-capacity/
├── manifest.csv
├── units.csv
├── summary.json
├── environment.json
├── slack_tokens_vs_budget.png
├── slack_throughput_vs_worker_kv.png
├── slack_vs_kv.png
└── listen_slack_histogram.png
```

Important `units.csv` columns include:

- identity/input: `sample_id`, `unit_idx`, `input_kind`, `valid_audio_samples`, `state`
- duplex timing: `prefill_host_ms`, `generate_host_ms`, `total_wall_ms`
- GPU timing: `prefill_gpu_ms`, `generate_gpu_ms`, `total_gpu_ms`
- original slack: `compute_slack_ms`, `deadline_slack_ms`, `deadline_miss`
- new capacity: `slack_text_tokens`, `slack_generation_ms`, `slack_tokens_per_s`, `slack_remaining_ms`
- deadline safety: `deadline_miss_after_slack`
- latency diagnostics: `slack_mean_token_ms`, `slack_max_token_ms`
- duplex internals: `n_tokens`, `n_tts_tokens`, `cost_llm_ms`, `cost_tts_ms`, `cost_token2wav_ms`
- context: `kv_cache_length`, `slack_worker_kv_length`
- CUDA memory usage

`summary.json` reports the slack/token statistics separately for `all`, `listen`, and `speak` populations.

## Input kinds

`input_kind` is one of:

- `AUDIO`: a complete 1-second speech unit.
- `AUDIO_PADDED`: the final partial speech unit, zero-padded to one second; `valid_audio_samples` records the real speech samples.
- `SILENCE`: an appended one-second zero waveform after the utterance.

## Warm-up and reproducibility

Two duplex units are run as an unrecorded warm-up by default. Change this with `--warmup-units`.

The benchmark seed defaults to 42 and is used only for deterministic benchmark/model RNG setup. Model generation parameters themselves come from upstream defaults.

For a final reproducible experiment, pin an exact upstream revision:

```bash
--model-revision <commit-sha>
```

## Repository layout

```text
src/slackbench/
├── benchmark.py   # real-time duplex + slack-capacity loop
├── dataset.py     # LibriSpeech discovery and real-time chunk protocol
├── manifest.py    # duration-filtered manifest generation
├── metrics.py     # summary statistics
├── model.py       # upstream-default duplex runner + isolated text worker
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
