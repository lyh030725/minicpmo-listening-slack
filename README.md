# MiniCPM-o 4.5 Real-Time Slack Token Capacity Benchmark

Measure both **LISTEN** and **SPEAK** real-time compute slack in MiniCPM-o 4.5, then directly measure how many additional text-decoding tokens fit inside that slack before the next 1-second audio deadline.

The benchmark uses the **entire LibriSpeech ASR `test-clean` split by default**, with no minimum-duration filter and no sample-count cap. Audio is streamed to the official MiniCPM-o duplex API in 1-second units. The last partial speech unit is zero-padded instead of dropped, and three 1-second silence units are appended by default so the model gets a natural opportunity to transition from LISTEN to SPEAK.

## Target RunPod environment

```text
runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2404
```

Target stack:

```text
PyTorch: 2.8.0
CUDA runtime: 12.8
Ubuntu: 24.04
```

The setup script intentionally preserves the base image's PyTorch/CUDA stack and verifies that dependency installation does not replace it.

## Listening-focused duplex behavior

The benchmark keeps MiniCPM-o 4.5's upstream duplex generation parameters, but supplies a custom system prompt that strongly favors listening.

```python
LISTENING_SYSTEM_PROMPT = (
    "Streaming Omni Conversation. "
    "You are a patient listener in a real-time full-duplex conversation. "
    "Prioritize listening over speaking. "
    "While the user is speaking or may continue speaking, stay silent and keep listening. "
    "Do not interrupt, backchannel, acknowledge, or respond during the user's utterance. "
    "Only begin speaking after you are confident that the user has clearly finished and a response is needed. "
    "If there is any uncertainty about whether the user has finished, continue listening."
)

model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-4_5",
    trust_remote_code=True,
    torch_dtype="auto",
)
model = model.eval().cuda()

duplex = model.as_duplex()
duplex.prepare(prefix_system_prompt=LISTENING_SYSTEM_PROMPT)
duplex.streaming_prefill(audio_waveform=chunk)
duplex.streaming_generate()
```

In particular:

- The system prompt explicitly asks the model to keep listening while the user is speaking or may continue speaking.
- The prompt asks the model not to interrupt, backchannel, acknowledge, or answer during the utterance.
- The model is asked to speak only after it is confident that the user has clearly finished.
- No LISTEN/SPEAK sampling parameters are overridden in `streaming_generate()`.
- No `listen_prob_scale`, forced-listen, or other LISTEN/SPEAK hyperparameter is changed; the listening bias comes only from the system prompt.
- No `generate_audio=False` or sliding-window override is supplied to `as_duplex()`.
- `torch_dtype="auto"` only asks Transformers to honor the dtype stored in the model config; it does not change generation behavior.
- The requested model revision and its resolved commit SHA are saved in `environment.json`.

### Default-audio caveat

`as_duplex()` enables speech generation by default. However, the upstream Token2Wav path is initialized from a prompt/reference WAV. This benchmark intentionally does **not** invent a reference voice or prompt WAV. As a result, SPEAK units still include the model's default text/TTS-token generation path, while the waveform/Token2Wav portion may remain inactive unless upstream itself provides an initialized prompt audio path. `cost_tts_ms`, `cost_token2wav_ms`, and `n_tts_tokens` are retained so this is visible in the results.

## Dataset protocol

The default dataset settings are:

```text
dataset: LibriSpeech test-clean
minimum duration: 0 seconds
maximum samples: 0 (all samples)
```

Therefore the normal benchmark run processes every non-empty FLAC utterance under `data/LibriSpeech/test-clean`.

The CLI still exposes `--min-duration` and `--max-samples` for optional smaller diagnostic runs, but both default to zero.

## What is measured

For every 1-second real-time unit:

1. Wait for the unit's scheduled arrival time.
2. Run the normal `streaming_prefill()` + `streaming_generate()` duplex work.
3. Record whether MiniCPM-o naturally chose LISTEN or SPEAK under the listening-focused system prompt.
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

The capacity metric is:

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

The benchmark uses all speech in each utterance:

```text
full speech units -> zero-padded final partial unit -> 3 x 1-second silence units
```

It does **not** stop after SPEAK. Change only the number of appended silence units with:

```bash
--trailing-silence-units N
```

This is an input-protocol option, not a model hyperparameter.

## Quick start on RunPod

Create the Pod using:

```text
runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2404
```

Then run:

```bash
bash scripts/setup_runpod.sh
bash scripts/download_librispeech.sh
bash scripts/run_test_clean.sh
```

By default, `run_test_clean.sh` uses:

```bash
MIN_DURATION=0
MAX_SAMPLES=0
```

so the complete `test-clean` split is used.

For a smaller diagnostic run, override either variable explicitly, for example:

```bash
MAX_SAMPLES=100 bash scripts/run_test_clean.sh
```

or:

```bash
MIN_DURATION=10 MAX_SAMPLES=100 bash scripts/run_test_clean.sh
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
- capacity: `slack_text_tokens`, `slack_generation_ms`, `slack_tokens_per_s`, `slack_remaining_ms`
- deadline safety: `deadline_miss_after_slack`
- latency diagnostics: `slack_mean_token_ms`, `slack_max_token_ms`
- duplex internals: `n_tokens`, `n_tts_tokens`, `cost_llm_ms`, `cost_tts_ms`, `cost_token2wav_ms`
- context: `kv_cache_length`, `slack_worker_kv_length`
- CUDA memory usage

`summary.json` reports the slack/token statistics separately for `all`, `listen`, and `speak` populations.

`environment.json` records the benchmark arguments, resolved model revision, and the exact listening-focused system prompt used for the run.

## Input kinds

`input_kind` is one of:

- `AUDIO`: a complete 1-second speech unit.
- `AUDIO_PADDED`: the final partial speech unit, zero-padded to one second; `valid_audio_samples` records the real speech samples.
- `SILENCE`: an appended one-second zero waveform after the utterance.

## Warm-up and reproducibility

Two duplex units are run as an unrecorded warm-up by default. Change this with `--warmup-units`.

The benchmark seed defaults to 42 and is used only for deterministic benchmark/model RNG setup. Model generation parameters themselves come from upstream defaults apart from the custom listening-focused system prompt.

For a final reproducible experiment, pin an exact upstream revision:

```bash
--model-revision <commit-sha>
```

## Repository layout

```text
src/slackbench/
├── benchmark.py   # real-time duplex + slack-capacity loop
├── dataset.py     # LibriSpeech discovery and real-time chunk protocol
├── manifest.py    # manifest generation
├── metrics.py     # summary statistics
├── model.py       # listening-focused duplex runner + isolated text worker
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
