from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import math
import random

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
UNIT_SECONDS = 1.0
UNIT_SAMPLES = TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class AudioSample:
    sample_id: str
    path: Path
    duration_s: float
    sample_rate: int
    frames: int


def discover_librispeech(
    root: Path,
    min_duration_s: float = 0.0,
    max_samples: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> list[AudioSample]:
    """Discover LibriSpeech FLAC files, optionally filtering by minimum duration."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    samples: list[AudioSample] = []
    for path in sorted(root.rglob("*.flac")):
        info = sf.info(str(path))
        duration_s = float(info.frames) / float(info.samplerate)
        if duration_s <= min_duration_s:
            continue
        samples.append(
            AudioSample(
                sample_id=path.stem,
                path=path,
                duration_s=duration_s,
                sample_rate=int(info.samplerate),
                frames=int(info.frames),
            )
        )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    if max_samples and max_samples > 0:
        samples = samples[:max_samples]
    return samples


def load_audio(path: Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    elif audio.ndim != 1:
        raise ValueError(f"Unexpected audio shape for {path}: {audio.shape}")

    if int(sr) != target_sample_rate:
        gcd = math.gcd(int(sr), target_sample_rate)
        up = target_sample_rate // gcd
        down = int(sr) // gcd
        audio = resample_poly(audio, up, down).astype(np.float32, copy=False)

    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def full_second_chunks(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield only complete 1-second chunks; kept for the original benchmark protocol."""
    audio = np.asarray(audio, dtype=np.float32)
    chunk_size = int(sample_rate)
    n_complete = len(audio) // chunk_size
    for unit_idx in range(n_complete):
        start = unit_idx * chunk_size
        end = start + chunk_size
        yield unit_idx, np.ascontiguousarray(audio[start:end], dtype=np.float32)


def realtime_chunks(
    audio: np.ndarray,
    trailing_silence_units: int = 3,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Iterator[tuple[int, np.ndarray, str, int]]:
    """Yield 1-second real-time units using all speech, then explicit trailing silence.

    The final partial speech chunk is zero-padded instead of dropped. This preserves the
    end of the utterance, which is necessary for natural LISTEN -> SPEAK transitions.

    Yields ``(unit_idx, chunk, input_kind, valid_audio_samples)`` where ``input_kind`` is
    ``AUDIO``, ``AUDIO_PADDED``, or ``SILENCE``.
    """
    if trailing_silence_units < 0:
        raise ValueError("trailing_silence_units must be >= 0")

    audio = np.asarray(audio, dtype=np.float32)
    chunk_size = int(sample_rate)
    if chunk_size <= 0:
        raise ValueError("sample_rate must be positive")

    n_audio_units = int(math.ceil(len(audio) / chunk_size)) if len(audio) else 0
    for unit_idx in range(n_audio_units):
        start = unit_idx * chunk_size
        end = min(start + chunk_size, len(audio))
        valid = max(0, end - start)
        if valid == chunk_size:
            chunk = np.ascontiguousarray(audio[start:end], dtype=np.float32)
            kind = "AUDIO"
        else:
            chunk = np.zeros(chunk_size, dtype=np.float32)
            if valid:
                chunk[:valid] = audio[start:end]
            kind = "AUDIO_PADDED"
        yield unit_idx, chunk, kind, valid

    for silence_idx in range(trailing_silence_units):
        unit_idx = n_audio_units + silence_idx
        yield unit_idx, np.zeros(chunk_size, dtype=np.float32), "SILENCE", 0
