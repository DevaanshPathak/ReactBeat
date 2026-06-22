from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".oga"} # Extensions supported by the audio decoder path


class AudioLoadError(RuntimeError): # Custom error type for user facing audio load/decode failures
    """Raised when an audio file cannot be decoded for playback."""


@dataclass(frozen=True)
class AudioData: # Container for decoded stereo/multichannel audio plus a mono analysis track
    path: Path
    samples: np.ndarray
    mono: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int: # Number of channels in the decoded audio buffer
        return int(self.samples.shape[1])

    @property
    def frames(self) -> int: # Number of sample frames in the decoded audio buffer
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float: # Duration derived from frame count and sample rate
        if self.sample_rate <= 0:
            return 0.0
        return self.frames / self.sample_rate


def load_audio_file(path: str | Path) -> AudioData: # Resolve and validate the requested audio file path before decoding
    audio_path = Path(path).expanduser()
    if not audio_path.exists():
        raise AudioLoadError(f"audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioLoadError(f"audio path is not a file: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS: # Rejected unsupported formats early with a clear error message
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioLoadError(
            f"unsupported audio extension {audio_path.suffix!r}; supported: {supported}"
        )

    sf = _import_soundfile() # Import soundfile lazily so missing audio dependencies produce a clean project error
    try:
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True) # Decode audio as float32 and force a consistent 2D shape: frames × channels.
    except Exception as exc:  # pragma: no cover - exact exception type is backend-specific
        raise AudioLoadError(f"failed to decode audio file {audio_path}: {exc}") from exc

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0: # Ensure decoded audio is non-empty and has the expected dimensions.
        raise AudioLoadError(f"decoded audio has invalid shape: {samples.shape!r}")

    samples = np.nan_to_num(samples, copy=False) # Replace NaN/inf values so bad samples do not break playback or analysis
    samples = np.ascontiguousarray(np.clip(samples, -1.0, 1.0), dtype=np.float32) # Clamp samples into the normal audio range and store them contigously
    mono = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32) # Build a mono version of energy analysis while preserving orignal channels

    return AudioData(
        path=audio_path,
        samples=samples,
        mono=mono,
        sample_rate=int(sample_rate),
    )


def _import_soundfile() -> Any:
    try: # Delay importing soundfile until audio loading is actually needed
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise AudioLoadError( # Convert a missing dependency into the same audio loading error type
            "soundfile is required for audio decode. Install project dependencies with "
            '`python -m pip install -e ".[dev]"`.'
        ) from exc
    return sf
