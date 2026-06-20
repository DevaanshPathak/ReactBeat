from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".oga"}


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be decoded for playback."""


@dataclass(frozen=True)
class AudioData:
    path: Path
    samples: np.ndarray
    mono: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frames / self.sample_rate


def load_audio_file(path: str | Path) -> AudioData:
    audio_path = Path(path).expanduser()
    if not audio_path.exists():
        raise AudioLoadError(f"audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioLoadError(f"audio path is not a file: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioLoadError(
            f"unsupported audio extension {audio_path.suffix!r}; supported: {supported}"
        )

    sf = _import_soundfile()
    try:
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except Exception as exc:  # pragma: no cover - exact exception type is backend-specific
        raise AudioLoadError(f"failed to decode audio file {audio_path}: {exc}") from exc

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise AudioLoadError(f"decoded audio has invalid shape: {samples.shape!r}")

    samples = np.nan_to_num(samples, copy=False)
    samples = np.ascontiguousarray(np.clip(samples, -1.0, 1.0), dtype=np.float32)
    mono = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)

    return AudioData(
        path=audio_path,
        samples=samples,
        mono=mono,
        sample_rate=int(sample_rate),
    )


def _import_soundfile() -> Any:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise AudioLoadError(
            "soundfile is required for audio decode. Install project dependencies with "
            '`python -m pip install -e ".[dev]"`.'
        ) from exc
    return sf
