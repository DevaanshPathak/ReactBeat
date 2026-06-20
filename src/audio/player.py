from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

from .loader import AudioData


class AudioPlaybackError(RuntimeError):
    """Raised when callback audio playback cannot be started."""


class AudioPlayer:
    """Callback-driven sounddevice player with thread-safe position tracking."""

    def __init__(self, audio: AudioData) -> None:
        self.audio = audio
        self._position = 0
        self._lock = Lock()
        self._stream: Any | None = None
        self._sd: Any | None = None
        self._playing = False
        self._finished = False

    @property
    def position_samples(self) -> int:
        with self._lock:
            return int(self._position)

    @property
    def position_seconds(self) -> float:
        return self.position_samples / self.audio.sample_rate

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def finished(self) -> bool:
        with self._lock:
            return bool(self._finished)

    def start(self) -> None:
        if self._finished:
            self.seek(0)

        if self._stream is None:
            self._sd = _import_sounddevice()
            try:
                self._stream = self._sd.OutputStream(
                    samplerate=self.audio.sample_rate,
                    channels=self.audio.channels,
                    dtype="float32",
                    callback=self._callback,
                )
            except Exception as exc:  # pragma: no cover - host audio backend-specific
                raise AudioPlaybackError(f"failed to create audio stream: {exc}") from exc

        try:
            self._stream.start()
        except Exception as exc:  # pragma: no cover - host audio backend-specific
            raise AudioPlaybackError(f"failed to start audio stream: {exc}") from exc
        self._playing = True

    def pause(self) -> None:
        if self._stream is not None:
            self._stream.stop()
        self._playing = False

    def resume(self) -> None:
        if not self._finished:
            self.start()

    def toggle_pause(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.resume()

    def seek(self, sample_index: int) -> None:
        sample_index = int(np.clip(sample_index, 0, self.audio.frames))
        with self._lock:
            self._position = sample_index
            self._finished = sample_index >= self.audio.frames

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._playing = False

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        del time_info, status

        with self._lock:
            start = self._position
            end = min(start + frames, self.audio.frames)
            self._position = end
            if end >= self.audio.frames:
                self._finished = True

        chunk = self.audio.samples[start:end]
        written = int(chunk.shape[0])
        if written:
            outdata[:written, :] = chunk
        if written < frames:
            outdata[written:frames, :] = 0.0
            self._playing = False
            if self._sd is not None:
                raise self._sd.CallbackStop


def _import_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise AudioPlaybackError(
            "sounddevice is required for playback. Install project dependencies with "
            '`python -m pip install -e ".[dev]"`.'
        ) from exc
    return sd
