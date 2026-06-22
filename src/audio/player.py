from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

from .loader import AudioData


class AudioPlaybackError(RuntimeError): # Custom error type for playback setup/start failures
    """Raised when callback audio playback cannot be started."""


class AudioPlayer: # Manages audio playback through a sounddevice callback stream
    """Callback-driven sounddevice player with thread-safe position tracking."""

    def __init__(self, audio: AudioData) -> None:
        self.audio = audio
        self._position = 0 # Current playback position, stored as a sample/frame index 
        self._lock = Lock() # Protect playback position because the audio callback runs on another thread
        self._stream: Any | None = None # Output stream is created lazily when playback starts
        self._sd: Any | None = None # Store the imported sounddevice module so the callback can raise CallbackStop
        self._playing = False
        self._finished = False

    @property
    def position_samples(self) -> int: # Return the current playback position safely across threads
        with self._lock:
            return int(self._position)

    @property
    def position_seconds(self) -> float:# Convert the current sample position into seconds
        return self.position_samples / self.audio.sample_rate

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def finished(self) -> bool:
        with self._lock:
            return bool(self._finished)

    def start(self) -> None: # Start playback, creating the output stream on the first use
        if self._finished: # Restart from beginning if playback had already reached the end
            self.seek(0)

        if self._stream is None: # Lazily create the sounddevice stream only once
            self._sd = _import_sounddevice()
            try:
                self._stream = self._sd.OutputStream( # Send audio frames to sounddevice through callback
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

    def pause(self) -> None: # Stop the stream without resetting the playback position
        if self._stream is not None:
            self._stream.stop()
        self._playing = False

    def resume(self) -> None: # Resume playback only if the track has not finished
        if not self._finished:
            self.start()

    def toggle_pause(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.resume()

    def seek(self, sample_index: int) -> None: # Move playback to a specific sample, clamped to the audio length
        sample_index = int(np.clip(sample_index, 0, self.audio.frames))
        with self._lock:
            self._position = sample_index
            self._finished = sample_index >= self.audio.frames

    def close(self) -> None: # Release the audio stream and mark playback as stopped
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
        del time_info, status # Ignore callback metadata that is not needed by this player

        with self._lock: # Advance playback position atomically before writing audio output
            start = self._position
            end = min(start + frames, self.audio.frames)
            self._position = end
            if end >= self.audio.frames:
                self._finished = True

        chunk = self.audio.samples[start:end] # Slice the next chunk of audio requested by the output device
        written = int(chunk.shape[0])
        if written: # Copy available audio samples into the output buffer
            outdata[:written, :] = chunk
        if written < frames: # Fill the remaining output frames with silence when audio runs out
            outdata[written:frames, :] = 0.0
            self._playing = False
            if self._sd is not None:
                raise self._sd.CallbackStop # Tell sounddevice to stop calling back after the final buffer


def _import_sounddevice() -> Any: # Import sounddevice only when playback is actually needed
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise AudioPlaybackError(
            "sounddevice is required for playback. Install project dependencies with "
            '`python -m pip install -e ".[dev]"`.'
        ) from exc
    return sd
