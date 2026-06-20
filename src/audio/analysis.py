from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..sim.particles import AudioFeatures


@dataclass(frozen=True)
class EnergyFrame:
    bass: float
    broadband: float
    onset: bool
    intensity: float
    sample_index: int

    def to_audio_features(self) -> AudioFeatures:
        return AudioFeatures(
            bass=self.bass,
            broadband=self.broadband,
            onset=self.onset,
            intensity=self.intensity,
        )


class EnergyAnalyzer:
    """Manual short-time FFT band-energy analyzer with adaptive onset detection."""

    def __init__(
        self,
        mono: np.ndarray,
        sample_rate: int,
        *,
        window_size: int = 2048,
        hop_size: int = 512,
        history_size: int = 48,
        bass_band: tuple[float, float] = (35.0, 180.0),
        broadband_band: tuple[float, float] = (35.0, 8000.0),
        onset_multiplier: float = 1.55,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if window_size <= 0 or hop_size <= 0:
            raise ValueError("window_size and hop_size must be positive")

        self.mono = np.asarray(mono, dtype=np.float32)
        if self.mono.ndim != 1:
            raise ValueError("mono audio must be a 1D array")

        self.sample_rate = int(sample_rate)
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.onset_multiplier = float(onset_multiplier)
        self._window = np.hanning(self.window_size).astype(np.float32)
        self._freqs = np.fft.rfftfreq(self.window_size, d=1.0 / self.sample_rate)
        self._bass_mask = _band_mask(self._freqs, bass_band)
        self._broadband_mask = _band_mask(self._freqs, broadband_band)
        if not np.any(self._bass_mask):
            raise ValueError("bass band does not overlap FFT bins")
        if not np.any(self._broadband_mask):
            raise ValueError("broadband band does not overlap FFT bins")

        self._bass_history: deque[float] = deque(maxlen=history_size)
        self._broadband_history: deque[float] = deque(maxlen=history_size)
        self._bass_peak = 1e-6
        self._broadband_peak = 1e-6
        self._last_bass = 0.0
        self._last_hop = -1
        self._last_onset_hop = -10_000
        self._cached = EnergyFrame(0.0, 0.0, False, 0.0, 0)
        self._min_onset_gap_hops = max(1, int(0.12 * self.sample_rate / self.hop_size))

    def analyze_at(self, sample_index: int) -> EnergyFrame:
        sample_index = int(np.clip(sample_index, 0, max(len(self.mono) - 1, 0)))
        hop = sample_index // self.hop_size
        if hop == self._last_hop:
            return EnergyFrame(
                bass=self._cached.bass,
                broadband=self._cached.broadband,
                onset=False,
                intensity=self._cached.intensity,
                sample_index=sample_index,
            )

        segment = self._window_at(sample_index)
        spectrum = np.abs(np.fft.rfft(segment * self._window))
        bass_raw = _rms(spectrum[self._bass_mask])
        broadband_raw = _rms(spectrum[self._broadband_mask])

        bass_average = _history_mean(self._bass_history, bass_raw)
        broadband_average = _history_mean(self._broadband_history, broadband_raw)
        bass_ratio = bass_raw / max(bass_average, 1e-6)
        broadband_ratio = broadband_raw / max(broadband_average, 1e-6)

        enough_history = len(self._bass_history) >= min(8, self._bass_history.maxlen or 8)
        onset_gap_ok = hop - self._last_onset_hop >= self._min_onset_gap_hops
        rising = bass_raw > self._last_bass * 1.08
        onset = bool(
            enough_history
            and onset_gap_ok
            and rising
            and bass_ratio >= self.onset_multiplier
        )
        if onset:
            self._last_onset_hop = hop

        self._bass_peak = max(bass_raw, self._bass_peak * 0.995)
        self._broadband_peak = max(broadband_raw, self._broadband_peak * 0.995)
        bass_level = _combine_level(bass_raw, self._bass_peak, bass_ratio)
        broadband_level = _combine_level(
            broadband_raw,
            self._broadband_peak,
            broadband_ratio,
        )
        intensity = float(np.clip(max(bass_level, broadband_level) + 0.35 * onset, 0.0, 1.0))

        self._bass_history.append(bass_raw)
        self._broadband_history.append(broadband_raw)
        self._last_bass = bass_raw
        self._last_hop = hop
        self._cached = EnergyFrame(
            bass=bass_level,
            broadband=broadband_level,
            onset=onset,
            intensity=intensity,
            sample_index=sample_index,
        )
        return self._cached

    def _window_at(self, sample_index: int) -> np.ndarray:
        half = self.window_size // 2
        start = sample_index - half
        end = start + self.window_size
        segment = np.zeros(self.window_size, dtype=np.float32)

        source_start = max(start, 0)
        source_end = min(end, len(self.mono))
        if source_end <= source_start:
            return segment

        target_start = source_start - start
        target_end = target_start + (source_end - source_start)
        segment[target_start:target_end] = self.mono[source_start:source_end]
        return segment


def _band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    return (freqs >= low) & (freqs <= high)


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _history_mean(history: deque[float], fallback: float) -> float:
    if not history:
        return max(fallback, 1e-6)
    return max(float(np.mean(history)), 1e-6)


def _combine_level(raw: float, peak: float, ratio: float) -> float:
    peak_level = raw / max(peak, 1e-6)
    ratio_level = min(ratio / 2.2, 1.0)
    return float(np.clip(0.65 * peak_level + 0.35 * ratio_level, 0.0, 1.0))
