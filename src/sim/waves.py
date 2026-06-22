from __future__ import annotations

import numpy as np

from .particles import AudioFeatures


class WaveSimulation:
    """Damped 2D wave equation solver for beat-reactive ripple fields."""

    def __init__(
        self,
        width: int = 160,
        height: int = 96,
        *,
        damping: float = 0.972,
        wave_speed: float = 0.235, # Store wave tuning values and initialize the simulation grid
    ) -> None:
        self.damping = float(damping)
        self.wave_speed = float(wave_speed)
        self.time = 0.0
        self.ensure_size(width, height) # Resize or initialize wave buffers when the render size changes
    # Keep a minimum grid size so interior slice wave updates remain valid.
    def ensure_size(self, width: int, height: int) -> None:
        width = max(12, int(width)) # Avoid reallocating buffers when the requested size is unchanged
        height = max(12, int(height))
        if getattr(self, "width", None) == width and getattr(self, "height", None) == height:
            return

        self.width = width
        self.height = height # Current and previous wave fields used by the finite difference solver
        shape = (height, width)
        self.current = np.zeros(shape, dtype=np.float32)
        self.previous = np.zeros(shape, dtype=np.float32) # Precompute coordinate grids used for ripple injection
        self.next_field = np.zeros(shape, dtype=np.float32)
        self.yy, self.xx = np.mgrid[0:height, 0:width].astype(np.float32) # Advance the wave simulation by one frame

    def step(self, dt: float, features: AudioFeatures | None = None) -> None: # Clamp frame time to large pauses do not destablize the solver
        features = features or AudioFeatures()
        dt = float(np.clip(dt, 0.0, 0.08))
        if dt == 0.0:
            return
        # Inject audio reactive ripples sources before solving the next wave state
        self.time += dt
        self._inject(features) # Compute the 2D Laplacian from neighbouring cells

        center = self.current[1:-1, 1:-1]
        laplacian = (
            self.current[1:-1, 2:]
            + self.current[1:-1, :-2]
            + self.current[2:, 1:-1]
            + self.current[:-2, 1:-1]
            - 4.0 * center
        ) # Apply the demped wave equation to update the next field
        self.next_field.fill(0.0)
        self.next_field[1:-1, 1:-1] = (
            (2.0 * center - self.previous[1:-1, 1:-1])
            + laplacian * self.wave_speed
        ) * self.damping # Add extra ripple energy on detected beats/onsets

        if features.onset: # Fade and clamp the field to keep wave amplitudes bounded
            self.next_field[1:-1, 1:-1] += laplacian * (0.015 + 0.035 * features.bass)
        # Rotate wave buffers so that the next frame can reuse the arrays
        self.next_field *= np.exp(-(0.36 + 0.16 * features.broadband) * dt)
        np.clip(self.next_field, -1.25, 1.25, out=self.next_field)
        self.previous, self.current, self.next_field = (
            self.current,
            self.next_field,
            self.previous,
        )

    def rasterize(
        self,
        width: int, # Convert the wave field into a boolean canvas and intensity map for rendering
        height: int,
        *,
        threshold: float = 0.08,
    ) -> tuple[np.ndarray, np.ndarray]:
        width = max(2, int(width))
        height = max(4, int(height)) # Resample the simulation field if render dimensions differ
        field = self.current
        if field.shape != (height, width):
            field = self._resample(field, width, height) # Highlight ripple edges more strongly than flat wave regions

        gradient = self._edge_detail(field)
        rings = np.abs(field)
        intensity = np.clip(rings * 0.16 + gradient * 4.20, 0.0, 1.0) # Inject moving drops and wave bands based on audio features
        return intensity > threshold, intensity

    def _inject(self, features: AudioFeatures) -> None:
        bass = float(features.bass)
        broadband = float(features.broadband)
        intensity = float(features.intensity)
        span = min(self.width, self.height) # Primary positive ripple source driven by bass and intensity
        t = self.time

        self._add_drop(
            self.width * (0.50 + 0.35 * np.sin(t * 0.47)),
            self.height * (0.50 + 0.25 * np.cos(t * 0.61)),
            span * (0.035 + 0.014 * bass), # Secondary negative ripple source driven by broadband energy
            0.030 + 0.085 * intensity + 0.075 * bass,
        )
        self._add_drop(
            self.width * (0.50 + 0.38 * np.sin(t * 0.31 + 2.20)),
            self.height * (0.50 + 0.31 * np.sin(t * 0.43 + 1.10)),
            span * (0.026 + 0.012 * broadband), # Add a subtle moving live wave for continuous background motion
            -(0.020 + 0.060 * broadband),
        )
        self._add_line(
            t,
            0.006 + 0.028 * broadband, # Add a stronger ripple burst when an onset/beat is detected
        )

        if features.onset:
            self._add_drop(
                self.width * (0.50 + 0.28 * np.sin(t * 1.70)),
                self.height * (0.52 + 0.26 * np.cos(t * 1.35)),
                span * (0.060 + 0.020 * bass),
                0.17 + 0.24 * bass, # Add a circular ripple into the current wave field
            )

    def _add_drop(self, cx: float, cy: float, radius: float, amount: float) -> None:
        radius = max(2.0, float(radius))
        x0 = max(1, int(cx - radius))
        x1 = min(self.width - 1, int(cx + radius) + 1)
        y0 = max(1, int(cy - radius)) # Clamp the ripple patch to valid interior cells
        y1 = min(self.height - 1, int(cy + radius) + 1)
        if x0 >= x1 or y0 >= y1:
            return

        dx = (self.xx[y0:y1, x0:x1] - cx) / radius
        dy = (self.yy[y0:y1, x0:x1] - cy) / radius
        dist2 = dx * dx + dy * dy # Build a soft circular ripple mask that fades towards the edge
        mask = np.maximum(0.0, 1.0 - dist2)
        ripple = np.cos(np.sqrt(dist2) * np.pi) * mask * mask
        self.current[y0:y1, x0:x1] += ripple.astype(np.float32) * float(amount) # Apply the ripple contribution to the selected path

    def _add_line(self, t: float, amount: float) -> None: # Add a moving horizontal wave band across the field
        y = self.height * (0.52 + 0.28 * np.sin(t * 0.37))
        width = max(2.0, self.height * 0.035)
        phase = self.xx * 0.12 + t * 2.0
        wave = np.sin(phase)
        distance = np.abs(self.yy - y - wave * self.height * 0.035)
        band = np.maximum(0.0, 1.0 - distance / width)
        self.current += (band * band * np.sin(phase * 1.7)).astype(np.float32) * float(amount)
    # Nearest neighbour resampling for matching render dimensions
    @staticmethod
    def _resample(field: np.ndarray, width: int, height: int) -> np.ndarray:
        y_indices = np.linspace(0, field.shape[0] - 1, height).astype(np.int32)
        x_indices = np.linspace(0, field.shape[1] - 1, width).astype(np.int32)
        return field[y_indices[:, None], x_indices[None, :]]
    # Estimate visual edge detail using local wave differences
    @staticmethod
    def _edge_detail(field: np.ndarray) -> np.ndarray:
        detail = np.zeros_like(field, dtype=np.float32)
        detail[:, 1:-1] += np.abs(field[:, 2:] - field[:, :-2])
        detail[1:-1, :] += np.abs(field[2:, :] - field[:-2, :])
        return detail
