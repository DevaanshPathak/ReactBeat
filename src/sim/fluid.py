from __future__ import annotations

import numpy as np

from .particles import AudioFeatures


class FluidSimulation:
    """Coarse Stable Fluids-style solver for terminal-density fields."""

    def __init__(
        self,
        width: int = 160,
        height: int = 96,
        *,
        diffusion: float = 0.00005,
        viscosity: float = 0.00002,
        iterations: int = 8,
    ) -> None:
        self.diffusion = float(diffusion)
        self.viscosity = float(viscosity)
        self.iterations = int(iterations)
        self.time = 0.0
        self.ensure_size(width, height)

    def ensure_size(self, width: int, height: int) -> None:
        width = max(8, int(width))
        height = max(8, int(height))
        if getattr(self, "width", None) == width and getattr(self, "height", None) == height:
            return

        self.width = width
        self.height = height
        shape = (height, width)
        self.density = np.zeros(shape, dtype=np.float32)
        self.density_prev = np.zeros(shape, dtype=np.float32)
        self.vx = np.zeros(shape, dtype=np.float32)
        self.vy = np.zeros(shape, dtype=np.float32)
        self.vx_prev = np.zeros(shape, dtype=np.float32)
        self.vy_prev = np.zeros(shape, dtype=np.float32)
        self.pressure = np.zeros(shape, dtype=np.float32)
        self.divergence = np.zeros(shape, dtype=np.float32)
        self.yy, self.xx = np.mgrid[0:height, 0:width].astype(np.float32)

    def step(self, dt: float, features: AudioFeatures | None = None) -> None:
        features = features or AudioFeatures()
        dt = float(np.clip(dt, 0.0, 0.08))
        if dt == 0.0:
            return

        self.time += dt
        self._inject(features)

        self.vx_prev[:] = self.vx
        self.vy_prev[:] = self.vy
        self._diffuse(1, self.vx, self.vx_prev, self.viscosity, dt)
        self._diffuse(2, self.vy, self.vy_prev, self.viscosity, dt)
        self._project(self.vx, self.vy, self.pressure, self.divergence)

        self.vx_prev[:] = self.vx
        self.vy_prev[:] = self.vy
        self._advect(1, self.vx, self.vx_prev, self.vx_prev, self.vy_prev, dt)
        self._advect(2, self.vy, self.vy_prev, self.vx_prev, self.vy_prev, dt)
        self._project(self.vx, self.vy, self.pressure, self.divergence)

        self.density_prev[:] = self.density
        self._diffuse(0, self.density, self.density_prev, self.diffusion, dt)
        self.density_prev[:] = self.density
        self._advect(0, self.density, self.density_prev, self.vx, self.vy, dt)

        self.density *= np.exp(-0.42 * dt)
        self.vx *= np.exp(-0.08 * dt)
        self.vy *= np.exp(-0.08 * dt)
        np.clip(self.density, 0.0, 1.4, out=self.density)
        np.clip(self.vx, -3.0, 3.0, out=self.vx)
        np.clip(self.vy, -3.0, 3.0, out=self.vy)

    def rasterize(
        self,
        width: int,
        height: int,
        *,
        threshold: float = 0.08,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.ensure_size(width, height)
        intensity = np.clip(self.density, 0.0, 1.0)
        return intensity > threshold, intensity

    def _inject(self, features: AudioFeatures) -> None:
        cx = self.width * (0.50 + 0.12 * np.sin(self.time * 0.9))
        cy = self.height * (0.54 + 0.10 * np.cos(self.time * 0.7))
        rx = max(3.0, self.width * (0.060 + 0.045 * features.bass))
        ry = max(3.0, self.height * (0.070 + 0.035 * features.broadband))

        dx = (self.xx - cx) / rx
        dy = (self.yy - cy) / ry
        blob = np.exp(-(dx * dx + dy * dy) * 2.6).astype(np.float32)

        density_amount = 0.18 + 0.55 * features.intensity + 0.55 * features.bass
        if features.onset:
            density_amount += 0.85
        self.density += blob * density_amount

        swirl = 0.26 + 1.15 * features.bass
        self.vx += -dy * blob * swirl
        self.vy += dx * blob * swirl

        if features.onset:
            radial = 1.15 + 1.65 * features.bass
            self.vx += dx * blob * radial
            self.vy += dy * blob * radial

    def _diffuse(
        self,
        boundary: int,
        target: np.ndarray,
        source: np.ndarray,
        amount: float,
        dt: float,
    ) -> None:
        scale = max(self.width, self.height)
        a = dt * amount * scale * scale
        self._linear_solve(boundary, target, source, a, 1.0 + 4.0 * a)

    def _linear_solve(
        self,
        boundary: int,
        target: np.ndarray,
        source: np.ndarray,
        a: float,
        c: float,
    ) -> None:
        for _ in range(self.iterations):
            target[1:-1, 1:-1] = (
                source[1:-1, 1:-1]
                + a
                * (
                    target[1:-1, 2:]
                    + target[1:-1, :-2]
                    + target[2:, 1:-1]
                    + target[:-2, 1:-1]
                )
            ) / c
            self._set_boundary(boundary, target)

    def _project(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        pressure: np.ndarray,
        divergence: np.ndarray,
    ) -> None:
        divergence[1:-1, 1:-1] = -0.5 * (
            (vx[1:-1, 2:] - vx[1:-1, :-2]) / self.width
            + (vy[2:, 1:-1] - vy[:-2, 1:-1]) / self.height
        )
        pressure.fill(0.0)
        self._set_boundary(0, divergence)
        self._set_boundary(0, pressure)
        self._linear_solve(0, pressure, divergence, 1.0, 4.0)

        vx[1:-1, 1:-1] -= 0.5 * self.width * (
            pressure[1:-1, 2:] - pressure[1:-1, :-2]
        )
        vy[1:-1, 1:-1] -= 0.5 * self.height * (
            pressure[2:, 1:-1] - pressure[:-2, 1:-1]
        )
        self._set_boundary(1, vx)
        self._set_boundary(2, vy)

    def _advect(
        self,
        boundary: int,
        target: np.ndarray,
        source: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        dt: float,
    ) -> None:
        y, x = np.mgrid[1:self.height - 1, 1:self.width - 1]
        back_x = np.clip(x - dt * (self.width - 2) * vx[1:-1, 1:-1], 0.5, self.width - 1.5)
        back_y = np.clip(y - dt * (self.height - 2) * vy[1:-1, 1:-1], 0.5, self.height - 1.5)

        x0 = np.floor(back_x).astype(np.int32)
        y0 = np.floor(back_y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        sx = back_x - x0
        sy = back_y - y0
        target[1:-1, 1:-1] = (
            (1.0 - sx) * ((1.0 - sy) * source[y0, x0] + sy * source[y1, x0])
            + sx * ((1.0 - sy) * source[y0, x1] + sy * source[y1, x1])
        )
        self._set_boundary(boundary, target)

    @staticmethod
    def _set_boundary(boundary: int, target: np.ndarray) -> None:
        if boundary == 1:
            target[:, 0] = -target[:, 1]
            target[:, -1] = -target[:, -2]
        else:
            target[:, 0] = target[:, 1]
            target[:, -1] = target[:, -2]

        if boundary == 2:
            target[0, :] = -target[1, :]
            target[-1, :] = -target[-2, :]
        else:
            target[0, :] = target[1, :]
            target[-1, :] = target[-2, :]

        target[0, 0] = 0.5 * (target[1, 0] + target[0, 1])
        target[0, -1] = 0.5 * (target[1, -1] + target[0, -2])
        target[-1, 0] = 0.5 * (target[-2, 0] + target[-1, 1])
        target[-1, -1] = 0.5 * (target[-2, -1] + target[-1, -2])
