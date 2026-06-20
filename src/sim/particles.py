from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFeatures:
    """Small feature bundle consumed by simulation code."""

    bass: float = 0.0
    broadband: float = 0.0
    onset: bool = False
    intensity: float = 0.0


class ParticleSystem:
    """Numpy-vectorized particle system in normalized simulation space."""

    def __init__(self, max_particles: int = 6000, seed: int | None = None) -> None:
        if max_particles <= 0:
            raise ValueError("max_particles must be positive")

        self.max_particles = max_particles
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros((max_particles, 2), dtype=np.float32)
        self.velocity = np.zeros((max_particles, 2), dtype=np.float32)
        self.age = np.zeros(max_particles, dtype=np.float32)
        self.lifetime = np.ones(max_particles, dtype=np.float32)
        self.energy = np.zeros(max_particles, dtype=np.float32)
        self.alive = np.zeros(max_particles, dtype=bool)

        self.spawn(max_particles // 8, intensity=0.35)

    def step(self, dt: float, features: AudioFeatures | None = None) -> None:
        features = features or AudioFeatures()
        dt = float(np.clip(dt, 0.0, 0.1))

        spawn_count = int(8 + 34 * features.broadband)
        if features.onset:
            spawn_count += int(120 + 180 * features.bass)
        self.spawn(spawn_count, intensity=max(features.intensity, features.bass))

        active = self.alive
        if not np.any(active):
            return

        pos = self.position[active]
        vel = self.velocity[active]

        radius = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-4
        radial = pos / radius
        tangent = np.column_stack((-radial[:, 1], radial[:, 0])).astype(np.float32)

        swirl_strength = 0.45 + 1.45 * features.bass
        inward_strength = 0.16 + 0.30 * features.broadband
        lift = np.array([0.0, 0.10 + 0.28 * features.intensity], dtype=np.float32)

        acceleration = (
            tangent * swirl_strength
            - radial * inward_strength
            + lift
        )

        if features.onset:
            acceleration += radial * (3.2 + 2.4 * features.bass)

        vel += acceleration * dt
        vel *= np.exp(-(0.48 + 0.35 * features.broadband) * dt)
        pos += vel * dt

        self.position[active] = pos
        self.velocity[active] = vel

        self.age[active] += dt
        escaped = np.linalg.norm(self.position, axis=1) > 1.85
        expired = self.age >= self.lifetime
        self.alive &= ~(escaped | expired)

    def spawn(self, count: int, *, intensity: float = 0.5) -> None:
        count = max(0, int(count))
        if count == 0:
            return

        free = np.flatnonzero(~self.alive)
        if free.size == 0:
            oldest = np.argsort(self.age)[-count:]
            indices = oldest
        else:
            indices = free[:count]

        count = indices.size
        angles = self.rng.uniform(0.0, np.pi * 2.0, count).astype(np.float32)
        radius = self.rng.uniform(0.0, 0.12 + 0.06 * intensity, count).astype(np.float32)
        direction = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        tangent = np.column_stack((-direction[:, 1], direction[:, 0])).astype(np.float32)

        speed = self.rng.uniform(0.28, 0.92 + intensity, count).astype(np.float32)
        spin = self.rng.uniform(0.06, 0.36 + intensity * 0.4, count).astype(np.float32)

        self.position[indices] = direction * radius[:, None]
        self.velocity[indices] = direction * speed[:, None] + tangent * spin[:, None]
        self.age[indices] = 0.0
        self.lifetime[indices] = self.rng.uniform(1.4, 3.8, count).astype(np.float32)
        self.energy[indices] = np.clip(
            self.rng.uniform(0.35, 1.0, count) * (0.55 + intensity),
            0.0,
            1.0,
        ).astype(np.float32)
        self.alive[indices] = True

    def rasterize(
        self,
        width: int,
        height: int,
        *,
        threshold: float = 0.11,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return boolean and intensity canvases sized in braille pixels."""

        width = max(2, int(width))
        height = max(4, int(height))
        intensity = np.zeros((height, width), dtype=np.float32)

        active = self.alive
        if not np.any(active):
            return intensity > threshold, intensity

        pos = self.position[active]
        age = self.age[active]
        lifetime = self.lifetime[active]
        particle_energy = self.energy[active]

        x = ((pos[:, 0] + 1.15) / 2.30 * (width - 1)).astype(np.int32)
        y = ((1.10 - pos[:, 1]) / 2.20 * (height - 1)).astype(np.int32)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(valid):
            return intensity > threshold, intensity

        fade = np.clip(1.0 - age[valid] / lifetime[valid], 0.0, 1.0)
        values = np.clip(particle_energy[valid] * (0.25 + fade), 0.0, 1.0)
        xv = x[valid]
        yv = y[valid]

        self._splat(intensity, xv, yv, values)
        self._splat(intensity, xv - 1, yv, values * 0.35)
        self._splat(intensity, xv + 1, yv, values * 0.35)
        self._splat(intensity, xv, yv - 1, values * 0.30)
        self._splat(intensity, xv, yv + 1, values * 0.30)

        np.clip(intensity, 0.0, 1.0, out=intensity)
        return intensity > threshold, intensity

    @staticmethod
    def _splat(
        target: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        value: np.ndarray,
    ) -> None:
        height, width = target.shape
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if np.any(valid):
            np.add.at(target, (y[valid], x[valid]), value[valid])
