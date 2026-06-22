from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFeatures: # Audio reactive values passed into particle and fluid simulations
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
        # Allocate fixed size particle arrays to avoid per frame object creation
        self.max_particles = max_particles
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros((max_particles, 2), dtype=np.float32)
        self.velocity = np.zeros((max_particles, 2), dtype=np.float32)
        self.age = np.zeros(max_particles, dtype=np.float32)
        self.lifetime = np.ones(max_particles, dtype=np.float32)
        self.energy = np.zeros(max_particles, dtype=np.float32) # Trach which particle slot should be used next for deterministic emmision patterns
        self.alive = np.zeros(max_particles, dtype=bool)
        self._emit_index = 0
        self.time = 0.0 # Seed the system with a small initial burst so the screen is not empty at start

        self.spawn(max_particles // 14, intensity=0.35)
    # Advance the particle simulation by one frame
    def step(
        self,
        dt: float,
        features: AudioFeatures | None = None,
    ) -> None: # Clamp frame time to prevent unstable jumps after lag or pauses
        features = features or AudioFeatures()
        dt = float(np.clip(dt, 0.0, 0.1))
        self.time += dt # Spawn more particles when broadband energy or beat onsets increase

        spawn_count = int(3 + 18 * features.broadband)
        if features.onset:
            spawn_count += int(58 + 98 * features.bass)
        self.spawn(spawn_count, intensity=max(features.intensity, features.bass)) # Skip physics updates when no particles are alive

        active = self.alive
        if not np.any(active):
            return

        pos = self.position[active] # Compute radial and tangent directions for swirl style motion
        vel = self.velocity[active]

        radius = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-4 # Shape particle forces from bass, broadband and overall intensity
        radial = pos / radius
        tangent = np.column_stack((-radial[:, 1], radial[:, 0])).astype(np.float32)

        swirl_strength = 0.45 + 1.45 * features.bass
        inward_strength = 0.22 + 0.42 * features.broadband # Add a slow procedural flow field so particles do not move too uniformly
        lift = np.array([0.0, 0.05 + 0.18 * features.intensity], dtype=np.float32)
        flow = np.column_stack(
            (
                np.sin(pos[:, 1] * 4.2 + self.time * 0.7),
                np.cos(pos[:, 0] * 3.6 - self.time * 0.5),
            )
        ).astype(np.float32) # Combine swirl, inward pull, lift and flow into acceleration

        acceleration = (
            tangent * swirl_strength
            - radial * inward_strength
            + lift
            + flow * (0.10 + 0.28 * features.broadband)
        )
        # Push particles outward on detected beats/onsets
        if features.onset:
            acceleration += radial * (1.75 + 1.55 * features.bass)
        # Apply acceleration, damping and position integration
        vel += acceleration * dt
        vel *= np.exp(-(0.48 + 0.35 * features.broadband) * dt)
        pos += vel * dt

        self.position[active] = pos
        self.velocity[active] = vel

        self.age[active] += dt # Remove particles that leave the visible area or exceed their lifetime
        escaped = np.linalg.norm(self.position, axis=1) > 1.55
        expired = self.age >= self.lifetime
        self.alive &= ~(escaped | expired) # Spawn or recycle particle slots

    def spawn(self, count: int, *, intensity: float = 0.5) -> None:
        count = max(0, int(count))
        if count == 0:
            return # Find currently unused particle slots first

        free = np.flatnonzero(~self.alive) # Resuse the oldest particles when no free slots are available
        if free.size == 0:
            oldest = np.argsort(self.age)[-count:]
            indices = oldest
        else:
            indices = free[:count]

        count = indices.size # Use a golden angle sequence for evenly distributed spawn directions
        sequence = (np.arange(count, dtype=np.float32) + self._emit_index)
        self._emit_index += count
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        angles = sequence * golden_angle + 0.22 * np.sin(sequence * 0.17)
        angles = angles.astype(np.float32) # Add slight radius variation so new particles do not form perfect rings
        radius_band = (sequence % 13.0) / 13.0
        jitter = self.rng.uniform(-0.006, 0.006, count).astype(np.float32)
        radius = (
            0.018
            + radius_band * (0.105 + 0.035 * intensity)
            + jitter
        ).astype(np.float32) # Convert spawn angles into radial and tangent direction vectors
        direction = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        tangent = np.column_stack((-direction[:, 1], direction[:, 0])).astype(np.float32)
        # Compute outward speed from intensity plus deterministic variation
        speed = (
            0.34
            + 0.42 * intensity
            + 0.24 * (0.5 + 0.5 * np.sin(sequence * 0.31))
        ).astype(np.float32)
        spin = ( # Compute tangential spin so particles start with orbital motion
            0.11
            + 0.28 * intensity
            + 0.12 * (0.5 + 0.5 * np.cos(sequence * 0.23))
        ).astype(np.float32)

        self.position[indices] = direction * radius[:, None]
        self.velocity[indices] = direction * speed[:, None] + tangent * spin[:, None]
        self.age[indices] = 0.0 # Initialize position, velocity, lifetime, energy and alive status for spawned particles
        self.lifetime[indices] = (
            1.15 + 1.45 * (0.5 + 0.5 * np.sin(sequence * 0.19))
        ).astype(np.float32)
        self.energy[indices] = np.clip(
            (0.34 + 0.46 * (0.5 + 0.5 * np.cos(sequence * 0.37))) * (0.55 + intensity),
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
    ) -> tuple[np.ndarray, np.ndarray]: # Convert active particles into a boolean canvas and intensity map for rendering
        """Return boolean and intensity canvases sized in braille pixels."""

        width = max(2, int(width))
        height = max(4, int(height))
        intensity = np.zeros((height, width), dtype=np.float32) # Capture an empty intensity field matching the braille pixel render size

        active = self.alive
        if not np.any(active):
            return intensity > threshold, intensity # Return an empty canvas if there are no active particles

        pos = self.position[active]
        age = self.age[active]
        lifetime = self.lifetime[active]
        particle_energy = self.energy[active]

        x = ((pos[:, 0] + 1.15) / 2.30 * (width - 1)).astype(np.int32)
        y = ((1.10 - pos[:, 1]) / 2.20 * (height - 1)).astype(np.int32) # Map normalised simulation coordinates into screen pixel coordinates

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(valid): # Discard particles that fall outside the render area
            return intensity > threshold, intensity

        fade = np.clip(1.0 - age[valid] / lifetime[valid], 0.0, 1.0)
        values = np.clip(particle_energy[valid] * (0.18 + 0.72 * fade), 0.0, 1.0) # Fade particle brightness as it approaches the end of its lifetime
        xv = x[valid]
        yv = y[valid]

        self._splat(intensity, xv, yv, values)
        self._splat(intensity, xv - 1, yv, values * 0.22) # Draw the particle center plus nearby pixels for a soft glow effect
        self._splat(intensity, xv + 1, yv, values * 0.22)
        self._splat(intensity, xv, yv - 1, values * 0.18)
        self._splat(intensity, xv, yv + 1, values * 0.18)

        np.clip(intensity, 0.0, 1.0, out=intensity)
        return intensity > threshold, intensity # Keep intensity values bounded before thresholding into visible pixels

    @staticmethod
    def _splat(
        target: np.ndarray,
        x: np.ndarray,
        y: np.ndarray, # Add particle brightness into the target canvas at valid pixel postions
        value: np.ndarray,
    ) -> None:
        height, width = target.shape
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if np.any(valid):
            np.add.at(target, (y[valid], x[valid]), value[valid])
            # Ignore splat positions outside the canvas bounds