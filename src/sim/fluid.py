from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .particles import AudioFeatures


@dataclass(frozen=True) # Tunable constants that shape how each visual style behaves in the fluid solver
class FluidProfile:
    density_scale: float
    radius_scale: float
    swirl_scale: float
    radial_scale: float
    decay: float
    fill_gain: float
    detail_gain: float
    velocity_gain: float


FLUID_PROFILES = { # Style specific fluid behaviour presets matched to the visual palettes
    "ember": FluidProfile(0.66, 0.72, 1.12, 1.22, 1.05, 0.20, 3.05, 0.20),
    "aurora": FluidProfile(0.56, 1.04, 0.72, 0.78, 1.18, 0.16, 3.45, 0.18),
    "voltage": FluidProfile(0.48, 0.58, 1.62, 1.35, 1.34, 0.12, 4.10, 0.28),
    "prism": FluidProfile(0.62, 0.82, 1.35, 1.05, 1.16, 0.15, 3.80, 0.24),
    "ghost": FluidProfile(0.42, 1.22, 0.52, 0.62, 1.46, 0.10, 4.35, 0.16),
}


class FluidSimulation: # Simulates a low resolution density and velocity field for terminal visuals
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
        self.diffusion = float(diffusion) # Store solver parameters and initialize the default fluid grid
        self.viscosity = float(viscosity)
        self.iterations = int(iterations)
        self.time = 0.0
        self.profile = FLUID_PROFILES["ember"]
        self.ensure_size(width, height)

    def ensure_size(self, width: int, height: int) -> None:
        width = max(8, int(width)) # Resize or initialize all simulation arrays when the output size changes
        height = max(8, int(height)) # Keep a minimum grid size so solver slices and boundaries remain valid
        if getattr(self, "width", None) == width and getattr(self, "height", None) == height:
            return # Avoid reallocating when the requested size is unchanged

        self.width = width
        self.height = height
        shape = (height, width)
        self.density = np.zeros(shape, dtype=np.float32)
        self.density_prev = np.zeros(shape, dtype=np.float32) # Main density field shown on screen
        self.vx = np.zeros(shape, dtype=np.float32)
        self.vy = np.zeros(shape, dtype=np.float32) # Velocity field used to advect the density through the simulation
        self.vx_prev = np.zeros(shape, dtype=np.float32)
        self.vy_prev = np.zeros(shape, dtype=np.float32)
        self.pressure = np.zeros(shape, dtype=np.float32)
        self.divergence = np.zeros(shape, dtype=np.float32)
        self.yy, self.xx = np.mgrid[0:height, 0:width].astype(np.float32)
# Precompute coordinate grids used for source injection
    def step(
        self,
        dt: float, # Advance the fluid simulation by 1 frame
        features: AudioFeatures | None = None,
        *,
        profile: str = "ember",
    ) -> None:
        features = features or AudioFeatures()
        self.profile = _fluid_profile(profile)
        dt = float(np.clip(dt, 0.0, 0.08)) # Clamp large frame times so the solver stays stable
        if dt == 0.0:
            return

        self.time += dt
        self._inject(features) # Inject audio reactive density and velocity before solving fluid motion

        self.vx_prev[:] = self.vx # Diffuse velocity, then project it to reduce compression artifacts
        self.vy_prev[:] = self.vy
        self._diffuse(1, self.vx, self.vx_prev, self.viscosity, dt)
        self._diffuse(2, self.vy, self.vy_prev, self.viscosity, dt)
        self._project(self.vx, self.vy, self.pressure, self.divergence)

        self.vx_prev[:] = self.vx
        self.vy_prev[:] = self.vy # Move velocity through the current velocity field, then project again
        self._advect(1, self.vx, self.vx_prev, self.vx_prev, self.vy_prev, dt)
        self._advect(2, self.vy, self.vy_prev, self.vx_prev, self.vy_prev, dt)
        self._project(self.vx, self.vy, self.pressure, self.divergence)

        self.density_prev[:] = self.density
        self._diffuse(0, self.density, self.density_prev, self.diffusion, dt) # Diffuse and advect density using the solved velocity field
        self.density_prev[:] = self.density
        self._advect(0, self.density, self.density_prev, self.vx, self.vy, dt)

        self.density *= np.exp( # Fade density over time, with broadband audio increasing decay slightly
            -(0.78 + 0.24 * features.broadband) * self.profile.decay * dt
        )
        self.vx *= np.exp(-0.14 * dt) # Slowly damp velocity so motion does not build up forever
        self.vy *= np.exp(-0.14 * dt)
        np.clip(self.density, 0.0, 1.4, out=self.density) # Clamp fields to keep terminal rendering stable and bounded
        np.clip(self.vx, -3.0, 3.0, out=self.vx)
        np.clip(self.vy, -3.0, 3.0, out=self.vy)

    def rasterize(
        self,
        width: int,
        height: int,
        *,
        threshold: float = 0.08, # Convert the simulation field into a boolean braille canvas plus intensity map
    ) -> tuple[np.ndarray, np.ndarray]:
        width = max(2, int(width))
        height = max(4, int(height))
        density = np.clip(self.density, 0.0, 1.0)
        velocity = np.sqrt(self.vx * self.vx + self.vy * self.vy)
        if density.shape != (height, width):
            density = self._resample(density, width, height)
            velocity = self._resample(velocity, width, height) # Downsample or upsample fields when render size differs from simulation size
        detail = self._edge_detail(density)
        intensity = np.clip(
            density * self.profile.fill_gain
            + detail * self.profile.detail_gain
            + np.clip(velocity * self.profile.velocity_gain, 0.0, 0.34),
            0.0,
            1.0,
        )
        return intensity > threshold, intensity

    def _inject(
        self,
        features: AudioFeatures,
    ) -> None:
        bass = float(features.bass)
        broadband = float(features.broadband)
        intensity = float(features.intensity) # Inject audio reactive sources into the fluid field
        span = min(self.width, self.height)
        profile = self.profile
        t = self.time
        density_amount = (0.035 + 0.16 * intensity + 0.16 * bass) * profile.density_scale

        self._inject_source(
            self.width * (0.50 + 0.27 * np.sin(t * 0.67)),
            self.height * (0.52 + 0.20 * np.cos(t * 0.83)), # Scale injected density based on bass and overall intensity
            span * (0.062 + 0.020 * bass) * profile.radius_scale,
            density_amount, # Primary moving source that reacts strongly to bass
            (0.34 + 0.88 * bass) * profile.swirl_scale,
            (0.10 + 0.30 * broadband) * profile.radial_scale,
            stretch=(1.35, 0.90),
        )
        self._inject_source(
            self.width * (0.50 + 0.31 * np.sin(t * 0.41 + 2.30)),
            self.height * (0.50 + 0.23 * np.sin(t * 0.58 + 1.10)),
            span * (0.044 + 0.018 * broadband) * profile.radius_scale,
            density_amount * (0.45 + 0.22 * broadband),
            (-0.28 - 0.68 * broadband) * profile.swirl_scale, # Secondary source that reacts more to broadband energy
            (-0.06 - 0.20 * bass) * profile.radial_scale,
            stretch=(0.90, 1.45),
        )
        self._inject_source(
            self.width * (0.50 + 0.38 * np.sin(t * 0.29 + 4.10)),
            self.height * (0.53 + 0.25 * np.cos(t * 0.37 + 0.70)),
            span * (0.032 + 0.016 * intensity) * profile.radius_scale,
            density_amount * (0.28 + 0.24 * broadband),
            (0.18 + 0.55 * intensity) * profile.swirl_scale,
            (0.05 + 0.18 * bass) * profile.radial_scale, # Smaller detail source driven by overall intensity
            stretch=(1.10, 1.10),
        )

        shear = (0.003 + 0.010 * broadband) * np.sin(
            self.yy * 0.11 + self.xx * 0.035 + t * 1.4
        )
        self.vx += shear.astype(np.float32)

        if features.onset:
            self._inject_source(
                self.width * (0.50 + 0.34 * np.sin(t * 1.30)),
                self.height * (0.78 - 0.20 * np.cos(t * 0.91)), # Add a subtle wave like horizontal shear for continuous motion
                span * (0.046 + 0.020 * bass) * profile.radius_scale,
                (0.18 + 0.22 * bass) * profile.density_scale,
                (-0.20 + 0.42 * np.sin(t)) * profile.swirl_scale,
                (0.48 + 0.52 * bass) * profile.radial_scale,
                stretch=(1.05, 0.82), # Add an extra burst source when an audio onset/beat is detected
            )

    def _inject_source(
        self,
        cx: float,
        cy: float,
        radius: float,
        density_amount: float,
        swirl: float,
        radial: float,
        *,
        linear: tuple[float, float] = (0.0, 0.0),
        stretch: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        radius = max(2.0, float(radius))
        sx = max(0.35, float(stretch[0]))
        sy = max(0.35, float(stretch[1])) # Add density and velocity into a soft elliptical region
        reach_x = radius * sx
        reach_y = radius * sy
        x0 = max(1, int(cx - reach_x))
        x1 = min(self.width - 1, int(cx + reach_x) + 1)
        y0 = max(1, int(cy - reach_y))
        y1 = min(self.height - 1, int(cy + reach_y) + 1)
        if x0 >= x1 or y0 >= y1:
            return

        patch_x = self.xx[y0:y1, x0:x1]
        patch_y = self.yy[y0:y1, x0:x1]
        dx = (patch_x - cx) / reach_x
        dy = (patch_y - cy) / reach_y # Clamp source bounds so injection only touches valid interior cells
        dist2 = dx * dx + dy * dy
        blob = np.maximum(0.0, 1.0 - dist2)
        blob = (blob * blob).astype(np.float32)

        density_patch = self.density[y0:y1, x0:x1]
        vx_patch = self.vx[y0:y1, x0:x1]
        vy_patch = self.vy[y0:y1, x0:x1]
        density_patch += blob * float(density_amount)
        vx_patch += blob * (float(linear[0]) - dy * float(swirl) + dx * float(radial))
        vy_patch += blob * (float(linear[1]) + dx * float(swirl) + dy * float(radial)) # Build a soft blob mask that fades toward the edge of the source

    @staticmethod
    def _resample(field: np.ndarray, width: int, height: int) -> np.ndarray:
        y_indices = np.linspace(0, field.shape[0] - 1, height).astype(np.int32)
        x_indices = np.linspace(0, field.shape[1] - 1, width).astype(np.int32)
        return field[y_indices[:, None], x_indices[None, :]]

    @staticmethod # Add density plus swirl/radial velocity to the selected patch
    def _edge_detail(field: np.ndarray) -> np.ndarray:
        detail = np.zeros_like(field, dtype=np.float32)
        detail[:, 1:-1] += np.abs(field[:, 2:] - field[:, :-2])
        detail[1:-1, :] += np.abs(field[2:, :] - field[:-2, :])
        return detail # Nearest neighbour resampling for matching render dimensions

    def _diffuse(
        self,
        boundary: int,
        target: np.ndarray,
        source: np.ndarray, # Estimate visual edge detail using local density differences
        amount: float,
        dt: float,
    ) -> None:
        scale = max(self.width, self.height)
        a = dt * amount * scale * scale
        self._linear_solve(boundary, target, source, a, 1.0 + 4.0 * a)

    def _linear_solve(
        self,
        boundary: int, # Diffuse a field by solving a linear system
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

        vx[1:-1, 1:-1] -= 0.5 * self.width * ( # Make the velocity field approximately divergence free
            pressure[1:-1, 2:] - pressure[1:-1, :-2]
        )
        vy[1:-1, 1:-1] -= 0.5 * self.height * (
            pressure[2:, 1:-1] - pressure[:-2, 1:-1]
        )
        self._set_boundary(1, vx) # Compute divergence from neighbouring velocity differences
        self._set_boundary(2, vy)

    def _advect(
        self,
        boundary: int,
        target: np.ndarray,
        source: np.ndarray, # Solve pressure from divergence
        vx: np.ndarray,
        vy: np.ndarray, # Substract pressure gradient from velocity
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
    # Move a field backward through the velocity field and sample from the source
    @staticmethod
    def _set_boundary(boundary: int, target: np.ndarray) -> None:
        if boundary == 1:
            target[:, 0] = -target[:, 1]
            target[:, -1] = -target[:, -2]
        else:
            target[:, 0] = target[:, 1]
            target[:, -1] = target[:, -2] # Trace each cell backward to find where its value came from

        if boundary == 2:
            target[0, :] = -target[1, :]
            target[-1, :] = -target[-2, :]
        else:
            target[0, :] = target[1, :]
            target[-1, :] = target[-2, :]

        target[0, 0] = 0.5 * (target[1, 0] + target[0, 1])
        target[0, -1] = 0.5 * (target[1, -1] + target[0, -2]) # Bilinearly sample the source field at the traced positions
        target[-1, 0] = 0.5 * (target[-2, 0] + target[-1, 1])
        target[-1, -1] = 0.5 * (target[-2, -1] + target[-1, -2])


def _fluid_profile(name: str) -> FluidProfile:
    return FLUID_PROFILES.get(name, FLUID_PROFILES["ember"])
