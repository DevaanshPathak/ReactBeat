from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.particles import AudioFeatures


@dataclass(frozen=True) # Color/style palette ordered from low to high visual intensity
class Palette:
    name: str
    levels: tuple[str, ...]


@dataclass(frozen=True) # Visual tuning profile that controls palette, audio response and draw threshold
class VisualStyle:
    name: str
    palette: Palette
    bass_gain: float = 1.0
    broadband_gain: float = 1.0
    intensity_gain: float = 1.0
    threshold: float = 0.11
    # Apply this style's gain values to raw audio features and clamp them to 0-1
    def shape_features(self, features: AudioFeatures) -> AudioFeatures:
        return AudioFeatures(
            bass=float(np.clip(features.bass * self.bass_gain, 0.0, 1.0)),
            broadband=float(np.clip(features.broadband * self.broadband_gain, 0.0, 1.0)),
            onset=features.onset,
            intensity=float(np.clip(features.intensity * self.intensity_gain, 0.0, 1.0)),
        )


VISUAL_STYLES = ( # Built in visual styles available from the CLI
    VisualStyle(
        name="ember",
        palette=Palette(
            name="ember",
            levels=(
                "#301616",
                "#7f2f24",
                "#c6552c",
                "#f28c38",
                "bold #ffd166",
                "bold #fff6d5",
            ),
        ),
        bass_gain=1.08,
        broadband_gain=0.96,
        intensity_gain=1.06,
        threshold=0.11,
    ),
    VisualStyle(
        name="aurora",
        palette=Palette(
            name="aurora",
            levels=(
                "#08221e",
                "#0f5c52",
                "#1b9aaa",
                "#6ee7b7",
                "bold #b8f7d4",
                "bold #f0fffb",
            ),
        ),
        bass_gain=0.82,
        broadband_gain=1.18,
        intensity_gain=0.92,
        threshold=0.085,
    ),
    VisualStyle(
        name="voltage",
        palette=Palette(
            name="voltage",
            levels=(
                "#101018",
                "#2c3e8f",
                "#287bd1",
                "#22d3ee",
                "bold #e8ff4f",
                "bold #ffffff",
            ),
        ),
        bass_gain=1.34,
        broadband_gain=1.12,
        intensity_gain=1.25,
        threshold=0.15,
    ),
    VisualStyle(
        name="prism",
        palette=Palette(
            name="prism",
            levels=(
                "#171717",
                "#9d174d",
                "#0891b2",
                "#84cc16",
                "bold #fde047",
                "bold #ffffff",
            ),
        ),
        bass_gain=0.96,
        broadband_gain=1.28,
        intensity_gain=1.10,
        threshold=0.10,
    ),
    VisualStyle(
        name="ghost",
        palette=Palette(
            name="ghost",
            levels=(
                "#111827",
                "#374151",
                "#94a3b8",
                "#a7f3d0",
                "bold #f8fafc",
                "bold #ffffff",
            ),
        ),
        bass_gain=0.74,
        broadband_gain=0.92,
        intensity_gain=0.86,
        threshold=0.075,
    ),
)
# Default style used when the user does not choose one
DEFAULT_STYLE = VISUAL_STYLES[0]


def style_names() -> tuple[str, ...]: # Return all supported style names for CLI choices and validation messages
    return tuple(style.name for style in VISUAL_STYLES)


def style_by_name(name: str) -> VisualStyle: # Look up a visual style by name, ignoring surrounding spaces and case
    normalized = name.strip().lower()
    for style in VISUAL_STYLES:
        if style.name == normalized:
            return style
    valid = ", ".join(style_names()) # Show a helpful error listing valid styles when the name is unknown
    raise ValueError(f"unknown style {name!r}; valid styles: {valid}")


def cell_styles_from_intensity(intensity: np.ndarray, palette: Palette) -> np.ndarray:
    """Map a pixel intensity field to a per-braille-cell style grid."""

    values = np.asarray(intensity, dtype=np.float32)
    if values.ndim != 2: # Normalize the intensity input to a 2D float array
        msg = f"intensity field must be 2D, got shape {values.shape!r}"
        raise ValueError(msg)

    row_pad = (-values.shape[0]) % 4
    col_pad = (-values.shape[1]) % 2 # Pad the field so it divides cleanly into 4x2 braille cells
    if row_pad or col_pad:
        values = np.pad(values, ((0, row_pad), (0, col_pad)), mode="constant")

    cell_rows = values.shape[0] // 4
    cell_cols = values.shape[1] // 2
    cells = values.reshape(cell_rows, 4, cell_cols, 2).transpose(0, 2, 1, 3) # Group pixels into braille cell sized blocks
    cell_energy = cells.max(axis=(2, 3))
    # Use the brightest pixel in each braille cell to choose its style
    if cell_energy.size == 0:
        return np.empty((0, 0), dtype=object)

    scaled = np.clip(cell_energy, 0.0, 1.0)
    level_indices = np.minimum( # Clamp intensities before converting them into palette indexes
        (scaled * len(palette.levels)).astype(np.int16), # Map each cell's 0-1 energy to a palette level
        len(palette.levels) - 1,
    )

    styles = np.empty(level_indices.shape, dtype=object)
    for index, style in enumerate(palette.levels):
        styles[level_indices == index] = style # Fill the style grid with the palette style matching each level index
    return styles
