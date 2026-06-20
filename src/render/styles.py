from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Palette:
    name: str
    levels: tuple[str, ...]


DEFAULT_PALETTE = Palette(
    name="ember",
    levels=(
        "#3b1d1d",
        "#7f2f24",
        "#c6552c",
        "#f28c38",
        "bold #ffd166",
        "bold #fff6d5",
    ),
)


def cell_styles_from_intensity(intensity: np.ndarray, palette: Palette) -> np.ndarray:
    """Map a pixel intensity field to a per-braille-cell style grid."""

    values = np.asarray(intensity, dtype=np.float32)
    if values.ndim != 2:
        msg = f"intensity field must be 2D, got shape {values.shape!r}"
        raise ValueError(msg)

    row_pad = (-values.shape[0]) % 4
    col_pad = (-values.shape[1]) % 2
    if row_pad or col_pad:
        values = np.pad(values, ((0, row_pad), (0, col_pad)), mode="constant")

    cell_rows = values.shape[0] // 4
    cell_cols = values.shape[1] // 2
    cells = values.reshape(cell_rows, 4, cell_cols, 2).transpose(0, 2, 1, 3)
    cell_energy = cells.max(axis=(2, 3))

    if cell_energy.size == 0:
        return np.empty((0, 0), dtype=object)

    scaled = np.clip(cell_energy, 0.0, 1.0)
    level_indices = np.minimum(
        (scaled * len(palette.levels)).astype(np.int16),
        len(palette.levels) - 1,
    )

    styles = np.empty(level_indices.shape, dtype=object)
    for index, style in enumerate(palette.levels):
        styles[level_indices == index] = style
    return styles
