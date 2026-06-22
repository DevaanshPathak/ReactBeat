from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from rich.text import Text


BRAILLE_BASE = 0x2800 # Unicode code point for an empty braille cell

# Standard Unicode braille dot bit layout for a 2 by 4 cell.
BRAILLE_BITS = np.array( # Bit values for each pixel position inside a 2x4 braille cell
    [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ],
    dtype=np.uint8,
)


def pack_braille(canvas: np.ndarray) -> list[str]: # Convert a boolean pixel canvas into compact Unicode braille rows
    """Pack a 2D boolean pixel canvas into braille text lines."""

    bool_canvas = _as_bool_canvas(canvas) # Validate and normalize the input canvas to a boolean pixels
    padded = _pad_canvas(bool_canvas) # Pad canvas dimensions so that they divide cleanly into 2x4 braille cells
    cell_rows = padded.shape[0] // 4
    cell_cols = padded.shape[1] // 2

    cells = padded.reshape(cell_rows, 4, cell_cols, 2).transpose(0, 2, 1, 3) # Reshape pixels into braille cells arranged as rows x columns x 4 x 2
    bit_values = (cells * BRAILLE_BITS).sum(axis=(2, 3)).astype(np.uint8) # Convert each 2x4 cell into its Unicode braille bitmask value

    return [
        "".join(chr(BRAILLE_BASE + int(value)) for value in row)
        for row in bit_values
    ]


def braille_canvas_to_text( # Convert a canvas into Rich Text, optionally applying styles per braille cell
    canvas: np.ndarray,
    cell_styles: np.ndarray | Sequence[Sequence[str | None]] | None = None,
    *,
    default_style: str | None = None,
) -> Text:
    """Pack a boolean canvas into Rich Text with optional per-cell styles."""

    lines = pack_braille(canvas)
    text = Text()

    if cell_styles is None: # Use one default style for the whole output when no style grid is provided
        for row_index, line in enumerate(lines):
            if row_index:
                text.append("\n")
            text.append(line, style=default_style)
        return text

    style_grid = np.asarray(cell_styles, dtype=object) # Normalize the style grid so that it can be indexed like a NumPy array
    for row_index, line in enumerate(lines):
        if row_index:
            text.append("\n")
        for col_index, char in enumerate(line):
            style = _style_at(style_grid, row_index, col_index, default_style) # Pick the style for this braille cell, falling back when missing
            text.append(char, style=style)
    return text


def _as_bool_canvas(canvas: np.ndarray) -> np.ndarray: # Ensure the canvas is a 2D boolean array before packing
    array = np.asarray(canvas)
    if array.ndim != 2:
        msg = f"braille canvas must be 2D, got shape {array.shape!r}"
        raise ValueError(msg)
    return array.astype(bool, copy=False)


def _pad_canvas(canvas: np.ndarray) -> np.ndarray: # Pad rows to multiples of 4 and columns to multiples of 2
    row_pad = (-canvas.shape[0]) % 4
    col_pad = (-canvas.shape[1]) % 2
    if row_pad == 0 and col_pad == 0:
        return canvas
    return np.pad(canvas, ((0, row_pad), (0, col_pad)), mode="constant")


def _style_at( # Safely fetch a cell style from the style grid
    style_grid: np.ndarray,
    row: int,
    col: int,
    fallback: str | None,
) -> Any:
    if row >= style_grid.shape[0] or col >= style_grid.shape[1]: # Fall back when the style grid is smaller than the braille output
        return fallback
    style = style_grid[row, col]
    return fallback if style is None else style
