from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .app import ReactBeatApp, render_smoke_frame


def write_unicode_line(value: str) -> None:
    """Write braille text even when the host console default is not UTF-8."""

    output = f"{value}\n"
    try:
        sys.stdout.write(output)
    except UnicodeEncodeError:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(output.encode("utf-8"))
            sys.stdout.buffer.flush()
        else:
            sys.stdout.write(output.encode("unicode_escape").decode("ascii"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reactbeat",
        description="Beat-reactive generative terminal art.",
    )
    parser.add_argument(
        "audiofile",
        nargs="?",
        type=Path,
        help="Optional audio file. Audio playback is implemented in Phase 2.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Render one deterministic braille frame and exit.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=48,
        help="Braille cell width for --smoke-test.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=16,
        help="Braille cell height for --smoke-test.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.smoke_test:
        frame = render_smoke_frame(
            width_cells=max(args.width, 4),
            height_cells=max(args.height, 2),
        )
        write_unicode_line(frame.plain)
        return 0

    ReactBeatApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
