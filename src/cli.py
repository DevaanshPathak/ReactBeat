from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .audio.loader import AudioLoadError, load_audio_file
from .audio.player import AudioPlayer
from .app import ReactBeatApp, mode_names, render_smoke_frame
from .render.styles import style_names


def write_unicode_line(value: str) -> None:
    """Write braille text even when the host console default is not UTF-8."""

    output = f"{value}\n"
    try: # Fall back to raw UTF-8 bytes when normal stdout encoding fails
        sys.stdout.write(output)
    except UnicodeEncodeError:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(output.encode("utf-8")) # Last resort fallback for consoles without a binary stdout buffer
            sys.stdout.buffer.flush()
        else:
            sys.stdout.write(output.encode("unicode_escape").decode("ascii"))
# Prefer a module's __version__, then fall back to package metadata

def package_version(module: object, distribution: str) -> str:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError: # Print runtime audio dependency versions for debugging packaging builds
        return "unknown"

# Report missing audio dependencies with a CLI friendly exit code
def print_diagnostics() -> int:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        sys.stderr.write(f"reactbeat: missing runtime dependency: {exc}\n")
        return 2 # Include PortAudio version because sounddevice depends on it for playback

    sys.stdout.write(f"soundfile={package_version(sf, 'soundfile')}\n")
    sys.stdout.write(f"sounddevice={package_version(sd, 'sounddevice')}\n")
    try:
        version, version_text = sd.get_portaudio_version()
        sys.stdout.write(f"portaudio={version} {version_text}\n")
    except Exception as exc:
        sys.stdout.write(f"portaudio=unavailable ({exc})\n")
    return 0 # Build the command line interface for launching and testing reactbeat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reactbeat", # Optional input can be either an audio file or a starting folder for browsing
        description="Beat-reactive generative terminal art.",
    )
    parser.add_argument(
        "audiofile",
        nargs="?",
        type=Path, # Render one deterministic frame for CI/smoke testing without opening the TUI
        help="Optional WAV, FLAC, or OGG file, or a folder to browse.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true", # Check installed/bundled audio libraries and exit
        help="Render one deterministic braille frame and exit.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true", # Decode audio metadata only, without starting playback
        help="Check bundled audio/runtime libraries and exit.",
    )
    parser.add_argument(
        "--check-audio",
        action="store_true", # Restrict style choices to the registered visual styles
        help="Decode the provided audio file and print metadata without playback.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"reactbeat {__version__}",
    )
    parser.add_argument(
        "--style",
        choices=style_names(),
        default=style_names()[0], # Restrict mode choices to the supported simulation backends
        help="Initial visual style.",
    )
    parser.add_argument(
        "--mode",
        choices=mode_names(),
        default="particles",
        help="Initial simulation mode.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=48, # Smoke test mode prints a deterministic braille frame and exits
        help="Braille cell width for --smoke-test.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=16,
        help="Braille cell height for --smoke-test.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int: # Diagnostics mode reports runtime library status and exits
    args = build_parser().parse_args(argv)

    if args.smoke_test:
        frame = render_smoke_frame(
            width_cells=max(args.width, 4), # Default to browsing from the current directory unless an input path is provided
            height_cells=max(args.height, 2),
            style_name=args.style, # Treat a directory argument as the initial file browser location
            mode=args.mode,
        )
        write_unicode_line(frame.plain)
        return 0 # Load an audio file immediately when a file path is provided

    if args.diagnostics:
        return print_diagnostics()
    # Return a user facing error if the audio file cannot be decoded
    audio = None
    player = None
    browse_start = Path.cwd()
    if args.audiofile is not None: # --check-audio requires a successfully loaded audio file
        if args.audiofile.expanduser().is_dir():
            browse_start = args.audiofile
        else:
            try: # Print basic decoded audio metadata for scripts or CI checks
                audio = load_audio_file(args.audiofile)
                player = AudioPlayer(audio)
            except AudioLoadError as exc:
                sys.stderr.write(f"reactbeat: {exc}\n")
                return 2

    if args.check_audio:
        if audio is None:
            sys.stderr.write("reactbeat: --check-audio requires an audiofile\n")
            return 2 # Launch the interactive textual app with the parsed startup options
        sys.stdout.write(
            "audio: "
            f"frames={audio.frames} "
            f"channels={audio.channels} "
            f"sample_rate={audio.sample_rate} "
            f"duration={audio.duration_seconds:.3f}s\n"
        )
        return 0

    ReactBeatApp(
        audio=audio, # Allow the module to be run directly with python -m reactbeat.cli
        player=player,
        initial_style=args.style,
        initial_mode=args.mode,
        browse_start=browse_start,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
