# reactbeat

`reactbeat` is a local terminal music visualizer. It decodes an audio file, plays it back, analyzes short-time energy in sync with playback, and renders particle, fluid, or wave motion as packed braille graphics in a Textual TUI.

The visual core is simulation-driven generative art, not FFT bars. Audio, analysis, simulation, and rendering all run locally.

## Features

- TUI start screen with Browse and Recent Files choices.
- Folder browser filtered to supported audio files.
- Persistent recent-file list for reopening previously selected tracks.
- Custom braille renderer using Unicode braille cells and Rich styles.
- Numpy-vectorized particle simulation.
- Stable Fluids-style density/velocity simulation.
- Damped wave/ripple simulation.
- Local WAV, FLAC, and OGG decode through `soundfile`.
- Callback-driven playback through `sounddevice`.
- Manual FFT energy analyzer with adaptive onset detection.
- Runtime visual styles: `ember`, `aurora`, `voltage`, `prism`, `ghost`.
- One-file Linux PyInstaller build with bundled PortAudio and ALSA config.

## Supported Audio

- WAV
- FLAC
- OGG/Vorbis

MP3 is intentionally not supported because common Python MP3 paths use `ffmpeg` subprocesses, which would break the self-contained runtime goal.

## Development Setup

Use Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run

Open the TUI start screen:

```bash
python -m src.cli
```

From there, choose Browse to navigate folders or Recent Files to reopen a previously selected track.

Open the picker at a specific folder:

```bash
python -m src.cli C:\Users\you\Music
```

Start directly with an audio file:

```bash
python -m src.cli path/to/audio.wav
```

Choose initial mode or style:

```bash
python -m src.cli path/to/audio.wav --mode waves --style prism
```

## Controls

- `Enter`: choose the highlighted menu item, folder, or audio file
- `b`: browse folders
- `r`: open recent files
- `h`: return to the start screen
- `u` / `Backspace`: move the picker root to the parent folder
- `q`: quit
- `space`: pause/resume visualization and playback
- `s`: cycle style
- `m`: cycle simulation mode

The visualizer also shows these controls in a status bar above the animation.

## Test

Run the cross-platform local check suite:

```bash
python -m scripts.check
```

That command runs unit tests, particle/fluid smoke renders, runtime diagnostics, and WAV decode validation. It works on Windows as well as Linux/macOS development environments.

Useful individual checks:

```bash
python -m src.cli --diagnostics
python -m src.cli --smoke-test --mode particles
python -m src.cli --smoke-test --mode fluid --style aurora
python -m src.cli --smoke-test --mode waves --style prism
python -m src.cli path/to/audio.wav --check-audio
```

## Linux Binary

From a machine with Docker:

```bash
bash packaging/linux/build.sh
```

The generated binary is written to:

```bash
dist/linux/reactbeat
```

The Docker build validates the frozen binary in a clean `debian:bookworm-slim` stage by running particle smoke render, fluid smoke render, and bundled runtime diagnostics.
