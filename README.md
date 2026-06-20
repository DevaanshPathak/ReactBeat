# reactbeat

`reactbeat` is a terminal music visualizer that renders beat-reactive generative art with real simulation state instead of spectrum bars. It decodes local audio, plays it back, analyzes short-time energy in sync with playback, and draws particles or fluid-like motion as packed braille graphics in a Textual TUI.

This project is being built for Hack Club TerminalCraft YSWS with a strict self-contained local runtime goal: no network APIs, no subprocess wrappers around external media tools, and no dependency on an installed Python runtime in the final Linux binary.

## Current Status

Phase 6 is complete. The app runs the particle/braille renderer without audio, decodes and plays WAV/FLAC/OGG files locally, tracks callback playback position, uses a manual short-time FFT analyzer to drive simulation energy and onset bursts, supports runtime visual style switching, includes an alternate Stable Fluids-style mode, and has a validated one-file Linux PyInstaller build. See [ROADMAP.md](ROADMAP.md) for the phase checklist.

## Planned Features

- Textual app with a custom simulation widget running at 30 FPS.
- Custom braille renderer that packs a boolean pixel canvas into Unicode braille cells.
- Numpy-vectorized particle simulation with audio-reactive spawning and force injection.
- Local audio decode with `soundfile` for WAV, FLAC, and OGG.
- Callback-driven playback with `sounddevice`, exposing live playback sample position.
- Manual short-time energy analysis using `numpy.fft`, with adaptive onset detection.
- Switchable visual styles while playback continues.
- Stretch goal: Stable Fluids solver as an alternate simulation mode.
- PyInstaller one-file Linux build validated in a clean Linux environment.

## Supported Audio

Initial supported formats are the formats handled by libsndfile through `soundfile`, especially:

- WAV
- FLAC
- OGG/Vorbis

MP3 is intentionally out of scope for the first version because common Python MP3 workflows rely on `ffmpeg` subprocesses, which violates the self-contained rule.

## Development Setup

Use Python 3.11+ during development.

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

## Running

During development:

```bash
python -m src.cli path/to/audio.wav
```

Check bundled runtime libraries:

```bash
python -m src.cli --diagnostics
```

Decode an audio file without starting playback:

```bash
python -m src.cli path/to/audio.wav --check-audio
```

Choose an initial style:

```bash
python -m src.cli path/to/audio.wav --style aurora
```

Start in fluid mode:

```bash
python -m src.cli path/to/audio.wav --mode fluid
```

Running without audio remains supported for development:

```bash
python -m src.cli
```

## Controls

Controls will grow by phase. The intended baseline is:

- `q`: quit
- `space`: pause/resume simulation or playback
- `s`: cycle render style (`ember`, `aurora`, `voltage`)
- `m`: cycle simulation mode (`particles`, `fluid`)

## Packaging Goal

The final release target is a single Linux binary built with PyInstaller one-file mode. From a machine with Docker:

```bash
bash packaging/linux/build.sh
```

The generated binary is written to:

```bash
dist/linux/reactbeat
```

The Docker build validates the frozen binary in a clean `debian:bookworm-slim` stage by running:

```bash
reactbeat --smoke-test --mode particles
reactbeat --smoke-test --mode fluid --style aurora
reactbeat --diagnostics
```

The PyInstaller spec bundles PortAudio, libsndfile support, and ALSA configuration data so `sounddevice` and `soundfile` load without a Python install or package install on the target machine.
