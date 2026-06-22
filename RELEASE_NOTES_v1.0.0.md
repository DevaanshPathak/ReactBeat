# ReactBeat v1.0.0

ReactBeat is a fully local terminal music visualizer that plays WAV, FLAC, and OGG files, analyzes audio energy in sync with playback, and renders beat-reactive generative art using Unicode braille graphics.

## Highlights

- Textual TUI with a start screen, folder browser, recent files, and visible controls.
- Local audio decode through `soundfile` and callback playback through `sounddevice`.
- Manual NumPy FFT energy/onset analysis with no external API or ffmpeg dependency.
- Three simulation modes: `particles`, `fluid`, and `waves`.
- Five runtime styles: `ember`, `aurora`, `voltage`, `prism`, and `ghost`.
- One-file Linux binary built with PyInstaller and validated in a clean Debian container.

## Run

```bash
chmod +x reactbeat
./reactbeat
```

Direct file launch:

```bash
./reactbeat path/to/audio.wav
```

Smoke check:

```bash
./reactbeat --smoke-test --mode waves --style prism
```

## Supported Audio

- WAV
- FLAC
- OGG/Vorbis

MP3 is intentionally not included in v1.0.0 because common Python MP3 workflows require external `ffmpeg` subprocesses, which conflicts with the self-contained runtime goal.
