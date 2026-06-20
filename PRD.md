# Product Requirements Document: reactbeat

## Purpose

Build a complete Python TUI application that turns local audio playback into beat-reactive generative terminal art. The differentiator is simulation-driven motion rendered through a custom braille graphics pipeline, not FFT bars or a wrapper around an existing visualizer.

## Target User

Terminal users and TerminalCraft judges who want a self-contained, visually expressive local app that runs from the command line and reacts convincingly to music.

## Goals

- Load and play local audio files fully offline.
- Keep visual state synchronized to audio playback position.
- Render a particle simulation in braille characters at about 30 FPS.
- Detect energy and beats with a lightweight manual analyzer.
- Provide multiple visual styles without restarting playback.
- Package a Linux one-file binary that works without a local Python install.

## Non-Goals

- MP3 support through `ffmpeg`, `pydub`, or any external executable.
- External APIs, cloud analysis, or network dependency.
- A bar chart, spectrum analyzer clone, or thin wrapper around CAVA, cli-visualizer, drawille, or similar tools.
- Heavy/JIT analysis stacks such as `librosa`, `numba`, `scipy`, or `essentia`.

## Hard Constraints

- The app must be self-contained for users.
- Audio decode, analysis, simulation, and rendering must run locally.
- The particle/fluid simulation and braille renderer must be original project code.
- The final deliverable must be a pre-built Linux binary.
- The implementation must remain suitable for PyInstaller freezing.

## Functional Requirements

### TUI

- Use Textual and Rich.
- Implement a custom `Widget` subclass for the simulation surface.
- Drive animation with `self.set_interval(1 / 30, self.tick)`.
- Return `rich.text.Text` from `render()`.
- Keep UI responsive while audio playback runs in a callback/background thread.

### Audio Loading

- Use `soundfile` to decode WAV, FLAC, and OGG into numpy arrays.
- Convert multi-channel audio to a predictable playback shape and analysis mono signal.
- Report clear errors for unsupported formats.

### Audio Playback

- Use `sounddevice` with callback-driven output.
- Track current playback sample position in a thread-safe way.
- Do not block the Textual asyncio event loop with audio I/O.

### Beat and Energy Analysis

- Implement manual windowed FFT analysis with numpy.
- Track at least bass/low-frequency and broadband energy.
- Maintain rolling local averages and adaptive thresholds.
- Expose per-frame features such as bass energy, broadband energy, onset flag, and beat intensity.

### Particle Simulation

- Store particle state in numpy arrays.
- Update positions, velocities, ages, and lifetimes vectorially.
- Use energy/onsets to inject particles and forces.
- Keep simulation logic independent of Textual rendering.

### Fluid Simulation Stretch

- Implement a coarse-grid Stable Fluids-style solver.
- Include advection, diffusion, projection, and source injection.
- Make fluid mode selectable through the same app pipeline.

### Braille Rendering

- Implement custom packing from a boolean pixel canvas into Unicode braille characters starting at U+2800.
- Treat each character as a 2 by 4 dot matrix using the standard braille bit layout.
- Support Rich style spans per braille cell for style/color variants.

### Styles

- Provide at least 2 to 3 distinct visual styles.
- Allow style switching at runtime.
- Keep style code separate from simulation code.

### Packaging

- Build with PyInstaller one-file mode.
- Validate the frozen binary on clean Linux.
- Document packaging commands, runtime expectations, and known dependency caveats.

## Quality Requirements

- The app should run at about 30 FPS on a typical terminal size.
- The code should keep UI, audio, analysis, simulation, and rendering separated.
- Each phase must end in a runnable state.
- Commits should mark phase boundaries.
- Errors should be visible and actionable from the CLI.

## Milestones

1. Documentation baseline.
2. Particle mode and braille renderer without audio.
3. Audio decode and playback position sync.
4. Beat/energy analysis and particle reactivity.
5. Runtime style switching.
6. Fluid mode stretch.
7. Linux PyInstaller package and validation.
