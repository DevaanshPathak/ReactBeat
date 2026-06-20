# Roadmap

Each phase should end with a runnable app and a commit. The command `python -m src.cli <audiofile>` should work once audio loading exists, and Phase 1 should also run without an audio path for renderer validation.

## Phase 0: Documentation Baseline

- [x] Create `README.md`.
- [x] Create `PRD.md`.
- [x] Create `ROADMAP.md`.
- [x] Commit documentation baseline.

## Phase 1: Particle Mode and Braille Renderer

Goal: prove the rendering pipeline end to end without audio.

- [x] Create package structure.
- [x] Add project metadata and dependencies.
- [x] Implement custom braille packer.
- [x] Implement numpy-vectorized particle simulation.
- [x] Implement Textual app and simulation widget at 30 FPS.
- [x] Add CLI entry point.
- [x] Verify `python -m src.cli` runs.
- [x] Commit Phase 1.

Phase 1 note: `python -m src.cli` launches the Textual particle renderer, and `python -m src.cli --smoke-test` renders one deterministic braille frame for non-interactive verification.

## Phase 2: Audio Decode and Playback Sync

Goal: load a local audio file, play it, and expose playback sample position to the render loop.

- [x] Implement `audio.loader`.
- [x] Implement callback-driven `audio.player`.
- [x] Wire CLI audio path into app.
- [x] Display or use live playback position in the widget.
- [x] Verify `python -m src.cli <audiofile>` runs.
- [x] Commit Phase 2.

Phase 2 note: audio decode uses `soundfile` for WAV/FLAC/OGG and playback uses a `sounddevice.OutputStream` callback with thread-safe sample-position tracking. Local verification decoded and played a generated WAV through PortAudio until the tracked position reached the final frame.

## Phase 3: Beat and Energy Detection

Goal: make particle motion react to live audio features.

- [x] Implement manual FFT energy analyzer.
- [x] Track bass and broadband energy with rolling adaptive thresholds.
- [x] Expose onset/beat flags per playback position.
- [x] Inject forces and particles from audio features.
- [x] Verify visible beat-reactive behavior.
- [x] Commit Phase 3.

Phase 3 note: `EnergyAnalyzer` computes bass and broadband FFT energy around the current playback sample, compares each band to rolling local averages, suppresses repeated onsets within a short cooldown, and feeds `AudioFeatures` directly into the particle simulation.

## Phase 4: Switchable Visual Styles

Goal: make the same simulation feel meaningfully different at runtime.

- [x] Implement style definitions in `render.styles`.
- [x] Add at least 3 styles.
- [x] Add keybinding to cycle styles during playback.
- [x] Keep styles decoupled from simulation state.
- [x] Commit Phase 4.

Phase 4 note: `ember`, `aurora`, and `voltage` styles define palettes plus feature gain and render-threshold behavior. `--style` selects the initial profile, and `s` cycles profiles in the running Textual app without touching playback state.

## Phase 5: Fluid Simulation Mode Stretch

Goal: add a second physics mode using a coarse Stable Fluids solver.

- [ ] Implement density and velocity grids.
- [ ] Implement diffuse, advect, and project steps.
- [ ] Add audio-reactive source injection.
- [ ] Add mode switch keybinding.
- [ ] Verify acceptable frame timing.
- [ ] Commit Phase 5 if completed.

## Phase 6: Packaging and Validation

Goal: produce a one-file Linux binary that works in a clean environment.

- [ ] Add PyInstaller build config or command docs.
- [ ] Build one-file `reactbeat` Linux binary.
- [ ] Validate in a clean Linux VM/container.
- [ ] Document install/run instructions and known limitations.
- [ ] Commit packaging work.
