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

Phase 1 note: `python -m src.cli` launches the Textual particle renderer, and `python -m src.cli --smoke-test` renders one deterministic braille frame for non-interactive verification. Audio arguments are parsed but intentionally unused until Phase 2.

## Phase 2: Audio Decode and Playback Sync

Goal: load a local audio file, play it, and expose playback sample position to the render loop.

- [ ] Implement `audio.loader`.
- [ ] Implement callback-driven `audio.player`.
- [ ] Wire CLI audio path into app.
- [ ] Display or use live playback position in the widget.
- [ ] Verify `python -m src.cli <audiofile>` runs.
- [ ] Commit Phase 2.

## Phase 3: Beat and Energy Detection

Goal: make particle motion react to live audio features.

- [ ] Implement manual FFT energy analyzer.
- [ ] Track bass and broadband energy with rolling adaptive thresholds.
- [ ] Expose onset/beat flags per playback position.
- [ ] Inject forces and particles from audio features.
- [ ] Verify visible beat-reactive behavior.
- [ ] Commit Phase 3.

## Phase 4: Switchable Visual Styles

Goal: make the same simulation feel meaningfully different at runtime.

- [ ] Implement style definitions in `render.styles`.
- [ ] Add at least 3 styles.
- [ ] Add keybinding to cycle styles during playback.
- [ ] Keep styles decoupled from simulation state.
- [ ] Commit Phase 4.

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
