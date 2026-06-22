from __future__ import annotations

import asyncio
from math import sin
from pathlib import Path
from time import monotonic

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import DirectoryTree, Footer, Header, ListItem, ListView, Static
from textual.widget import Widget

from .audio.analysis import EnergyAnalyzer
from .audio.loader import SUPPORTED_EXTENSIONS, AudioData, AudioLoadError, load_audio_file
from .audio.player import AudioPlaybackError, AudioPlayer
from .render.braille import braille_canvas_to_text
from .render.styles import (
    DEFAULT_STYLE,
    VISUAL_STYLES,
    VisualStyle,
    cell_styles_from_intensity,
    style_by_name,
)
from .recent import RecentFiles
from .sim.fluid import FluidSimulation
from .sim.particles import AudioFeatures, ParticleSystem
from .sim.waves import WaveSimulation


TARGET_FPS = 30.0 # Target render/update rate for the textual animation loop
SIMULATION_MODES = ("particles", "fluid", "waves") # Simulation modes available in the visualizer
MAX_FLUID_GRID_WIDTH = 180 # Cap fluid simulation size to keep rendering responsive in large terminals
MAX_FLUID_GRID_HEIGHT = 96
MAX_WAVE_GRID_WIDTH = 220 # Cap wave simulation size seperately because waves can handle a slightly larger grid
MAX_WAVE_GRID_HEIGHT = 112


class HomeWidget(Widget):
    """Start screen that asks how to choose audio."""

    DEFAULT_CSS = """
    HomeWidget {
        height: 1fr;
        layout: vertical;
        background: black;
        padding: 1 2;
    }

    HomeWidget .home-title {
        height: 2;
        color: #f0fffb;
        text-style: bold;
    }

    HomeWidget .home-subtitle {
        height: 2;
        color: #b8f7d4;
    }

    HomeWidget ListView {
        height: auto;
        margin: 1 0;
    }

    HomeWidget ListItem {
        height: 3;
        padding: 0 1;
    }

    HomeWidget .home-help {
        height: auto;
        color: #ffd166;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult: # Build the home screen menu and keyboard hint text 
        yield Static("reactbeat", classes="home-title")
        yield Static("Choose an audio source", classes="home-subtitle")
        self.options = ListView(
            ListItem(Static("Browse folders"), id="home-browse"),
            ListItem(Static("Open recent files"), id="home-recent"),
            id="home-options",
        )
        yield self.options
        yield Static(
            "Enter select | b browse | r recents | q quit",
            classes="home-help",
        )

    def on_mount(self) -> None:
        self.options.focus()


class RecentFilesWidget(Widget):
    """Recent audio file selector."""

    DEFAULT_CSS = """
    RecentFilesWidget {
        height: 1fr;
        layout: vertical;
        background: black;
        padding: 1 2;
    }

    RecentFilesWidget .recent-title {
        height: 2;
        color: #f0fffb;
        text-style: bold;
    }

    RecentFilesWidget ListView {
        height: 1fr;
    }

    RecentFilesWidget ListItem {
        height: 3;
        padding: 0 1;
    }

    RecentFilesWidget .recent-help {
        height: 1;
        color: #ffd166;
    }
    """

    def __init__(self, recent_paths: list[Path]) -> None:
        super().__init__()
        self.recent_paths = recent_paths

    def compose(self) -> ComposeResult: # Build recent file entries or show a browse fallback when history is empty
        yield Static("Recent audio files", classes="recent-title")
        if self.recent_paths:
            items = [
                ListItem(Static(_recent_label(path)), id=f"recent-{index}")
                for index, path in enumerate(self.recent_paths)
            ]
        else:
            items = [
                ListItem(Static("No recent files yet. Browse folders."), id="recent-browse")
            ]
        self.options = ListView(*items, id="recent-options")
        yield self.options
        yield Static("Enter open | b browse | h home | q quit", classes="recent-help")

    def on_mount(self) -> None:
        self.options.focus()


class AudioDirectoryTree(DirectoryTree):
    """Directory tree filtered to folders and supported audio files."""

    def filter_paths(self, paths):  # type: ignore[no-untyped-def] 
        return [
            path
            for path in paths
            if _is_directory(path) or path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]


class FilePickerWidget(Widget):
    """Audio file picker shown before the visualizer has a selected file."""

    DEFAULT_CSS = """
    FilePickerWidget {
        height: 1fr;
        layout: vertical;
        background: black;
    }

    FilePickerWidget .picker-title {
        height: 1;
        padding: 0 1;
        color: #f0fffb;
        background: #111111;
    }

    FilePickerWidget AudioDirectoryTree {
        height: 1fr;
        padding: 0 1;
    }

    FilePickerWidget .picker-status {
        height: 1;
        padding: 0 1;
        color: #ffd166;
        background: #111111;
    }
    """

    can_focus = True

    def __init__(self, start_path: Path) -> None:
        super().__init__()
        self.start_path = start_path

    def compose(self) -> ComposeResult:
        yield Static("Audio files", classes="picker-title")
        self.audio_tree = AudioDirectoryTree(self.start_path, id="audio-tree")
        yield self.audio_tree
        self.status_line = Static(
            f"{self.start_path} | Enter open | u up | h home | r recents",
            classes="picker-status",
        )
        yield self.status_line

    def on_mount(self) -> None:
        self.audio_tree.focus()

    def go_to_parent(self) -> None: # Move the file picker root one directory upward
        current = Path(self.audio_tree.path).expanduser().resolve()
        parent = current.parent
        if parent != current:
            self.set_root(parent)

    def set_root(self, path: Path) -> None: # Change the picker root and refresh the status line
        root = path.expanduser().resolve()
        self.audio_tree.path = root
        self.status_line.update(f"{root} | Enter open | u up | h home | r recents")
        self.audio_tree.focus()


class ControlsBar(Static):
    """Visible controls and playback status for the visualizer."""

    DEFAULT_CSS = """
    ControlsBar {
        height: 2;
        padding: 0 1;
        color: #f0fffb;
        background: #111111;
    }
    """


class SimulationWidget(Widget):
    """Textual widget that renders a particle simulation as braille text."""

    DEFAULT_CSS = """
    SimulationWidget {
        width: 100%;
        height: 1fr;
        background: black;
        color: white;
    }
    """

    can_focus = True

    def __init__(
        self,
        *,
        audio: AudioData | None = None,
        player: AudioPlayer | None = None,
        initial_style: str = DEFAULT_STYLE.name,
        initial_mode: str = "particles",
        seed: int = 4,
        max_particles: int = 5000,
    ) -> None:
        super().__init__()
        self.audio = audio
        self.player = player
        self.analyzer = ( # Create an audio analyzer only when real audio is available
            EnergyAnalyzer(audio.mono, audio.sample_rate)
            if audio is not None
            else None
        )
        self.style = style_by_name(initial_style) # Resolve the initial visual style from CLI/config name
        self.system = ParticleSystem(max_particles=max_particles, seed=seed) # Initialize all simulation backends so switching modes is instant
        self.fluid = FluidSimulation(iterations=5)
        self.waves = WaveSimulation()
        self.mode = _validate_mode(initial_mode)
        self.paused = False
        self._started_at = monotonic()
        self._last_tick = self._started_at
        self._frame = Text("")

    def on_mount(self) -> None: # Start the fixed rate render loop after the widget mounts
        self.set_interval(1.0 / TARGET_FPS, self.tick)
        self.focus()

    def tick(self) -> None: # Advance simulation state and redraw one visual frame
        now = monotonic()
        dt = min(max(now - self._last_tick, 0.0), 1.0 / 12.0) # Clamp frame delta so pauses or lag do not destablize simulations
        self._last_tick = now
        width_cells = max(self.size.width, 20) # Convert terminal cell size into braille pixel dimensions
        height_cells = max(self.size.height, 8)
        pixel_width = width_cells * 2
        pixel_height = height_cells * 4

        if not self.paused: # Skip simulation updates when paused, but still render the latest frame
            features = self.style.shape_features(self._features_for_frame(now, dt)) # Pull audio features and apply the current style's gain shaping
            if self.mode == "particles": # Step the selected simulation backend
                self.system.step(dt, features)
            else:
                if self.mode == "fluid":
                    fluid_width, fluid_height = _fluid_grid_size(pixel_width, pixel_height) # Use a capped internal grid size for fluid mode performance
                    self.fluid.ensure_size(fluid_width, fluid_height)
                    self.fluid.step(dt, features, profile=self.style.name)
                else:
                    wave_width, wave_height = _wave_grid_size(pixel_width, pixel_height) # Use a capped internal grid size for wave mode performance
                    self.waves.ensure_size(wave_width, wave_height)
                    self.waves.step(dt, features)

        canvas, intensity = self._rasterize(pixel_width, pixel_height) # Convert simulation state into a pixel canvas and intensity field
        styles = cell_styles_from_intensity(intensity, self.style.palette) # Convert intensity values into per cell Rich styles
        self._frame = braille_canvas_to_text(canvas, styles) # Pack the canvas into styled braille text for terminal rendering
        self.refresh()

    def render(self) -> Text:
        return self._frame

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def cycle_style(self) -> VisualStyle: # Cycle to the next visual palette/style
        current_index = VISUAL_STYLES.index(self.style) 
        self.style = VISUAL_STYLES[(current_index + 1) % len(VISUAL_STYLES)]
        return self.style

    def cycle_mode(self) -> str: # Cycle to the next simulation mode
        current_index = SIMULATION_MODES.index(self.mode)
        self.mode = SIMULATION_MODES[(current_index + 1) % len(SIMULATION_MODES)]
        return self.mode

    def _playback_elapsed(self, fallback_now: float) -> float: # Get elapsed playback time, falling back to app runtime without audio
        if self.audio is None or self.player is None:
            return fallback_now - self._started_at
        return self.player.position_samples / self.audio.sample_rate

    def _features_for_frame(self, now: float, dt: float) -> AudioFeatures: # Use real audio analysis when available, otherwise generate fake demo features
        if self.analyzer is not None and self.player is not None:
            return self.analyzer.analyze_at(self.player.position_samples).to_audio_features()
        elapsed = self._playback_elapsed(now)
        return self._synthetic_features(elapsed, dt)

    def _rasterize(self, pixel_width: int, pixel_height: int) -> tuple[object, object]: # Rasterize whichever simulation mode is currently active
        if self.mode == "particles":
            return self.system.rasterize(
                width=pixel_width,
                height=pixel_height,
                threshold=self.style.threshold,
            )
        if self.mode == "fluid":
            return self.fluid.rasterize(
                width=pixel_width,
                height=pixel_height,
                threshold=max(0.045, self.style.threshold * 0.72),
            )
        return self.waves.rasterize(
            width=pixel_width,
            height=pixel_height,
            threshold=max(0.040, self.style.threshold * 0.62),
        )

    @staticmethod
    def _synthetic_features(elapsed: float, dt: float) -> AudioFeatures: # Generate synthetic beat like features for demo/smoke test rendering
        pulse_rate = 1.6
        cycle_pos = (elapsed * pulse_rate) % 1.0
        pulse = max(0.0, 1.0 - cycle_pos * 5.0) ** 3
        wave = 0.5 + 0.5 * sin(elapsed * 2.7)
        beat = cycle_pos < max(dt * pulse_rate, 0.035)
        return AudioFeatures(
            bass=0.22 + 0.78 * pulse,
            broadband=0.25 + 0.45 * wave + 0.30 * pulse,
            onset=beat,
            intensity=max(pulse, wave * 0.55),
        )


class ReactBeatApp(App[None]):
    """Top-level Textual application for reactbeat."""

    CSS = """
    Screen {
        background: black;
    }

    #body {
        height: 1fr;
        layout: vertical;
    }

    Header, Footer {
        background: #111111;
    }
    """

    BINDINGS = [ # Global key bindings for navigation and controls
        Binding("q", "quit", "Quit"),
        Binding("space", "toggle_pause", "Pause"),
        Binding("s", "cycle_style", "Style"),
        Binding("m", "cycle_mode", "Mode"),
        Binding("b", "browse_files", "Browse"),
        Binding("r", "recent_files", "Recent"),
        Binding("h", "home_screen", "Home"),
        Binding("u", "picker_parent", "Up"),
        Binding("backspace", "picker_parent", "Up", show=False),
    ]

    TITLE = "reactbeat"
    SUB_TITLE = "generative audio TUI"

    def __init__( # Store initial app state and shared screen/widget references
        self,
        *,
        audio: AudioData | None = None,
        player: AudioPlayer | None = None,
        initial_style: str = DEFAULT_STYLE.name,
        initial_mode: str = "particles",
        browse_start: Path | None = None,
        recent_files: RecentFiles | None = None,
    ) -> None:
        super().__init__()
        self.audio = audio
        self.player = player
        self.initial_style = initial_style
        self.initial_mode = _validate_mode(initial_mode)
        self.browse_start = (browse_start or Path.cwd()).expanduser().resolve() # Default the file browser to the current working directory
        self.recent_files = recent_files or RecentFiles()
        self.body: Container | None = None
        self.home: HomeWidget | None = None
        self.picker: FilePickerWidget | None = None
        self.recents: RecentFilesWidget | None = None
        self.controls: ControlsBar | None = None
        self.simulation: SimulationWidget | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self.body = Container(id="body")
        yield self.body
        yield Footer()

    async def on_mount(self) -> None: # Show the visualizer immediately if audio was provided, otherwise show home
        if self.audio is not None:
            await self.show_visualizer(self.audio, self.player)
        else:
            await self.show_home()

    def action_toggle_pause(self) -> None: # Toggle both visual simulation pause and audio playback pause
        if self.simulation is None:
            return
        self.simulation.toggle_pause()
        if self.player is not None:
            self.player.toggle_pause()
        self.update_controls()

    def action_cycle_style(self) -> None: # Switch visual style and notify the user
        if self.simulation is None:
            return
        style = self.simulation.cycle_style()
        self.update_controls()
        self.notify(f"Style: {style.name}", timeout=1.4)

    def action_cycle_mode(self) -> None: # Switch simulation mode and notify the user
        if self.simulation is None:
            return
        mode = self.simulation.cycle_mode()
        self.update_controls()
        self.notify(f"Mode: {mode}", timeout=1.4)

    def action_picker_parent(self) -> None:
        if self.picker is not None:
            self.picker.go_to_parent()

    async def action_browse_files(self) -> None:
        await self.show_picker(self.browse_start)

    async def action_recent_files(self) -> None:
        await self.show_recents()

    async def action_home_screen(self) -> None:
        await self.show_home()

    async def on_list_view_selected(self, event: ListView.Selected) -> None: # Handle menu selections from home and recent file screens
        event.stop()
        item_id = event.item.id or ""
        if item_id == "home-browse" or item_id == "recent-browse":
            await self.show_picker(self.browse_start)
        elif item_id == "home-recent":
            await self.show_recents()
        elif item_id.startswith("recent-") and self.recents is not None:
            index = int(item_id.removeprefix("recent-"))
            await self.open_audio_file(self.recents.recent_paths[index])

    async def on_directory_tree_file_selected( # Open an audio file selected from the directory tree
        self,
        event: DirectoryTree.FileSelected,
    ) -> None:
        event.stop()
        await self.open_audio_file(event.path)

    async def show_picker(self, start_path: Path) -> None: # Replace the current screen with the file picker
        assert self.body is not None
        self.close_current_player()
        await self.body.remove_children()
        self.home = None
        self.simulation = None
        self.recents = None
        self.controls = None
        self.picker = FilePickerWidget(start_path)
        await self.body.mount(self.picker)

    async def show_home(self) -> None: # Replace the current screen with the home menu
        assert self.body is not None
        self.close_current_player()
        await self.body.remove_children()
        self.home = HomeWidget()
        self.picker = None
        self.recents = None
        self.controls = None
        self.simulation = None
        await self.body.mount(self.home)

    async def show_recents(self) -> None: # Replace the current screen with recent files
        assert self.body is not None
        self.close_current_player()
        await self.body.remove_children()
        self.home = None
        self.picker = None
        self.controls = None
        self.simulation = None
        self.recents = RecentFilesWidget(self.recent_files.list())
        await self.body.mount(self.recents)

    async def show_visualizer( # Mount the visualizer screen for a loaded audio file
        self,
        audio: AudioData,
        player: AudioPlayer | None,
    ) -> None:
        assert self.body is not None
        if self.player is not None and self.player is not player: # Close any previous player before switching to a new one
            self.player.close()
        await self.body.remove_children()
        self.home = None
        self.picker = None
        self.recents = None
        self.audio = audio
        self.player = player
        self.recent_files.add(audio.path) # Save the opened file to recent history
        self.controls = ControlsBar()
        self.simulation = SimulationWidget(
            audio=audio,
            player=player,
            initial_style=self.initial_style,
            initial_mode=self.initial_mode,
        )
        await self.body.mount(self.controls, self.simulation)
        self.update_controls()
        self.start_player()

    async def open_audio_file(self, path: Path) -> None: # Load audio off the UI thread so the app stays responsive
        try:
            audio = await asyncio.to_thread(load_audio_file, path)
        except AudioLoadError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        await self.show_visualizer(audio, AudioPlayer(audio))

    def start_player(self) -> None: # Start playback and gracefully fallback to silent visuals on failure
        if self.player is None:
            return
        try:
            self.player.start()
        except AudioPlaybackError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            self.player.close()
            self.player = None
            if self.simulation is not None:
                self.simulation.player = None

    def close_current_player(self) -> None: # Stop and clear the current audio player before leaving the visualizer
        if self.player is not None:
            self.player.close()
        self.player = None
        self.audio = None

    def update_controls(self) -> None: # Refresh the controls bar with current track, state, mode and style
        if self.controls is None or self.simulation is None:
            return
        audio_name = self.audio.path.name if self.audio is not None else "no audio"
        state = "paused" if self.simulation.paused else "playing"
        self.controls.update(
            f"{audio_name} | {state} | mode {self.simulation.mode} | "
            f"style {self.simulation.style.name}\n"
            "Space pause | s style | m mode | b browse | r recents | h home | q quit"
        )

    def on_unmount(self) -> None: # Clean up the audio stream when the app is closed
        if self.player is not None:
            self.player.close()


def render_smoke_frame( # Render a deterministic frame for tests or CLI smoke checks without launching TUI
    width_cells: int = 48,
    height_cells: int = 16,
    *,
    style_name: str = DEFAULT_STYLE.name,
    mode: str = "particles",
) -> Text:
    """Render one deterministic frame without starting Textual."""

    style = style_by_name(style_name)
    mode = _validate_mode(mode)
    pixel_width = width_cells * 2
    pixel_height = height_cells * 4
    if mode == "particles": # Run a short deterministic particle simulation before rasterizing
        system = ParticleSystem(max_particles=1800, seed=12)
        for frame in range(36):
            cycle_pos = frame / 12.0
            system.step(
                1.0 / TARGET_FPS,
                style.shape_features(
                    AudioFeatures(
                        bass=0.35 + 0.60 * (frame % 12 == 0),
                        broadband=0.45 + 0.20 * sin(cycle_pos),
                        onset=frame % 12 == 0,
                        intensity=0.75 if frame % 12 == 0 else 0.35,
                    )
                ),
            )
        canvas, intensity = system.rasterize(
            pixel_width,
            pixel_height,
            threshold=style.threshold,
        )
    elif mode == "fluid": # Run a short deterministic fluid simulation before rasterizing
        fluid = FluidSimulation(pixel_width, pixel_height, iterations=5)
        for frame in range(42):
            cycle_pos = frame / 12.0
            fluid.step(
                1.0 / TARGET_FPS,
                style.shape_features(
                    AudioFeatures(
                        bass=0.32 + 0.58 * (frame % 14 == 0),
                        broadband=0.45 + 0.20 * sin(cycle_pos),
                        onset=frame % 14 == 0,
                        intensity=0.70 if frame % 14 == 0 else 0.36,
                    )
                ),
                profile=style.name,
            )
        canvas, intensity = fluid.rasterize(
            pixel_width,
            pixel_height,
            threshold=max(0.045, style.threshold * 0.72),
        )
    else: # Run a short deterministic wave simulation before rasterizing
        waves = WaveSimulation(pixel_width, pixel_height)
        for frame in range(44):
            cycle_pos = frame / 12.0
            waves.step(
                1.0 / TARGET_FPS,
                style.shape_features(
                    AudioFeatures(
                        bass=0.28 + 0.60 * (frame % 13 == 0),
                        broadband=0.40 + 0.24 * sin(cycle_pos),
                        onset=frame % 13 == 0,
                        intensity=0.74 if frame % 13 == 0 else 0.34,
                    )
                ),
            )
        canvas, intensity = waves.rasterize(
            pixel_width,
            pixel_height,
            threshold=max(0.040, style.threshold * 0.62),
        )
    styles = cell_styles_from_intensity(intensity, style.palette) # Apply palette styling before returning the smoke test frame
    return braille_canvas_to_text(canvas, styles)


def _validate_mode(mode: str) -> str: # Normalize and validate a requested simulation mode
    normalized = mode.strip().lower()
    if normalized not in SIMULATION_MODES:
        valid = "', '".join(SIMULATION_MODES)
        raise ValueError(f"mode must be one of '{valid}'")
    return normalized


def mode_names() -> tuple[str, ...]:
    return SIMULATION_MODES


def _is_directory(path: Path) -> bool: # Safely check whether a path is a directory
    try:
        return path.is_dir()
    except OSError:
        return False


def _recent_label(path: Path) -> str: # Format a recent file entry with filename and parent folder
    parent = path.parent
    return f"{path.name}\n{parent}"


def _fluid_grid_size(pixel_width: int, pixel_height: int) -> tuple[int, int]: # Scale down fluid grids when the terminal render size is too large
    scale = min(
        1.0,
        MAX_FLUID_GRID_WIDTH / max(pixel_width, 1),
        MAX_FLUID_GRID_HEIGHT / max(pixel_height, 1),
    )
    width = max(32, int(pixel_width * scale))
    height = max(24, int(pixel_height * scale))
    return width, height


def _wave_grid_size(pixel_width: int, pixel_height: int) -> tuple[int, int]: # Scale down wave grids when the terminal render size is too large
    scale = min(
        1.0,
        MAX_WAVE_GRID_WIDTH / max(pixel_width, 1),
        MAX_WAVE_GRID_HEIGHT / max(pixel_height, 1),
    )
    width = max(40, int(pixel_width * scale))
    height = max(28, int(pixel_height * scale))
    return width, height
