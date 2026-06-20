from __future__ import annotations

from math import sin
from time import monotonic

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header
from textual.widget import Widget

from .render.braille import braille_canvas_to_text
from .render.styles import DEFAULT_PALETTE, cell_styles_from_intensity
from .sim.particles import AudioFeatures, ParticleSystem


TARGET_FPS = 30.0


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

    def __init__(self, *, seed: int = 4, max_particles: int = 5000) -> None:
        super().__init__()
        self.system = ParticleSystem(max_particles=max_particles, seed=seed)
        self.paused = False
        self._started_at = monotonic()
        self._last_tick = self._started_at
        self._frame = Text("")

    def on_mount(self) -> None:
        self.set_interval(1.0 / TARGET_FPS, self.tick)
        self.focus()

    def tick(self) -> None:
        now = monotonic()
        dt = min(max(now - self._last_tick, 0.0), 1.0 / 12.0)
        self._last_tick = now

        if not self.paused:
            features = self._phase_one_features(now - self._started_at, dt)
            self.system.step(dt, features)

        width_cells = max(self.size.width, 20)
        height_cells = max(self.size.height, 8)
        canvas, intensity = self.system.rasterize(
            width=width_cells * 2,
            height=height_cells * 4,
        )
        styles = cell_styles_from_intensity(intensity, DEFAULT_PALETTE)
        self._frame = braille_canvas_to_text(canvas, styles)
        self.refresh()

    def render(self) -> Text:
        return self._frame

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    @staticmethod
    def _phase_one_features(elapsed: float, dt: float) -> AudioFeatures:
        pulse_rate = 1.6
        phase = (elapsed * pulse_rate) % 1.0
        pulse = max(0.0, 1.0 - phase * 5.0) ** 3
        wave = 0.5 + 0.5 * sin(elapsed * 2.7)
        beat = phase < max(dt * pulse_rate, 0.035)
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

    Header, Footer {
        background: #111111;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("space", "toggle_pause", "Pause"),
    ]

    TITLE = "reactbeat"
    SUB_TITLE = "Phase 1 particle renderer"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self.simulation = SimulationWidget()
        yield self.simulation
        yield Footer()

    def action_toggle_pause(self) -> None:
        self.simulation.toggle_pause()


def render_smoke_frame(width_cells: int = 48, height_cells: int = 16) -> Text:
    """Render one deterministic frame without starting Textual."""

    system = ParticleSystem(max_particles=1800, seed=12)
    for frame in range(36):
        phase = frame / 12.0
        system.step(
            1.0 / TARGET_FPS,
            AudioFeatures(
                bass=0.35 + 0.60 * (frame % 12 == 0),
                broadband=0.45 + 0.20 * sin(phase),
                onset=frame % 12 == 0,
                intensity=0.75 if frame % 12 == 0 else 0.35,
            ),
        )

    canvas, intensity = system.rasterize(width_cells * 2, height_cells * 4)
    styles = cell_styles_from_intensity(intensity, DEFAULT_PALETTE)
    return braille_canvas_to_text(canvas, styles)
