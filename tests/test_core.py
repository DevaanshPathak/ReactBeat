from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.audio.analysis import EnergyAnalyzer
from src.audio.loader import AudioData, load_audio_file
from src.audio.player import AudioPlayer
from src.app import FilePickerWidget, HomeWidget, RecentFilesWidget, ReactBeatApp, mode_names
from src.recent import RecentFiles
from src.render.braille import pack_braille
from src.render.styles import VISUAL_STYLES, style_by_name
from src.sim.fluid import FluidSimulation
from src.sim.particles import AudioFeatures, ParticleSystem
from src.sim.waves import WaveSimulation

# Tests that braille dot positions map to the correct Unicode characters
class BraillePackingTests(unittest.TestCase): # Expected Unicode braille character for each dot position in a 2x4 cell
    def test_standard_dot_mapping(self) -> None:
        expected = {
            (0, 0): "\u2801",
            (1, 0): "\u2802",
            (2, 0): "\u2804",
            (3, 0): "\u2840",
            (0, 1): "\u2808",
            (1, 1): "\u2810",
            (2, 1): "\u2820",
            (3, 1): "\u2880",
        } # Verify each single active pixel produces the matching braille dot

        for (row, col), char in expected.items():
            with self.subTest(row=row, col=col):
                canvas = np.zeros((4, 2), dtype=bool)
                canvas[row, col] = True
                self.assertEqual(pack_braille(canvas), [char]) # Ensure canvases with odd dimensions are padded into full braille cells

    def test_canvas_pads_to_full_braille_cells(self) -> None:
        canvas = np.ones((5, 3), dtype=bool)
        lines = pack_braille(canvas)
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(lines[0]), 2)


class ParticleSystemTests(unittest.TestCase): # Verify particles can step forward and rasterize into visible output
    def test_step_and_rasterize_produce_visible_canvas(self) -> None:
        system = ParticleSystem(max_particles=512, seed=3)
        for _ in range(4):
            system.step(
                1 / 30,
                AudioFeatures(bass=0.9, broadband=0.8, onset=True, intensity=0.9),
            )

        canvas, intensity = system.rasterize(80, 32)
        self.assertEqual(canvas.shape, (32, 80))
        self.assertEqual(intensity.shape, (32, 80))
        self.assertGreater(int(canvas.sum()), 0)


class AudioLoaderTests(unittest.TestCase):
    def test_loader_downmixes_to_mono_float32(self) -> None:
        stereo = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float32)
        fake_soundfile = SimpleNamespace(read=lambda *args, **kwargs: (stereo, 48_000))

        with TemporaryDirectory() as directory: # Mock soundfile so loader behaviour can be tested without a real decoder
            path = Path(directory) / "tone.wav"
            path.write_bytes(b"fake")
            with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
                audio = load_audio_file(path)

        self.assertEqual(audio.sample_rate, 48_000) # Confirm stereo audio is decoded as float32 and downmixed to mono correctly
        self.assertEqual(audio.samples.dtype, np.float32)
        np.testing.assert_allclose(audio.mono, np.array([0.0, 0.5], dtype=np.float32))


class AudioPlayerTests(unittest.TestCase):
    def test_callback_tracks_position_and_fills_output(self) -> None:
        samples = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], # Verify the playback callback advances postition and writes audio frames
            dtype=np.float32,
        )
        audio = AudioData(
            path=Path("tone.wav"),
            samples=samples,
            mono=samples.mean(axis=1),
            sample_rate=44_100,
        )
        player = AudioPlayer(audio)
        out = np.zeros((2, 2), dtype=np.float32)

        player._callback(out, 2, None, None)

        self.assertEqual(player.position_samples, 2)
        np.testing.assert_allclose(out, samples[:2])


class EnergyAnalyzerTests(unittest.TestCase):
    def test_detects_low_frequency_burst_onset(self) -> None:
        sample_rate = 8_000
        mono = np.zeros(sample_rate, dtype=np.float32)
        start = int(0.35 * sample_rate) # Build a synthetic low frequency burst to test onset detection
        burst_length = int(0.16 * sample_rate)
        time = np.arange(burst_length, dtype=np.float32) / sample_rate
        mono[start:start + burst_length] = 0.9 * np.sin(2 * np.pi * 90.0 * time)

        analyzer = EnergyAnalyzer(
            mono,
            sample_rate,
            window_size=512,
            hop_size=128,
            history_size=12,
            onset_multiplier=1.45,
        )
        frames = [ # Analyze each hop and confirm the burst produces a bass onset
            analyzer.analyze_at(sample)
            for sample in range(0, len(mono), analyzer.hop_size)
        ]

        self.assertTrue(any(frame.onset for frame in frames))
        self.assertGreater(max(frame.bass for frame in frames), 0.5)

    def test_repeated_hop_does_not_repeat_onset(self) -> None:
        sample_rate = 8_000
        mono = np.zeros(sample_rate, dtype=np.float32) # Ensure repeated analysis within the same hop does not duplicate an onset
        mono[3000:3600] = 1.0
        analyzer = EnergyAnalyzer(mono, sample_rate, window_size=512, hop_size=128)

        frame = analyzer.analyze_at(3200)
        repeated = analyzer.analyze_at(3201)

        self.assertEqual(frame.sample_index, 3200)
        self.assertFalse(repeated.onset)


class VisualStyleTests(unittest.TestCase):
    def test_has_five_named_styles(self) -> None:
        self.assertGreaterEqual(len(VISUAL_STYLES), 5)
        self.assertEqual(style_by_name("ember").name, "ember")
        self.assertEqual(style_by_name("prism").name, "prism")
        self.assertEqual(style_by_name("ghost").name, "ghost")

    def test_style_shapes_audio_features(self) -> None:
        features = AudioFeatures(bass=0.6, broadband=0.5, onset=True, intensity=0.7)
        voltage = style_by_name("voltage")
        shaped = voltage.shape_features(features) # Verify style gain shaping preserves onset and boosts configured features

        self.assertTrue(shaped.onset)
        self.assertGreaterEqual(shaped.bass, features.bass)
        self.assertGreaterEqual(shaped.intensity, features.intensity)


class FluidSimulationTests(unittest.TestCase):
    def test_fluid_step_and_rasterize_produce_visible_density(self) -> None:
        fluid = FluidSimulation(48, 32, iterations=4)
        for _ in range(3):
            fluid.step( # Confirm fluid simulation produces visible density after stepping
                1 / 30,
                AudioFeatures(bass=0.9, broadband=0.7, onset=True, intensity=0.9),
            )

        canvas, intensity = fluid.rasterize(48, 32)
        self.assertEqual(canvas.shape, (32, 48))
        self.assertEqual(intensity.shape, (32, 48))
        self.assertGreater(float(intensity.max()), 0.0)
        self.assertGreater(int(canvas.sum()), 0)

    def test_rasterize_can_resample_fluid_grid(self) -> None:
        fluid = FluidSimulation(32, 24, iterations=3)
        fluid.step(
            1 / 30,
            AudioFeatures(bass=0.5, broadband=0.6, onset=True, intensity=0.7),
        )

        canvas, intensity = fluid.rasterize(96, 40) # Verify fluid rasterization can resample to a different render size

        self.assertEqual(canvas.shape, (40, 96))
        self.assertEqual(intensity.shape, (40, 96))
        self.assertGreater(float(intensity.max()), 0.0)


class WaveSimulationTests(unittest.TestCase):
    def test_wave_step_and_rasterize_produce_visible_ripples(self) -> None:
        waves = WaveSimulation(64, 36)
        for _ in range(5):
            waves.step(
                1 / 30,
                AudioFeatures(bass=0.8, broadband=0.7, onset=True, intensity=0.9),
            )

        canvas, intensity = waves.rasterize(80, 40) # Confirm wave simulation produces visible ripple intensity after stepping

        self.assertEqual(canvas.shape, (40, 80))
        self.assertEqual(intensity.shape, (40, 80))
        self.assertGreater(float(intensity.max()), 0.0)
        self.assertGreater(int(canvas.sum()), 0)


class RecentFilesTests(unittest.TestCase):
    def test_recent_files_keep_existing_paths_first(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = RecentFiles(root / "recent.json", limit=3)
            first = root / "first.wav"
            second = root / "second.ogg"
            first.write_bytes(b"fake")
            second.write_bytes(b"fake")
            # Verify recent files are ordered newest first without duplicates
            store.add(first)
            store.add(second)
            store.add(first)

            self.assertEqual(store.list(), [first.resolve(), second.resolve()])


class AppSmokeTests(unittest.IsolatedAsyncioTestCase):
    def make_recent_store(self) -> tuple[TemporaryDirectory[str], RecentFiles]:
        directory = TemporaryDirectory()
        store = RecentFiles(Path(directory.name) / "recent.json")
        return directory, store

    @staticmethod
    def make_audio(path: Path) -> AudioData:
        samples = np.zeros((128, 2), dtype=np.float32)
        return AudioData(
            path=path, # Create an isolated recent files store for app tests
            samples=samples,
            mono=samples.mean(axis=1),
            sample_rate=44_100,
        )

    async def test_textual_app_mounts_home_without_audio(self) -> None: # Build minimal silent audio data for textual app smoke tests
        temp_dir, store = self.make_recent_store()
        self.addCleanup(temp_dir.cleanup)
        app = ReactBeatApp(recent_files=store)
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause(0.1)
            self.assertIsInstance(app.home, HomeWidget)
            self.assertIsNone(app.picker)
            self.assertIsNone(app.simulation)

    async def test_browse_action_mounts_file_picker(self) -> None: # Verify the app opens the home screen when no audio is provided
        temp_dir, store = self.make_recent_store()
        self.addCleanup(temp_dir.cleanup)
        app = ReactBeatApp(recent_files=store)
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.press("b")
            await pilot.pause(0.1)
            self.assertIsInstance(app.picker, FilePickerWidget)
            self.assertIsNone(app.home)

    async def test_recent_action_mounts_recent_files(self) -> None: # Verify the browse shortcut replaces home with the file picker
        temp_dir, store = self.make_recent_store()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        recent_file = root / "track.wav"
        recent_file.write_bytes(b"fake")
        store.add(recent_file)

        app = ReactBeatApp(recent_files=store)
        async with app.run_test(size=(60, 20)) as pilot: # Verify the recent files shortcut shows saved recent audio paths
            await pilot.press("r")
            await pilot.pause(0.1)
            self.assertIsInstance(app.recents, RecentFilesWidget)
            self.assertEqual(app.recents.recent_paths, [recent_file.resolve()])

    async def test_textual_app_mounts_visualizer_with_audio(self) -> None:
        temp_dir, store = self.make_recent_store()
        self.addCleanup(temp_dir.cleanup)
        audio_path = Path(temp_dir.name) / "tone.wav"
        audio_path.write_bytes(b"fake")
        audio = self.make_audio(audio_path)
        app = ReactBeatApp(audio=audio, recent_files=store)
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause(0.1)
            self.assertIsNotNone(app.simulation) # Verify the app mounts the visualizer when audio is provided
            self.assertIsNone(app.picker)
            self.assertEqual(store.list(), [audio_path.resolve()])

    async def test_style_can_cycle_without_restarting_app(self) -> None:
        temp_dir, store = self.make_recent_store()
        self.addCleanup(temp_dir.cleanup)
        audio_path = Path(temp_dir.name) / "tone.wav"
        audio_path.write_bytes(b"fake")
        app = ReactBeatApp(audio=self.make_audio(audio_path), recent_files=store)
        async with app.run_test(size=(60, 20)) as pilot:
            self.assertIsNotNone(app.simulation)
            before = app.simulation.style.name
            await pilot.press("s") # Verify styles can cycle through the UI without restarting
            after = app.simulation.style.name
            self.assertNotEqual(before, after)

    async def test_mode_can_cycle_without_restarting_app(self) -> None:
        temp_dir, store = self.make_recent_store()
        self.addCleanup(temp_dir.cleanup)
        audio_path = Path(temp_dir.name) / "tone.wav"
        audio_path.write_bytes(b"fake")
        app = ReactBeatApp(audio=self.make_audio(audio_path), recent_files=store)
        async with app.run_test(size=(60, 20)) as pilot:
            self.assertIsNotNone(app.simulation)
            before = app.simulation.mode
            await pilot.press("m") # Verify modes cycle through particles, fluid and waves then back to particles
            second = app.simulation.mode
            await pilot.press("m")
            third = app.simulation.mode
            await pilot.press("m")
            fourth = app.simulation.mode
            self.assertEqual((before, second, third, fourth), mode_names() + ("particles",))


if __name__ == "__main__":
    unittest.main()
