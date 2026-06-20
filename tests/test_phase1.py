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
from src.app import ReactBeatApp
from src.render.braille import pack_braille
from src.sim.particles import AudioFeatures, ParticleSystem


class BraillePackingTests(unittest.TestCase):
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
        }

        for (row, col), char in expected.items():
            with self.subTest(row=row, col=col):
                canvas = np.zeros((4, 2), dtype=bool)
                canvas[row, col] = True
                self.assertEqual(pack_braille(canvas), [char])

    def test_canvas_pads_to_full_braille_cells(self) -> None:
        canvas = np.ones((5, 3), dtype=bool)
        lines = pack_braille(canvas)
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(lines[0]), 2)


class ParticleSystemTests(unittest.TestCase):
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

        with TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            path.write_bytes(b"fake")
            with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
                audio = load_audio_file(path)

        self.assertEqual(audio.sample_rate, 48_000)
        self.assertEqual(audio.samples.dtype, np.float32)
        np.testing.assert_allclose(audio.mono, np.array([0.0, 0.5], dtype=np.float32))


class AudioPlayerTests(unittest.TestCase):
    def test_callback_tracks_position_and_fills_output(self) -> None:
        samples = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
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
        start = int(0.35 * sample_rate)
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
        frames = [
            analyzer.analyze_at(sample)
            for sample in range(0, len(mono), analyzer.hop_size)
        ]

        self.assertTrue(any(frame.onset for frame in frames))
        self.assertGreater(max(frame.bass for frame in frames), 0.5)

    def test_repeated_hop_does_not_repeat_onset(self) -> None:
        sample_rate = 8_000
        mono = np.zeros(sample_rate, dtype=np.float32)
        mono[3000:3600] = 1.0
        analyzer = EnergyAnalyzer(mono, sample_rate, window_size=512, hop_size=128)

        frame = analyzer.analyze_at(3200)
        repeated = analyzer.analyze_at(3201)

        self.assertEqual(frame.sample_index, 3200)
        self.assertFalse(repeated.onset)


class AppSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_textual_app_mounts_headlessly(self) -> None:
        app = ReactBeatApp()
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause(0.1)
            self.assertIsNotNone(app.simulation)


if __name__ == "__main__":
    unittest.main()
