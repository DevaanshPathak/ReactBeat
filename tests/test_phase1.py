from __future__ import annotations

import unittest

import numpy as np

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


class AppSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_textual_app_mounts_headlessly(self) -> None:
        app = ReactBeatApp()
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause(0.1)
            self.assertIsNotNone(app.simulation)


if __name__ == "__main__":
    unittest.main()
