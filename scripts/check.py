from __future__ import annotations

import math
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commands = [ # Run the main validation suite: unit tests, CLI smoke tests, and diagnostics
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        [
            sys.executable,
            "-m",
            "src.cli",
            "--smoke-test",
            "--mode",
            "particles",
            "--style",
            "ember",
            "--width",
            "18",
            "--height",
            "6",
        ],
        [
            sys.executable,
            "-m",
            "src.cli",
            "--smoke-test",
            "--mode",
            "fluid",
            "--style",
            "aurora",
            "--width",
            "18",
            "--height",
            "6",
        ],
        [
            sys.executable,
            "-m",
            "src.cli",
            "--smoke-test",
            "--mode",
            "waves",
            "--style",
            "prism",
            "--width",
            "18",
            "--height",
            "6",
        ],
        [sys.executable, "-m", "src.cli", "--diagnostics"],
    ]

    with tempfile.TemporaryDirectory() as temp_dir: # Create a temporary audio file for checking audio-input handling without needing test assets
        audio_path = Path(temp_dir) / "reactbeat-check.wav"
        write_tone(audio_path) # Generate a short synthetic WAV tone used by the audio checker
        commands.append([sys.executable, "-m", "src.cli", str(audio_path), "--check-audio"])

        for command in commands: # Execute each validation command and stop immediately on the first failure
            print("$ " + " ".join(command), flush=True) # Print the command before running it to make CI/debug logs easier to follow
            completed = subprocess.run(command, cwd=root)
            if completed.returncode != 0:
                return completed.returncode

    return 0


def write_tone(path: Path) -> None: # Write a short stereo sine wave WAV file for audio validation
    sample_rate = 22_050 # Use a modest sample rate to keep the generated test file small
    frames = int(sample_rate * 0.15) # Generate 0.15 sounds of audio, enough for a quick smoke check
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(frames):
            sample = int( # Scale the sine wave to low volume to avoid clipping
                0.08
                * 32767
                * math.sin(2 * math.pi * 220.0 * index / sample_rate)
            )
            wav.writeframes(struct.pack("<hh", sample, sample)) # Write the same sample to both left and right channels


if __name__ == "__main__":
    raise SystemExit(main())
