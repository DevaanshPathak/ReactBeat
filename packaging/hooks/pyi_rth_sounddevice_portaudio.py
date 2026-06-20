from __future__ import annotations

import ctypes.util
import os
import sys


_original_find_library = ctypes.util.find_library
_bundle_base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))

_alsa_conf = os.path.join(_bundle_base, "share", "alsa", "alsa.conf")
if os.path.exists(_alsa_conf):
    os.environ.setdefault("ALSA_CONFIG_PATH", _alsa_conf)
    os.environ.setdefault("ALSA_CONFIG_DIR", os.path.dirname(_alsa_conf))


def _find_library(name: str) -> str | None:
    if name == "portaudio" or name.startswith("portaudio"):
        for filename in ("libportaudio.so.2", "libportaudio.so"):
            candidate = os.path.join(_bundle_base, filename)
            if os.path.exists(candidate):
                return candidate
    return _original_find_library(name)


ctypes.util.find_library = _find_library
