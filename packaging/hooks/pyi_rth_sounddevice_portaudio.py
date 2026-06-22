from __future__ import annotations

import ctypes.util
import os
import sys


_original_find_library = ctypes.util.find_library # Keep the orignal library resolver so we can fall back to normal system lookup
_bundle_base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)) # Locate the PyInstaller bundle directory, or fall back to the executable directory

_alsa_conf = os.path.join(_bundle_base, "share", "alsa", "alsa.conf") # Point to the bundled ALSA configuration file, if it exists
if os.path.exists(_alsa_conf): # Configure ASLA to use bundled configuration instead of relying on system paths
    os.environ.setdefault("ALSA_CONFIG_PATH", _alsa_conf)
    os.environ.setdefault("ALSA_CONFIG_DIR", os.path.dirname(_alsa_conf))


def _find_library(name: str) -> str | None: # Custom library resolver used to prefer bundled audio dependencies
    if name == "portaudio" or name.startswith("portaudio"): # Intercept PortAudio lookups so packaged builds can load the bundled shared library
        for filename in ("libportaudio.so.2", "libportaudio.so"): # Try common PortAudio shared library filenames in priority order
            candidate = os.path.join(_bundle_base, filename)
            if os.path.exists(candidate):
                return candidate
    return _original_find_library(name) # Fall back to platform's normal library search for everything else


ctypes.util.find_library = _find_library # Monkey patch ctypes so audio libraries resolve correctly inside the packaged app
