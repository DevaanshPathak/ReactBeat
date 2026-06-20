# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import ctypes.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


def _first_existing_library(name: str) -> tuple[str, str] | None:
    library = ctypes.util.find_library(name)
    search_roots = (
        Path("/lib"),
        Path("/usr/lib"),
        Path("/usr/local/lib"),
        Path("/lib/x86_64-linux-gnu"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/lib/x86_64-linux-gnu"),
    )
    patterns = []
    if library:
        patterns.append(library + "*")
    patterns.append(f"lib{name}.so*")

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for candidate in root.glob(pattern):
                if candidate.is_file():
                    return str(candidate), "."
    return None


def _collect_tree(source: Path, target: str) -> list[tuple[str, str]]:
    if not source.exists():
        return []
    return [
        (str(path), str(Path(target) / path.relative_to(source).parent))
        for path in source.rglob("*")
        if path.is_file()
    ]


binaries = []
datas = []

datas += collect_data_files("soundfile")
datas += _collect_tree(Path("/usr/share/alsa"), "share/alsa")
binaries += collect_dynamic_libs("soundfile")

portaudio = _first_existing_library("portaudio")
if portaudio is not None:
    binaries.append(portaudio)

a = Analysis(
    ["packaging/pyinstaller_entry.py"],
    pathex=[str(Path.cwd())],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "_cffi_backend",
        "sounddevice",
        "soundfile",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["packaging/hooks/pyi_rth_sounddevice_portaudio.py"],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="reactbeat",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
