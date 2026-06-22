from __future__ import annotations

import json
import os
from pathlib import Path


MAX_RECENT_FILES = 12 # Max number of recent audio files to remember


class RecentFiles:
    """Persistent recent audio file list."""

    def __init__(self, path: Path | None = None, *, limit: int = MAX_RECENT_FILES) -> None:
        self.path = path or default_recent_path()
        self.limit = max(1, int(limit))

    def list(self) -> list[Path]: # Return existing recent files, newest first, with duplicates removed
        paths = []
        for value in self._read_values():
            path = Path(value).expanduser()
            if path.is_file() and path not in paths: # Keep only files that still exist and have not already been added
                paths.append(path)
        return paths[: self.limit]

    def add(self, path: Path) -> None: # Add a file to the top of recent list
        resolved = path.expanduser().resolve()
        values = [str(resolved)]
        values.extend(str(item) for item in self.list() if item.resolve() != resolved) # Preserve previous recent files whe removing the file being re-added
        self._write_values(values[: self.limit])

    def _read_values(self) -> list[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): # Treat missing, unreadable, or invalid JSON files as an empty history
            return []
        if not isinstance(data, list):
            return []
        return [value for value in data if isinstance(value, str)]

    def _write_values(self, values: list[str]) -> None: # Persist recent file values to disk as formatted JSON
        self.path.parent.mkdir(parents=True, exist_ok=True) # Create the state directory if it does not already exist
        self.path.write_text(
            json.dumps(values, indent=2),
            encoding="utf-8",
        )


def default_recent_path() -> Path: # Choose the default recent files location for the current platform
    if os.name == "nt": # On windows, prefer the APPDATA folder
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "reactbeat" / "recent.json"

    state_home = os.environ.get("XDG_STATE_HOME") # On unix-like system, respect XDG_STATE_HOME when set
    if state_home:
        return Path(state_home) / "reactbeat" / "recent.json"

    return Path.home() / ".local" / "state" / "reactbeat" / "recent.json" # Fall back to the standard local state directory under the user's home folder
