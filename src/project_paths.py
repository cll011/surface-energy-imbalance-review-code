"""Load reviewer-local paths without committing machine-specific settings."""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "paths.example.json"
LOCAL_CONFIG = PROJECT_ROOT / "config" / "paths.local.json"


def load_paths() -> dict[str, Path]:
    config_path = Path(os.environ.get("ALBEDO_PATH_CONFIG", LOCAL_CONFIG))
    if not config_path.exists():
        config_path = DEFAULT_CONFIG
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {key: Path(value).expanduser() for key, value in payload.items()}


def get_path(key: str) -> Path:
    paths = load_paths()
    if key not in paths:
        raise KeyError(f"Path key '{key}' is absent from the configuration.")
    value = paths[key]
    if "PATH" in str(value).upper() or "WORK" in str(value).upper():
        raise RuntimeError(
            f"Configure '{key}' in config/paths.local.json before full-data execution."
        )
    return value

