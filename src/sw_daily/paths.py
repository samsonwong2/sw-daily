"""Runtime path defaults for the Shenwan industry index project.

Machine-specific paths live in ``configs/sw_daily.json`` under the top-level
``paths`` object. An empty or omitted value keeps the default derived from
the project location and the home directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sw_daily.json"


def _default_paths() -> dict[str, str]:
    home = Path.home()
    return {
        "info_dir": str(home / "temp" / "sw"),
        "qlib_dir": str(home / "data" / "qlib_data" / "sw_index_data"),
        "qlib_scripts_dir": str(home / "python" / "qlib" / "scripts"),
        "log_dir": str(PROJECT_ROOT / "logs"),
    }


def _optional_path(value: object) -> Path | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _load_config_paths() -> dict[str, Any]:
    if not DEFAULT_CONFIG_PATH.exists():
        return {}
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"Config paths must be a JSON object: {DEFAULT_CONFIG_PATH}")
    return paths


def _expand_paths(paths: dict[str, Any]) -> dict[str, str]:
    expanded: dict[str, str] = dict(_default_paths())
    for key, value in paths.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        expanded[str(key)] = text
    expanded["project_root"] = str(PROJECT_ROOT)
    for _ in range(8):
        changed = False
        context = dict(expanded)
        for key, value in list(expanded.items()):
            formatted = value.format(**context)
            if formatted != value:
                expanded[key] = formatted
                changed = True
        if not changed:
            break
    return expanded


_PATHS = _expand_paths(_load_config_paths())

INFO_DIR = Path(_PATHS["info_dir"]).expanduser().resolve()
QLIB_DIR = Path(_PATHS["qlib_dir"]).expanduser().resolve()
CSV_DIR = QLIB_DIR / "csv"
LOG_DIR = Path(_PATHS["log_dir"]).expanduser().resolve()
QLIB_SCRIPTS_DIR = _optional_path(_PATHS.get("qlib_scripts_dir"))
FIRST_INFO_CSV = INFO_DIR / "sw_index_first_info.csv"
SECOND_INFO_CSV = INFO_DIR / "sw_index_second_info.csv"
POOL_DIR = INFO_DIR / "pool"
REGIME_DIR = INFO_DIR / "regime"


def as_str(path: str | Path) -> str:
    return str(Path(path).expanduser())
