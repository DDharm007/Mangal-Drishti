"""Shared configuration loader and path helpers for MANGAL DRISHTI data modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path("configs/data_config.yaml")


def get_project_root() -> Path:
    """Return absolute path to the repository root."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load YAML configuration relative to project root or absolute path."""
    p = Path(config_path)
    if not p.is_absolute():
        p = get_project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path_value: str | Path, project_root: Path | None = None) -> Path:
    """Resolve a path relative to project root unless already absolute."""
    root = project_root or get_project_root()
    p = Path(path_value)
    return p if p.is_absolute() else (root / p)
