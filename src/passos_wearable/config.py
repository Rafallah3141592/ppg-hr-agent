"""Configuration loading (YAML + environment overrides)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    p = Path(path) if path else PROJECT_ROOT / ".env"
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the YAML config. Precedence: ``overrides`` > env vars > file."""
    cfg_path = Path(path or os.environ.get("PASSOS_CONFIG") or DEFAULT_CONFIG_PATH)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    if os.environ.get("PASSOS_DATA_DIR"):
        cfg.setdefault("data", {})["dir"] = os.environ["PASSOS_DATA_DIR"]
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(p: str | Path) -> Path:
    """Relative paths are resolved against the project root (not the cwd)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p
