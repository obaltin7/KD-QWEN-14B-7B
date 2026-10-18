"""Loading of configs/base.yaml and path resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """Recursively expands ``${VAR}`` and ``${VAR:-default}`` references in string values."""
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Returns the YAML configuration as a dictionary.

    Environment references (``${WORKSPACE:-/workspace}``) are expanded. If the HF_HOME environment
    variable is set, it overrides ``paths.hf_home``.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = expand_env(yaml.safe_load(f))
    cfg.setdefault("paths", {})
    if os.environ.get("HF_HOME"):
        cfg["paths"]["hf_home"] = os.environ["HF_HOME"]
    resolved = cfg_path.resolve()
    cfg["_config_path"] = (resolved.relative_to(REPO_ROOT).as_posix() if resolved.is_relative_to(REPO_ROOT)
                           else str(resolved))
    return cfg


def repo_path(rel: str | os.PathLike) -> Path:
    """Resolves a relative path against the repository root; absolute paths are returned unchanged."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p
