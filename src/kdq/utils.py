"""Seed fixing and run metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from .config import REPO_ROOT

TRACKED_PACKAGES = (
    "torch", "transformers", "peft", "accelerate", "datasets", "vllm",
    "tokenizers", "huggingface-hub", "numpy", "pandas", "scipy", "pyyaml",
)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fixes random / numpy / torch seeds (skips torch if unavailable)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def adapter_tag(adapter: str | Path) -> str:
    """Output name from an adapter path: .../S2_seed42/adapter -> S2_seed42."""
    p = Path(adapter)
    return p.parent.name if p.name == "adapter" and p.parent.name else p.name


def lr_tag(lr: float) -> str:
    """Short tag in 1e-4 form: 0.0001 -> '1e-4', 0.0002 -> '2e-4', 5e-05 -> '5e-5'."""
    m, e = f"{lr:.2e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}"


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions(names=TRACKED_PACKAGES) -> dict[str, str | None]:
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def gpu_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    n = torch.cuda.device_count()
    return {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "device_count": n,
        "names": [torch.cuda.get_device_name(i) for i in range(n)],
    }


def config_summary(cfg: Mapping[str, Any]) -> dict[str, Any]:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return {"sha256": hashlib.sha256(blob).hexdigest(), "config": cfg}


def run_metadata(cfg: Mapping[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_info(),
    }
    if cfg is not None:
        meta["config"] = config_summary(cfg)
    meta.update(extra)
    return meta


def write_run_metadata(path: str | Path, cfg: Mapping[str, Any] | None = None, **extra: Any) -> dict:
    meta = run_metadata(cfg, **extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    return meta
