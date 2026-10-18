"""GSM8K loading, splitting, and reading/writing split files."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_gsm8k(cfg: Mapping[str, Any]):
    """Loads GSM8K via HF datasets (from the HF_HOME cache)."""
    from datasets import load_dataset  # heavy import, only when needed

    d = cfg["data"]
    return load_dataset(d["dataset"], d["config"])


def make_split_indices(n: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Splits indices [0, n) into (val, train_rest); both are returned sorted.

    Uses Python's random module: reproducible and independent of the numpy version.
    """
    if not 0 < val_size < n:
        raise ValueError(f"invalid val_size={val_size} (n={n})")
    rng = random.Random(seed)
    val = sorted(rng.sample(range(n), val_size))
    val_set = set(val)
    rest = [i for i in range(n) if i not in val_set]
    return val, rest


def pick_fewshot(pool: Sequence[int], k: int, seed: int) -> list[int]:
    """Samples k examples from the pool; the returned order is the order in the prompt."""
    return random.Random(seed).sample(list(pool), k)


def content_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    """Order-sensitive sha256 digest of the rows' (question, answer) content."""
    h = hashlib.sha256()
    for r in rows:
        h.update(r["question"].encode("utf-8"))
        h.update(b"\x00")
        h.update(r["answer"].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_split(splits_dir: str | Path, name: str) -> list[int]:
    with open(Path(splits_dir) / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)["indices"]
