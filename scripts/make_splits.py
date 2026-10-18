"""GSM8K splits: train (7473) -> val (300) + train_rest (7173); test is left untouched.

Outputs (committed):
  data/splits/val.json, train_rest.json, fewshot_4shot.json, manifest.json
If the existing files would change, aborts unless --force is given (prevents
committed splits from changing silently).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kdq.answer import extract_gold  # noqa: E402
from kdq.config import load_config, repo_path  # noqa: E402
from kdq.data import (  # noqa: E402
    content_sha256, load_gsm8k, make_split_indices, pick_fewshot, save_json,
)
from kdq.utils import run_metadata  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sp, dcfg = cfg["split"], cfg["data"]
    out_dir = repo_path(cfg["paths"]["splits_dir"])

    ds = load_gsm8k(cfg)
    train, test = ds["train"], ds["test"]
    assert len(train) == dcfg["train_size"], f"train size {len(train)} != {dcfg['train_size']}"
    assert len(test) == dcfg["test_size"], f"test size {len(test)} != {dcfg['test_size']}"

    val_idx, rest_idx = make_split_indices(len(train), sp["val_size"], sp["seed"])
    shot_idx = pick_fewshot(rest_idx, sp["fewshot_k"], sp["fewshot_seed"])
    assert not set(val_idx) & set(rest_idx)
    assert set(shot_idx) <= set(rest_idx)

    def rows(idx):
        return [train[i] for i in idx]

    def split_obj(name, idx, seed, extra=None):
        return {
            "name": name,
            "source_split": "train",
            "seed": seed,
            "n": len(idx),
            "sha256": content_sha256(rows(idx)),
            **(extra or {}),
            "indices": idx,
        }

    k = sp["fewshot_k"]
    files = {
        "val": split_obj("val", val_idx, sp["seed"]),
        "train_rest": split_obj("train_rest", rest_idx, sp["seed"]),
        f"fewshot_{k}shot": split_obj(
            f"fewshot_{k}shot", shot_idx, sp["fewshot_seed"],
            {"note": "drawn from train_rest; order is the order in the prompt",
             "gold": [extract_gold(train[i]["answer"]) for i in shot_idx]},
        ),
    }
    files["manifest"] = {
        "dataset": dcfg["dataset"],
        "config": dcfg["config"],
        "train_n": len(train),
        "test_n": len(test),
        "train_sha256": content_sha256(list(train)),
        "test_sha256": content_sha256(list(test)),
        "train_fingerprint": getattr(train, "_fingerprint", None),
        "test_fingerprint": getattr(test, "_fingerprint", None),
        "test_note": "test split is untouched; final evaluation only",
        "split_sha256": {n: f["sha256"] for n, f in files.items()},
        "meta": run_metadata(None, script="make_splits.py"),
    }

    # Consistency with the existing committed splits (excluding the manifest timestamp)
    changed = []
    for name, obj in files.items():
        p = out_dir / f"{name}.json"
        if name != "manifest" and p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            if old.get("indices") != obj["indices"] or old.get("sha256") != obj["sha256"]:
                changed.append(name)
    if changed and not args.force:
        print(f"[ERROR] Existing splits differ: {changed}. Use --force to change them deliberately.")
        return 1

    for name, obj in files.items():
        save_json(obj, out_dir / f"{name}.json")

    print(f"train={len(train)} -> val={len(val_idx)} + train_rest={len(rest_idx)} | test={len(test)} (untouched)")
    print(f"{k}-shot indices: {shot_idx}")
    for name, obj in files.items():
        if name != "manifest":
            print(f"  {name:14s} n={obj['n']:5d} sha256={obj['sha256'][:16]}…")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
