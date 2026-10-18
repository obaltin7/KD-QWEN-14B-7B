"""Cache the general-benchmark datasets and record the subset indices.

  MMLU (cais/mmlu, all, test)          -> 1000, stratified proportionally by subject, seed 42
  HellaSwag (Rowan/hellaswag, val.)    -> 1000, simple random, seed 42
  ARC-Challenge (test) and WikiText-2 (test) -> full set (only a content hash is recorded)

Outputs (committed): data/splits/general_mmlu.json, general_hellaswag.json,
general_manifest.json. Aborts if a recorded split would change; use --force to overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kdq.config import load_config, repo_path  # noqa: E402
from kdq.data import save_json  # noqa: E402
from kdq.general import (  # noqa: E402
    TASKS, group_counts, load_task_dataset, rows_sha256, simple_sample, stratified_sample,
)
from kdq.utils import run_metadata  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    g = cfg["general"]
    seed = g["seed"]
    out_dir = repo_path(cfg["paths"]["splits_dir"])

    ds = {}
    for task in TASKS:
        ds[task] = load_task_dataset(cfg, task)
        print(f"[subsets] {task}: {len(ds[task])} rows ({g['datasets'][task]['path']})", flush=True)

    def src(task):
        d = g["datasets"][task]
        return {"path": d["path"], "name": d.get("name"), "split": d["split"]}

    mmlu = ds["mmlu"]
    subjects = list(mmlu["subject"])
    m_idx = stratified_sample(subjects, g["datasets"]["mmlu"]["n"], seed)
    h_idx = simple_sample(len(ds["hellaswag"]), g["datasets"]["hellaswag"]["n"], seed)

    files = {
        "general_mmlu": {
            "name": "general_mmlu", "source": src("mmlu"), "seed": seed,
            "sampling": "stratified proportionally by subject, largest-remainder method",
            "n": len(m_idx), "n_source": len(mmlu),
            "sha256": rows_sha256(mmlu.select(m_idx)),
            "per_subject": group_counts(subjects, m_idx),
            "indices": m_idx,
        },
        "general_hellaswag": {
            "name": "general_hellaswag", "source": src("hellaswag"), "seed": seed,
            "sampling": "simple random",
            "n": len(h_idx), "n_source": len(ds["hellaswag"]),
            "sha256": rows_sha256(ds["hellaswag"].select(h_idx)),
            "indices": h_idx,
        },
    }

    changed = []
    for name, obj in files.items():
        p = out_dir / f"{name}.json"
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            if old.get("indices") != obj["indices"] or old.get("sha256") != obj["sha256"]:
                changed.append(name)
    if changed and not a.force:
        print(f"[ERROR] Recorded splits differ: {changed}. Use --force to change them deliberately.")
        return 1

    files["general_manifest"] = {
        "tasks": {t: {**src(t), "n_source": len(ds[t]), "sha256_full": rows_sha256(ds[t])} for t in TASKS},
        "subset_sha256": {n: f["sha256"] for n, f in files.items()},
        "meta": run_metadata(None, script="make_general_subsets.py"),
    }
    for name, obj in files.items():
        save_json(obj, out_dir / f"{name}.json")
    print(f"[subsets] MMLU {len(m_idx)} ({len(files['general_mmlu']['per_subject'])} subjects) | "
          f"HellaSwag {len(h_idx)} | ARC-C {len(ds['arc_c'])} | WikiText-2 {len(ds['wikitext2'])} rows -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
