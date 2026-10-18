"""(Re)score all runs under results/day1 from raw generations with answer extraction v2.

Requires no GPU/model. Source for each run prefix <prefix>:
  (a) if <prefix>.jsonl + <prefix>.summary.json exist: these (on the first pass the v1
      values are kept as pred_v1/correct_v1 and the v1 summary as <prefix>.summary.v1.json);
  (b) otherwise <prefix>.raw.jsonl + <prefix>.raw.meta.json: runs that crashed during
      scoring are completed from these;
  (c) if neither exists (incomplete run), the prefix is skipped.
Per-question scoring errors do not fail the run; they are counted in n_score_errors in the summary.
engine_crosscheck.json is regenerated at the end. Idempotent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kdq.data import save_json  # noqa: E402
from kdq.evaluation import SCORING_VERSION, write_scored  # noqa: E402

RAW_SUFFIX = ".raw.jsonl"


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sibling(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def backup_once(path: Path, data: dict) -> None:
    v1 = path.with_name(path.name.removesuffix(".json") + ".v1.json")
    if not v1.exists() and data.get("scoring_version") != SCORING_VERSION:
        shutil.copy2(path, v1)


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_prefixes(base: Path) -> list[Path]:
    out = set()
    for p in base.rglob("*.jsonl"):
        name = p.name
        stem = name.removesuffix(RAW_SUFFIX) if name.endswith(RAW_SUFFIX) else name.removesuffix(".jsonl")
        out.add(p.with_name(stem))
    return sorted(out)


def rescore_prefix(prefix: Path) -> dict | None:
    scored, summ = sibling(prefix, ".jsonl"), sibling(prefix, ".summary.json")
    raw, raw_meta = sibling(prefix, RAW_SUFFIX), sibling(prefix, ".raw.meta.json")
    if scored.exists() and summ.exists():
        source = "jsonl"
        base = json.loads(summ.read_text(encoding="utf-8"))
        backup_once(summ, base)
        records = []
        for r in read_jsonl(scored):
            if "pred_flexible" not in r and "pred" in r:
                r["pred_v1"], r["correct_v1"] = r["pred"], r.get("correct")
            records.append(r)
    elif raw.exists() and raw_meta.exists():
        source = "raw"
        base = json.loads(raw_meta.read_text(encoding="utf-8"))
        records = read_jsonl(raw)
    else:
        print(f"[SKIP] {rel(prefix)} (no summary/raw meta; possibly an incomplete run)")
        return None
    old_acc = base.get("accuracy")
    s = write_scored(prefix, records, {**base, "scoring_note":
                     f"answer extraction v2 (strict + flexible); source: {source}"})
    print(f"{rel(prefix):60s} [{source}] n={s['n']:5d} previous={old_acc if old_acc is None else round(old_acc, 4)} -> "
          f"strict={s['accuracy_strict']:.4f} flexible={s['accuracy_flexible']:.4f} "
          f"format_error={s['format_error_rate']:.3f} scoring_errors={s['n_score_errors']}")
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="results/day1")
    a = ap.parse_args()
    base = ROOT / a.dir

    prefixes = run_prefixes(base)
    if not prefixes:
        print(f"[WARNING] no JSONL under {base}.")
    for p in prefixes:
        rescore_prefix(p)

    # Cross-check report
    spec = importlib.util.spec_from_file_location("engine_crosscheck", ROOT / "scripts/engine_crosscheck.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    in_scope = cc.OUT_DIR.resolve().is_relative_to(base.resolve())   # only if under --dir
    if in_scope and all(sibling(cc.prefix_for(e), ".jsonl").exists() for e in ("vllm", "hf")):
        if cc.REPORT.exists():
            backup_once(cc.REPORT, json.loads(cc.REPORT.read_text(encoding="utf-8")))
        rep = cc.build_report()
        save_json(rep, cc.REPORT)
        cc.print_report(rep)
    else:
        print("[SKIP] engine_crosscheck: outputs of both engines are not available")
    print("Next step: python scripts/gap_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
