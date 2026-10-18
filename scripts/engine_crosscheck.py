"""vllm <-> hf engine cross-check: 7B-base, 0-shot, first 50 test questions.

Each engine runs in a separate process (so GPU memory is released); an engine
whose output exists is not rerun. Report: results/day1/engine_crosscheck.json
(strict and flexible scoring reported separately).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kdq.data import save_json  # noqa: E402
from kdq.evaluation import crosscheck_report, score_safe  # noqa: E402

MODEL, SHOTS, LIMIT = "student_base", 0, 50
OUT_DIR = ROOT / "results/day1/crosscheck"
REPORT = ROOT / "results/day1/engine_crosscheck.json"
WARN_BELOW = 0.90


def prefix_for(eng: str) -> Path:
    return OUT_DIR / f"{MODEL}_{SHOTS}shot_{eng}_n{LIMIT}"


def load_jsonl(p: Path) -> dict[int, dict]:
    """Read records; v1 records are rescored with v2 from the raw generation."""
    with open(p, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return {r["idx"]: {**r, **score_safe(r["generation"], r["gold"])} for r in recs}


def build_report() -> dict:
    v = load_jsonl(prefix_for("vllm").with_name(prefix_for("vllm").name + ".jsonl"))
    h = load_jsonl(prefix_for("hf").with_name(prefix_for("hf").name + ".jsonl"))
    rep = {"model": MODEL, "shots": SHOTS, **crosscheck_report(v, h)}
    rep["note"] = ("small text deviations are expected in bf16 due to batching/kernel differences; "
                   f"a warning is issued if flexible answer agreement < {WARN_BELOW:.0%}; the pipeline does not stop.")
    return rep


def print_report(rep: dict) -> None:
    print(f"[crosscheck] n={rep['n']} | text exact match={rep['text_exact_match']:.3f}")
    for mode in ("strict", "flexible"):
        r = rep[mode]
        print(f"  {mode:8s} accuracy vllm={r['accuracy_vllm']:.3f} hf={r['accuracy_hf']:.3f} "
              f"(diff {r['accuracy_diff_points']:+.1f} points) | correct/incorrect agreement={r['correct_agreement']:.3f} "
              f"| answer agreement={r['pred_agreement']:.3f}")
    if rep["flexible"]["pred_agreement"] < WARN_BELOW:
        print(f"[WARNING] flexible answer agreement below {WARN_BELOW:.0%}.")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for eng in ("vllm", "hf"):
        prefix = prefix_for(eng)
        if prefix.with_name(prefix.name + ".summary.json").exists():
            print(f"[crosscheck] {eng}: using existing output")
            continue
        cmd = [sys.executable, str(ROOT / "scripts/eval_gsm8k.py"), "--model", MODEL,
               "--shots", str(SHOTS), "--limit", str(LIMIT), "--engine", eng, "--out", str(prefix)]
        print("[crosscheck] $", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    rep = build_report()
    save_json(rep, REPORT)
    print_report(rep)
    print(f"-> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
