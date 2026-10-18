"""General-benchmark report: model x task table -> results/general/general_report.{md,json}.

MMLU: acc; HellaSwag and ARC-C: acc_norm (primary) and acc; WikiText-2: perplexity.
Accuracies with Wilson 95% CIs. Smoke-test directories (*_limitN) are excluded by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kdq.config import load_config, repo_path  # noqa: E402
from kdq.data import save_json  # noqa: E402
from kdq.general import TASKS  # noqa: E402
from kdq.utils import run_metadata  # noqa: E402

LABELS = {"mmlu": "MMLU", "hellaswag": "HellaSwag", "arc_c": "ARC-Challenge", "wikitext2": "WikiText-2"}
MODEL_ORDER = ("teacher", "student_instruct", "student_base")


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def ci(c):
    return "—" if not c else f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def model_sort_key(name: str):
    for i, m in enumerate(MODEL_ORDER):
        if name == m:
            return (i, name)
    return (len(MODEL_ORDER), name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--include-smoke", action="store_true", help="also include *_limitN directories")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    base = repo_path(cfg["general"]["out_dir"])
    table: dict[str, dict] = {}
    if base.exists():
        for d in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: model_sort_key(p.name)):
            if "_limit" in d.name and not a.include_smoke:
                continue
            for t in TASKS:
                sp = d / f"{t}.summary.json"
                if sp.exists():
                    s = json.loads(sp.read_text(encoding="utf-8"))
                    table.setdefault(d.name, {})[t] = {k: s.get(k) for k in (
                        "n", "n_errors", "acc", "acc_ci95", "acc_norm", "acc_norm_ci95", "ppl",
                        "n_tokens", "n_windows", "primary_metric", "time_s", "adapter")}

    out = Path(a.out) if a.out else base / "general_report"
    save_json({"table": table, "meta": run_metadata(None, script="general_report.py")},
              out.with_suffix(".json"))

    L = ["# General benchmarks (forgetting axis)", "",
         "Log-likelihood based, zero-shot, bf16, no chat template. Accuracies in % (Wilson 95% CI).", "",
         "| Model | MMLU acc | HellaSwag acc_norm | ARC-C acc_norm | WikiText-2 PPL ↓ |",
         "|---|---|---|---|---:|"]
    for m, row in table.items():
        cells = []
        for t, key in (("mmlu", "acc"), ("hellaswag", "acc_norm"), ("arc_c", "acc_norm")):
            r = row.get(t)
            cells.append("—" if r is None else f"{pct(r[key])} {ci(r[key + '_ci95'])}")
        w = row.get("wikitext2")
        cells.append("—" if w is None or w["ppl"] is None else f"{w['ppl']:.3f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Details", "",
          "| Model | Task | n | errors | acc | 95% CI | acc_norm | 95% CI | PPL | time (s) |",
          "|---|---|---:|---:|---:|---|---:|---|---:|---:|"]
    for m, row in table.items():
        for t in TASKS:
            r = row.get(t)
            if r is None:
                continue
            if t == "wikitext2":
                L.append(f"| {m} | {LABELS[t]} | {r['n_tokens']} tokens / {r['n_windows']} windows | "
                         f"{r['n_errors']} | — | — | — | — | {r['ppl']:.3f} | {r['time_s']} |")
            else:
                L.append(f"| {m} | {LABELS[t]} | {r['n']} | {r['n_errors']} | {pct(r['acc'])} | {ci(r['acc_ci95'])} | "
                         f"{'—' if t == 'mmlu' else pct(r['acc_norm'])} | {'—' if t == 'mmlu' else ci(r['acc_norm_ci95'])} | "
                         f"— | {r['time_s']} |")
    L += ["", "_In MMLU the options are single letters, hence acc_norm = acc (not shown). Primary: MMLU acc, "
          "HellaSwag/ARC-C acc_norm, WikiText-2 PPL._", ""]
    out.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"-> {out.with_suffix('.md')} / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
