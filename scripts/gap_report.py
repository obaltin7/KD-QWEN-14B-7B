"""Teacher–student gap report: results/day1/gap_report.{md,json}.

- model x shots accuracy table: flexible and strict (Wilson 95% CI), format error side by side
- T − S0 gaps (points; flexible and strict) and an informational paired exact McNemar test
- RECOMMENDATION according to the decision rule, based on flexible scoring
If the summaries do not contain answer extraction v2, run scripts/rescore.py first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kdq.config import load_config, repo_path  # noqa: E402
from kdq.data import save_json  # noqa: E402
from kdq.evaluation import MODES, gap_status, paired_mcnemar, recommend  # noqa: E402
from kdq.utils import run_metadata  # noqa: E402

LABELS = {"teacher": "T (14B-Instruct)", "student_base": "S0-base (7B)",
          "student_instruct": "S0-Instruct (7B-Instruct)"}
STUDENTS = ("student_base", "student_instruct")
DECISION_MODE = "flexible"
KEYS = ("n", "accuracy_strict", "ci95_strict", "accuracy_flexible", "ci95_flexible",
        "format_error_rate", "no_answer_rate", "marker_truncated_rate", "truncated_rate",
        "mean_gen_tokens", "n_score_errors", "gen_s", "tokens_per_s")


def load_run(eval_dir: Path, model: str, shots: int, engine: str):
    prefix = eval_dir / f"{model}_{shots}shot_{engine}"
    sp = prefix.with_name(prefix.name + ".summary.json")
    if not sp.exists():
        return None, None
    summary = json.loads(sp.read_text(encoding="utf-8"))
    if "accuracy_strict" not in summary:
        sys.exit(f"[ERROR] {sp} was scored with answer extraction v1; run first: python scripts/rescore.py")
    with open(prefix.with_name(prefix.name + ".jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return summary, records


def pct(x):
    return f"{100 * x:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--engine", default="vllm")
    ap.add_argument("--out", default="results/day1/gap_report")
    a = ap.parse_args()

    cfg = load_config(a.config)
    dec = cfg["decision"]
    shots_list = cfg["eval"]["shots"]
    eval_dir = repo_path(cfg["eval"]["out_dir"])

    runs, table = {}, {}
    for m in ("teacher", *STUDENTS):
        for k in shots_list:
            s, r = load_run(eval_dir, m, k, a.engine)
            runs[(m, k)] = r
            if s:
                table[f"{m}@{k}"] = {key: s.get(key) for key in KEYS}

    gaps = {mode: {s: {} for s in STUDENTS} for mode in MODES}
    tests = {mode: {s: {} for s in STUDENTS} for mode in MODES}
    missing = []
    for s in STUDENTS:
        for k in shots_list:
            if f"teacher@{k}" not in table or f"{s}@{k}" not in table:
                missing.append(f"{s}@{k}")
                continue
            for mode in MODES:
                acc = f"accuracy_{mode}"
                gaps[mode][s][k] = 100 * (table[f"teacher@{k}"][acc] - table[f"{s}@{k}"][acc])
                tests[mode][s][k] = paired_mcnemar(runs[("teacher", k)], runs[(s, k)],
                                                   key=f"correct_{mode}")
    statuses = {
        s: gap_status(list(gaps[DECISION_MODE][s].values()), dec["min_gap_points"],
                      dec["comfortable_gap_points"])
        for s in STUDENTS if len(gaps[DECISION_MODE][s]) == len(shots_list)
    }
    rec = recommend(statuses, dec["preference"]) if not missing else None

    def strkeys(d):
        return {mode: {s: {str(k): v for k, v in g.items()} for s, g in dm.items()} for mode, dm in d.items()}

    report = {
        "engine": a.engine, "scoring_version": "v2", "decision_mode": DECISION_MODE,
        "table": table, "gaps_points": strkeys(gaps), "mcnemar": strkeys(tests),
        "decision_rule": dec, "statuses": statuses, "recommendation": rec,
        "missing": missing,
        "meta": run_metadata(None, script="gap_report.py"),
    }
    out = repo_path(a.out)
    save_json(report, out.with_suffix(".json"))

    L = ["# Baseline — teacher–student gap report (GSM8K test)", ""]
    L += [f"Engine: `{a.engine}` · greedy · max_new_tokens={cfg['eval']['max_new_tokens']} · "
          "plain Question/Answer prompt (no chat template) · answer extraction v2", "",
          "- **flexible:** the generation is cut at the first hallucinated-question marker / after the first "
          "`####` line; `#### number` → `answer is number` → last number.",
          "- **strict:** only the first `#### number`; otherwise unanswered (= format error).", ""]
    L += ["## Accuracy (%, Wilson 95% CI)", "",
          "| Model | shots | n | flexible | 95% CI | strict | 95% CI | format error | unanswered (flex) "
          "| cut at marker | truncated at 512 |",
          "|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|"]
    for m in ("teacher", *STUDENTS):
        for k in shots_list:
            r = table.get(f"{m}@{k}")
            if r is None:
                L.append(f"| {LABELS[m]} | {k} |" + " — |" * 9)
                continue
            fl, st = r["ci95_flexible"], r["ci95_strict"]
            L.append(f"| {LABELS[m]} | {k} | {r['n']} | {pct(r['accuracy_flexible'])} | [{pct(fl[0])}, {pct(fl[1])}] | "
                     f"{pct(r['accuracy_strict'])} | [{pct(st[0])}, {pct(st[1])}] | {pct(r['format_error_rate'])} | "
                     f"{pct(r['no_answer_rate'])} | {pct(r['marker_truncated_rate'])} | {pct(r['truncated_rate'])} |")
    L += ["", "## T − S0 gap (points) and paired McNemar test (informational)", "",
          "| Student | shots | gap flexible | p flexible | gap strict | p strict |",
          "|---|---:|---:|---:|---:|---:|"]
    for s in STUDENTS:
        for k in shots_list:
            if k in gaps["flexible"][s]:
                L.append(f"| {LABELS[s]} | {k} | {gaps['flexible'][s][k]:+.1f} | {tests['flexible'][s][k]['p_value']:.3g} | "
                         f"{gaps['strict'][s][k]:+.1f} | {tests['strict'][s][k]['p_value']:.3g} |")
    L += ["", "## Recommendation (flexible scoring)", "",
          f"Rule: smaller of the 0- and 4-shot T − S0 gaps ≥ {dec['comfortable_gap_points']} points → *sufficient*; "
          f"{dec['min_gap_points']}–{dec['comfortable_gap_points']} → *borderline*; "
          f"< {dec['min_gap_points']} → *insufficient*. Preference within the same category: "
          f"{', '.join(dec['preference'])}.", ""]
    for s, st in statuses.items():
        g = ", ".join(f"{k}-shot {v:+.1f}" for k, v in gaps["flexible"][s].items())
        gs = ", ".join(f"{k}-shot {v:+.1f}" for k, v in gaps["strict"][s].items())
        L.append(f"- {LABELS[s]}: **{st}** (flexible: {g}; strict: {gs})")
    if missing:
        L.append(f"- Missing runs: {', '.join(missing)} — no recommendation possible.")
    elif rec:
        L.append(f"\n**RECOMMENDATION:** student = `{rec}` ({LABELS[rec]}).")
    else:
        L.append("\n**RECOMMENDATION:** no candidate leaves a gap of at least "
                 f"{dec['min_gap_points']} points; no student is recommended.")
    L += ["", "_The recommendation follows solely from the pre-specified rule. "
          "The primary evaluation setting (0/4-shot) is fixed in the pre-registration._", ""]
    out.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"-> {out.with_suffix('.md')} / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
