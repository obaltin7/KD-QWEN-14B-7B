"""Per-question scoring, summarisation and the student-selection decision rule."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .answer import extract_flexible, extract_strict, is_correct
from .stats import mcnemar_exact, wilson_ci

STATUS_RANK = {"insufficient": 0, "borderline": 1, "sufficient": 2}
SCORING_VERSION = "v2"
MODES = ("strict", "flexible")


def score(text: str, gold: str) -> dict:
    """Answer extraction v2: strict ("#### number") and flexible (truncate + cascaded extraction).

    format_error: no valid "#### number" (unanswered under strict).
    no_answer: no number could be extracted in flexible mode either.
    pred/correct: equal to flexible, for backward compatibility.
    """
    p_strict = extract_strict(text)
    p_flex, cut_at = extract_flexible(text)
    c_flex = is_correct(p_flex, gold)
    return {
        "pred_strict": p_strict,
        "correct_strict": is_correct(p_strict, gold),
        "pred_flexible": p_flex,
        "correct_flexible": c_flex,
        "truncated_at": cut_at,
        "pred": p_flex,
        "correct": c_flex,
        "format_error": p_strict is None,
        "no_answer": p_flex is None,
    }


def score_safe(text: str, gold: str) -> dict:
    """score() with per-question error handling: on error the question counts as wrong/unanswered."""
    try:
        return {**score(text, gold), "score_error": None}
    except Exception as e:  # an error on a single question must not abort the run
        return {
            "pred_strict": None, "correct_strict": False,
            "pred_flexible": None, "correct_flexible": False,
            "truncated_at": None, "pred": None, "correct": False,
            "format_error": True, "no_answer": True,
            "score_error": f"{type(e).__name__}: {e}"[:300],
        }


def score_records(raw: Iterable[Mapping]) -> list[dict]:
    """Scores raw records ({idx, gold, generation, ...}) with v2; other fields are preserved."""
    return [{**r, **score_safe(r["generation"], r["gold"])} for r in raw]


def write_jsonl(records: Iterable[Mapping], path: Path) -> None:
    """Writes to a temporary file, then moves it into place (no partial files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_scored(prefix: Path, raw: Iterable[Mapping], base_summary: Mapping) -> dict:
    """Scores; writes <prefix>.jsonl and (last) <prefix>.summary.json; returns the summary."""
    records = score_records(raw)
    write_jsonl(records, prefix.with_name(prefix.name + ".jsonl"))
    summary = {**base_summary, **summarize(records),
               "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = prefix.with_name(prefix.name + ".summary.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    return summary


def summarize(records: Sequence[Mapping]) -> dict:
    n = len(records)

    def rate(key, val=True):
        return sum(r.get(key) == val for r in records) / n if n else 0.0

    out: dict = {"n": n, "scoring_version": SCORING_VERSION}
    for mode in MODES:
        k = sum(bool(r[f"correct_{mode}"]) for r in records)
        out[f"n_correct_{mode}"] = k
        out[f"accuracy_{mode}"] = k / n if n else 0.0
        out[f"ci95_{mode}"] = list(wilson_ci(k, n))
    # Backward compatibility: accuracy/ci95 = flexible
    out["n_correct"] = out["n_correct_flexible"]
    out["accuracy"] = out["accuracy_flexible"]
    out["ci95"] = out["ci95_flexible"]
    toks = [r["n_gen_tokens"] for r in records if r.get("n_gen_tokens") is not None]
    out.update({
        "format_error_rate": rate("format_error"),
        "no_answer_rate": rate("no_answer"),
        "marker_truncated_rate": sum(r.get("truncated_at") is not None for r in records) / n if n else 0.0,
        "n_score_errors": sum(bool(r.get("score_error")) for r in records),
        "truncated_rate": rate("finish_reason", "length"),
        "mean_gen_tokens": sum(toks) / len(toks) if toks else None,
    })
    return out


def crosscheck_report(v: Mapping[int, Mapping], h: Mapping[int, Mapping]) -> dict:
    """Per-question agreement between vllm (v) and hf (h); idx -> record dicts."""
    idx = sorted(v.keys() & h.keys())
    n = len(idx)
    rep: dict = {"n": n, "scoring_version": SCORING_VERSION}
    for mode in MODES:
        acc_v = sum(bool(v[i][f"correct_{mode}"]) for i in idx) / n
        acc_h = sum(bool(h[i][f"correct_{mode}"]) for i in idx) / n
        rep[mode] = {
            "accuracy_vllm": acc_v,
            "accuracy_hf": acc_h,
            "accuracy_diff_points": 100 * (acc_v - acc_h),  # vllm − hf
            "correct_agreement": sum(v[i][f"correct_{mode}"] == h[i][f"correct_{mode}"] for i in idx) / n,
            "pred_agreement": sum(v[i][f"pred_{mode}"] == h[i][f"pred_{mode}"] for i in idx) / n,
        }
    rep["text_exact_match"] = sum(v[i]["generation"].strip() == h[i]["generation"].strip() for i in idx) / n
    rep["disagreements_flexible"] = [
        {"idx": i, "gold": v[i]["gold"], "pred_vllm": v[i]["pred_flexible"], "pred_hf": h[i]["pred_flexible"],
         "correct_vllm": v[i]["correct_flexible"], "correct_hf": h[i]["correct_flexible"]}
        for i in idx if v[i]["pred_flexible"] != h[i]["pred_flexible"]
    ]
    return rep


def paired_mcnemar(a: Iterable[Mapping], b: Iterable[Mapping], key: str = "correct") -> dict:
    """McNemar test paired on idx (a = teacher, b = student)."""
    ca = {r["idx"]: bool(r[key]) for r in a}
    cb = {r["idx"]: bool(r[key]) for r in b}
    common = sorted(ca.keys() & cb.keys())
    only_a = sum(ca[i] and not cb[i] for i in common)
    only_b = sum(cb[i] and not ca[i] for i in common)
    return {"n_paired": len(common), "only_a": only_a, "only_b": only_b,
            "p_value": mcnemar_exact(only_a, only_b)}


def gap_status(gaps_points: Sequence[float], min_gap: float, comfortable: float) -> str:
    """Status based on the smallest gap across all shot settings."""
    g = min(gaps_points)
    if g >= comfortable:
        return "sufficient"
    if g >= min_gap:
        return "borderline"
    return "insufficient"


def recommend(statuses: Mapping[str, str], preference: Sequence[str]) -> str | None:
    """Candidate with the best status; ties broken by preference order. None if none qualifies."""
    best = max((STATUS_RANK[s] for s in statuses.values()), default=0)
    if best == 0:
        return None
    for cand in preference:
        if cand in statuses and STATUS_RANK[statuses[cand]] == best:
            return cand
    return None
