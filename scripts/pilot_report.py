"""Hyperparameter pilot report -> results/train/pilot/{smoke_report.json, pilot_report.md, pilot_report.json}.

- Smoke tests (smoke_S1/S2/S3_seed42): peak memory, tokens/s, steps/s, CE/KL/feature losses of the
  first 5 steps, β·L_feat / CE ratio, estimated duration of a full run (2 epochs).
- Full S1 pilot runs (one per lr): val 0-shot flexible accuracy (Wilson 95% CI), strict alongside.
- Pre-registered stability rule: any NaN/divergence; does β·L_feat initially exceed 10x CE.
- Total duration estimate for the main runs (6) and their evaluations.
Writes a RECOMMENDATION only (pre-registration §4).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kdq.config import load_config, repo_path  # noqa: E402
from kdq.data import save_json  # noqa: E402
from kdq.distill import steps_per_epoch  # noqa: E402
from kdq.utils import run_metadata  # noqa: E402

CFGS = ("S1", "S2", "S3")
RATIO_LIMIT = 10.0      # prereg §4: feature loss initially exceeding CE by more than an order of magnitude


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_log(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def smoke_entry(d: Path, total_steps: int) -> dict | None:
    s, recs = read_json(d / "summary.json"), read_log(d / "train_log.jsonl")
    if s is None or not recs:
        return None
    first = recs[:5]
    steady = recs[2:] if len(recs) > 2 else recs
    step_t = sum(r["step_time_s"] for r in steady) / len(steady)
    ratio = [(r["feat"] / r["ce"]) if r.get("feat") is not None and r["ce"] else None for r in first]
    return {
        "steps": len(recs), "peak_mem_gb": s.get("peak_mem_gb"),
        "tokens_per_s": sum(r["tokens_per_s"] for r in steady) / len(steady),
        "steps_per_s": 1.0 / step_t, "mean_step_time_s": step_t,
        "first5": [{k: r.get(k) for k in ("step", "ce", "kl", "feat", "loss", "grad_norm")} for r in first],
        "feat_over_ce_first5": ratio,
        "any_nonfinite": s.get("any_nonfinite"),
        "est_full_run_s": step_t * total_steps,
        "n_trainable_lora": s.get("n_trainable_lora"), "n_projector_params": s.get("n_projector_params"),
    }


def divergence(recs: list[dict]) -> dict:
    """Flags 'divergence' if the mean loss of the last 10% exceeds that of the first 10% (heuristic only)."""
    if len(recs) < 10:
        return {"checked": False}
    k = max(1, len(recs) // 10)
    head = sum(r["loss"] for r in recs[:k]) / k
    tail = sum(r["loss"] for r in recs[-k:]) / k
    return {"checked": True, "first10pct_loss": head, "last10pct_loss": tail, "diverged": tail > head,
            "any_nonfinite": any(not math.isfinite(r["loss"]) for r in recs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--pilot-dir", default="results/train/pilot")
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    tcfg = cfg["train"]
    pdir = repo_path(a.pilot_dir)
    n_train = (read_json(repo_path(cfg["paths"]["splits_dir"]) / "train_rest.json") or {}).get("n", 7173)
    total_steps = int(tcfg["epochs"]) * steps_per_epoch(n_train, int(tcfg["effective_batch"]))

    # --- smoke tests
    smoke = {c: smoke_entry(pdir / f"smoke_{c}_seed42", total_steps) for c in CFGS}
    beta = float(tcfg["hidden"]["beta"])
    s3 = smoke.get("S3")
    ratio_flag = None
    if s3 and any(r is not None for r in s3["feat_over_ce_first5"]):
        ratios = [beta * r for r in s3["feat_over_ce_first5"] if r is not None]
        ratio_flag = {"beta": beta, "max_beta_feat_over_ce_first5": max(ratios), "exceeds_10x": max(ratios) > RATIO_LIMIT}
    smoke_report = {"total_steps_full_run": total_steps, "n_train": n_train, "smoke": smoke,
                    "stability_ratio_rule": ratio_flag,
                    "meta": run_metadata(None, script="pilot_report.py", mode="smoke")}
    save_json(smoke_report, pdir / "smoke_report.json")
    for c, e in smoke.items():
        if e:
            print(f"[smoke] {c}: mem={e['peak_mem_gb']} GB | {e['tokens_per_s']:.0f} tok/s | "
                  f"{e['steps_per_s']:.3f} steps/s | full run ~{e['est_full_run_s'] / 60:.0f} min | NaN={e['any_nonfinite']}")
        else:
            print(f"[smoke] {c}: missing")
    if ratio_flag:
        print(f"[smoke] β·L_feat/CE (first 5 steps, max) = {ratio_flag['max_beta_feat_over_ce_first5']:.2f} "
              f"-> {'EXCEEDS 10x' if ratio_flag['exceeds_10x'] else 'within rule'}")
    if a.smoke_only:
        return 0

    # --- full S1 pilot runs and val
    runs = {}
    for d in sorted(pdir.glob("S1_seed*_lr*")):
        if not d.is_dir():
            continue
        s = read_json(d / "summary.json")
        val = read_json(pdir / f"{d.name}_val0shot.summary.json")
        runs[d.name] = {"train": {k: s.get(k) for k in ("lr", "steps_done", "total_steps", "completed",
                                                        "wall_time_s", "peak_mem_gb", "final", "any_nonfinite")} if s else None,
                        "divergence": divergence(read_log(d / "train_log.jsonl")),
                        "val": {k: val.get(k) for k in ("n", "accuracy_flexible", "ci95_flexible", "accuracy_strict",
                                                        "ci95_strict", "format_error_rate", "truncated_rate")} if val else None}
    # Recommendation: lr with the highest val flexible accuracy; ties go to the smaller lr (conservative).
    cands = [(r["val"]["accuracy_flexible"], -r["train"]["lr"], name) for name, r in runs.items()
             if r["val"] and r["train"] and r["train"]["completed"] and not r["divergence"].get("diverged")]
    rec = max(cands)[2] if cands else None
    stability = {
        "nonfinite_any": any((r["train"] or {}).get("any_nonfinite") for r in runs.values())
        or any((e or {}).get("any_nonfinite") for e in smoke.values()),
        "diverged_any": any(r["divergence"].get("diverged") for r in runs.values()),
        "feat_ratio_rule": ratio_flag,
    }

    # --- duration estimate (main runs)
    day1 = repo_path(cfg["eval"]["out_dir"])
    gsm_s = 0.0
    for k in cfg["eval"]["shots"]:
        s = read_json(day1 / f"student_instruct_{k}shot_vllm.summary.json")
        gsm_s += (s["gen_s"] + s["load_s"]) if s else 600.0
    gen_dir = repo_path(cfg["general"]["out_dir"]) / "student_instruct"
    gen_s = sum((read_json(p) or {}).get("time_s", 0) for p in gen_dir.glob("*.summary.json")) or 900.0
    gen_s += 120.0  # model loading
    train_s = {c: (smoke[c]["est_full_run_s"] if smoke.get(c) else None) for c in CFGS}
    est = {"per_run_eval_s": gsm_s + gen_s, "train_s_per_cfg": train_s,
           "total_main_s": sum((t or 0) * 2 for t in train_s.values()) + 6 * (gsm_s + gen_s),
           "note": "2 seeds x 3 configurations training + GSM8K test (0/4-shot) and general benchmarks per run"}

    report = {"runs": runs, "smoke": smoke, "stability": stability, "recommended_lr_run": rec,
              "estimate": est, "meta": run_metadata(None, script="pilot_report.py")}
    save_json(report, pdir / "pilot_report.json")

    L = ["# Hyperparameter pilot (S1, val 300, 0-shot flexible)", "",
         f"Full run: {total_steps} steps (n={n_train}, effective batch {tcfg['effective_batch']}, "
         f"{tcfg['epochs']} epochs).", "",
         "## Val accuracy (per lr)", "",
         "| Run | lr | flexible % | 95% CI | strict % | format error % | truncated at 512 % | final loss | NaN | diverged |",
         "|---|---:|---:|---|---:|---:|---:|---:|---|---|"]
    for name, r in runs.items():
        v, t, dv = r["val"], r["train"], r["divergence"]
        if v and t:
            lo, hi = v["ci95_flexible"]
            L.append(f"| {name} | {t['lr']:g} | {100 * v['accuracy_flexible']:.1f} | [{100 * lo:.1f}, {100 * hi:.1f}] | "
                     f"{100 * v['accuracy_strict']:.1f} | {100 * v['format_error_rate']:.1f} | {100 * v['truncated_rate']:.1f} | "
                     f"{t['final']['loss']:.4f} | {t['any_nonfinite']} | {dv.get('diverged')} |")
        else:
            L.append(f"| {name} | — | missing (train: {'yes' if t else 'no'}, val: {'yes' if v else 'no'}) | | | | | | | |")
    L += ["", "## Smoke tests (20 steps, lr 1e-4)", "",
          "| Cfg | memory GB | tok/s | steps/s | first 5 CE | first 5 KL | first 5 feat | β·feat/CE | full-run estimate |",
          "|---|---:|---:|---:|---|---|---|---|---:|"]
    for c, e in smoke.items():
        if not e:
            L.append(f"| {c} | — | — | — | — | — | — | — | — |")
            continue
        f5 = e["first5"]
        fmt = lambda key: ", ".join("—" if r[key] is None else f"{r[key]:.3f}" for r in f5)  # noqa: E731
        rat = ", ".join("—" if x is None else f"{beta * x:.2f}" for x in e["feat_over_ce_first5"])
        mem = "—" if e["peak_mem_gb"] is None else f"{e['peak_mem_gb']:.1f}"
        L.append(f"| {c} | {mem} | {e['tokens_per_s']:.0f} | {e['steps_per_s']:.3f} | {fmt('ce')} | {fmt('kl')} | "
                 f"{fmt('feat')} | {rat} | {e['est_full_run_s'] / 60:.0f} min |")
    L += ["", "## Stability rule (prereg §4)", "",
          f"- NaN/inf: **{'YES' if stability['nonfinite_any'] else 'none'}**",
          f"- Divergence (last 10% > first 10% loss): **{'YES' if stability['diverged_any'] else 'none'}**",
          ("- β·L_feat / CE (S3, first 5 steps, max): "
           f"**{ratio_flag['max_beta_feat_over_ce_first5']:.2f}** -> "
           f"{'EXCEEDS 10x: β must be re-evaluated and the rationale recorded' if ratio_flag['exceeds_10x'] else 'below 10x; τ, α, β defaults (2, 0.5, 1) unchanged'}"
           if ratio_flag else "- β·L_feat / CE: no S3 smoke test"),
          "", "## Duration estimate (main runs)", "",
          "- Training (estimated from smoke tests): " + ", ".join(
              f"{c} ~{(t or 0) / 60:.0f} min" for c, t in train_s.items()),
          f"- Evaluation per run: ~{est['per_run_eval_s'] / 60:.0f} min (GSM8K test 0/4-shot + general benchmarks)",
          f"- **Total ~{est['total_main_s'] / 3600:.1f} h** (6 runs)", "", "## Recommendation", ""]
    if rec:
        L.append(f"**RECOMMENDATION:** lr = `{runs[rec]['train']['lr']:g}` ({rec}); highest val flexible accuracy "
                 "(ties go to the smaller lr). Number of epochs: 2 (prereg). The frozen value is recorded in "
                 "`configs/base.yaml` -> `train.lr` and in `docs/decisions.md`.")
    else:
        L.append("**RECOMMENDATION:** no completed, non-diverged S1 pilot run; no recommendation possible.")
    L.append("")
    (pdir / "pilot_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
