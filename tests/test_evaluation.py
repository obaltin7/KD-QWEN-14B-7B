import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from kdq.config import DEFAULT_CONFIG, REPO_ROOT
import kdq.evaluation as ev
from kdq.evaluation import (
    crosscheck_report, gap_status, paired_mcnemar, recommend, score, score_safe, summarize,
    write_jsonl, write_scored,
)


def test_score():
    r = score(" 2+3=5\n#### 5", "5")
    assert r["pred_strict"] == r["pred_flexible"] == r["pred"] == "5"
    assert r["correct_strict"] and r["correct_flexible"] and r["correct"]
    assert not r["format_error"] and not r["no_answer"] and r["truncated_at"] is None
    s = score(" 9*2 = 18. The answer is 18.\nQ: other 7", "18")
    assert s["pred_strict"] is None and not s["correct_strict"] and s["format_error"]
    assert s["pred_flexible"] == "18" and s["correct_flexible"] and s["correct"]
    assert s["truncated_at"] is not None
    assert score(" no idea", "5")["no_answer"]


def _rec(cs, cf, fe=False, na=False, tok=5, fin="stop", cut=None):
    return {"correct_strict": cs, "correct_flexible": cf, "correct": cf, "format_error": fe,
            "no_answer": na, "n_gen_tokens": tok, "finish_reason": fin, "truncated_at": cut}


def test_summarize():
    recs = [_rec(True, True, tok=10), _rec(False, True, fe=True, cut=40),
            _rec(False, False, fe=True, na=True, tok=512, fin="length")]
    s = summarize(recs)
    assert s["n"] == 3 and s["scoring_version"] == "v2"
    assert s["accuracy_strict"] == 1 / 3 and s["accuracy_flexible"] == 2 / 3
    assert s["accuracy"] == s["accuracy_flexible"] and s["ci95"] == s["ci95_flexible"]
    assert s["format_error_rate"] == 2 / 3 and s["no_answer_rate"] == 1 / 3
    assert s["marker_truncated_rate"] == 1 / 3 and s["truncated_rate"] == 1 / 3
    assert s["ci95_strict"][0] < 1 / 3 < s["ci95_strict"][1]


def test_crosscheck_report():
    v = {0: {"generation": "a", "gold": "1", "pred_strict": "1", "correct_strict": True,
             "pred_flexible": "1", "correct_flexible": True},
         1: {"generation": "b", "gold": "2", "pred_strict": None, "correct_strict": False,
             "pred_flexible": "2", "correct_flexible": True}}
    h = {0: dict(v[0]), 1: {**v[1], "generation": "c", "pred_flexible": "3", "correct_flexible": False}}
    r = crosscheck_report(v, h)
    assert r["n"] == 2 and r["text_exact_match"] == 0.5
    assert r["flexible"]["accuracy_diff_points"] == 50.0 and r["strict"]["accuracy_diff_points"] == 0.0
    assert r["flexible"]["pred_agreement"] == 0.5 and r["strict"]["pred_agreement"] == 1.0
    assert [d["idx"] for d in r["disagreements_flexible"]] == [1]


def test_paired_mcnemar():
    a = [{"idx": i, "correct": c} for i, c in enumerate([1, 1, 1, 0])]
    b = [{"idx": i, "correct": c} for i, c in enumerate([1, 0, 0, 1])]
    r = paired_mcnemar(a, b)
    assert (r["n_paired"], r["only_a"], r["only_b"]) == (4, 2, 1)


def test_gap_status_and_recommend():
    assert gap_status([10, 9], 5, 8) == "sufficient"
    assert gap_status([10, 6], 5, 8) == "borderline"
    assert gap_status([10, 3], 5, 8) == "insufficient"
    pref = ["student_base", "student_instruct"]
    assert recommend({"student_base": "sufficient", "student_instruct": "insufficient"}, pref) == "student_base"
    assert recommend({"student_base": "borderline", "student_instruct": "sufficient"}, pref) == "student_instruct"
    assert recommend({"student_base": "sufficient", "student_instruct": "sufficient"}, pref) == "student_base"
    assert recommend({"student_base": "insufficient", "student_instruct": "insufficient"}, pref) is None


def _fake_run(d: Path, model: str, shots: int, correct: list[int]):
    prefix = d / f"{model}_{shots}shot_vllm"
    # strict: half of the correct answers behave as if "####" was not used (wrong under strict)
    recs = [{"idx": i, **_rec(bool(c) and i % 2 == 0, bool(c))} for i, c in enumerate(correct)]
    prefix.with_name(prefix.name + ".jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    s = summarize(recs) | {"gen_s": 1.0, "tokens_per_s": 5.0}
    prefix.with_name(prefix.name + ".summary.json").write_text(json.dumps(s), encoding="utf-8")


def test_gap_report_end_to_end(tmp_path):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["eval"]["out_dir"] = str(tmp_path / "eval")
    (tmp_path / "eval").mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    n = 100
    acc = {"teacher": 0.80, "student_base": 0.60, "student_instruct": 0.76}
    for m, p in acc.items():
        for k in (0, 4):
            _fake_run(tmp_path / "eval", m, k, [1] * int(p * n) + [0] * (n - int(p * n)))
    out = tmp_path / "gap_report"
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts/gap_report.py"),
                    "--config", str(cfg_path), "--out", str(out)],
                   check=True, capture_output=True,
                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    rep = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert rep["statuses"] == {"student_base": "sufficient", "student_instruct": "insufficient"}
    assert rep["recommendation"] == "student_base"
    assert round(rep["gaps_points"]["flexible"]["student_base"]["0"], 6) == 20.0
    assert "strict" in rep["gaps_points"] and rep["decision_mode"] == "flexible"
    assert "RECOMMENDATION" in out.with_suffix(".md").read_text(encoding="utf-8")


def test_score_safe_catches_errors(monkeypatch):
    def boom(_):
        raise ArithmeticError("corrupt")
    monkeypatch.setattr(ev, "extract_flexible", boom)
    r = score_safe(" #### 5", "5")
    assert r["score_error"].startswith("ArithmeticError") and r["correct"] is False
    assert r["pred_flexible"] is None and r["no_answer"] and r["format_error"]


def test_score_safe_long_number_no_error():
    r = score_safe(" #### " + "9" * 500, "18")
    assert r["score_error"] is None and not r["correct"] and r["no_answer"]


def test_summarize_counts_score_errors():
    recs = [{**_rec(True, True), "score_error": None}, {**_rec(False, False), "score_error": "X"}]
    assert summarize(recs)["n_score_errors"] == 1


def _raw(i, gen, gold):
    return {"idx": i, "gold": gold, "generation": gen, "n_gen_tokens": 7, "finish_reason": "stop"}


def test_write_scored(tmp_path):
    prefix = tmp_path / "m_4shot_vllm"
    s = write_scored(prefix, [_raw(0, " 2+3=5\n#### 5", "5"), _raw(1, " #### " + "9" * 99, "1")],
                     {"model_key": "m", "gen_s": 2.0})
    assert s["model_key"] == "m" and s["n"] == 2 and s["accuracy_strict"] == 0.5
    assert s["n_score_errors"] == 0 and s["scoring_version"] == "v2"
    lines = (tmp_path / "m_4shot_vllm.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1])["n_gen_tokens"] == 7  # raw fields are preserved
    assert (tmp_path / "m_4shot_vllm.summary.json").exists()


def test_rescore_from_raw_only(tmp_path):
    d = tmp_path / "eval"
    prefix = d / "student_base_4shot_vllm"
    write_jsonl([_raw(0, " 9*2 = 18. The answer is 18.\nQ: 7", "18"),
                 _raw(1, " #### " + "9" * 300, "4")], prefix.with_name(prefix.name + ".raw.jsonl"))
    prefix.with_name(prefix.name + ".raw.meta.json").write_text(
        json.dumps({"model_key": "student_base", "shots": 4, "gen_s": 60.0}), encoding="utf-8")
    # a partial run with neither a scored file nor raw metadata must be skipped
    write_jsonl([_raw(0, "x", "1")], d / "half.raw.jsonl")
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts/rescore.py"), "--dir", str(tmp_path)],
                   check=True, capture_output=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    s = json.loads(prefix.with_name(prefix.name + ".summary.json").read_text(encoding="utf-8"))
    assert s["model_key"] == "student_base" and s["gen_s"] == 60.0
    assert s["accuracy_flexible"] == 0.5 and s["accuracy_strict"] == 0.0 and s["n_score_errors"] == 0
    assert not (d / "half.summary.json").exists()
