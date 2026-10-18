import importlib.util
import math
from collections import Counter

import pytest

from kdq.config import REPO_ROOT
from kdq.general import (
    ItemError, arc_doc, choice_metrics, hellaswag_doc, hellaswag_preprocess, mmlu_doc, mmlu_prompt,
    perplexity, plan_windows, proportional_allocation, rows_sha256, run_batched, simple_sample,
    stratified_sample,
)

_spec = importlib.util.spec_from_file_location("eval_general", REPO_ROOT / "scripts/eval_general.py")
eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eg)


# --- stratified sampling ----------------------------------------------------

def test_proportional_allocation_exact_total():
    counts = {"a": 100, "b": 250, "c": 1, "d": 649}
    alloc = proportional_allocation(counts, 100)
    assert sum(alloc.values()) == 100
    assert alloc == {"a": 10, "b": 25, "c": 0, "d": 65}


def test_proportional_allocation_largest_remainder():
    alloc = proportional_allocation({"x": 1, "y": 1, "z": 1}, 2)
    assert sum(alloc.values()) == 2 and alloc == {"x": 1, "y": 1, "z": 0}  # ties broken by name


def test_stratified_sample_mmlu_like():
    # 57 subjects with unequal sizes (similar to the 14042-item MMLU test set)
    groups = []
    for s in range(57):
        groups += [f"subj{s:02d}"] * (100 + 7 * s)
    idx = stratified_sample(groups, 1000, 42)
    assert len(idx) == 1000 == len(set(idx)) and idx == sorted(idx)
    got = Counter(groups[i] for i in idx)
    total = len(groups)
    for g, c in Counter(groups).items():
        assert abs(got[g] - 1000 * c / total) < 1.0  # largest remainder: deviation < 1
    assert idx == stratified_sample(groups, 1000, 42)
    assert idx != stratified_sample(groups, 1000, 1234)


def test_simple_sample():
    idx = simple_sample(10042, 1000, 42)
    assert len(set(idx)) == 1000 and max(idx) < 10042 and idx == simple_sample(10042, 1000, 42)
    with pytest.raises(ValueError):
        simple_sample(10, 11, 0)


def test_rows_sha256_order_sensitive():
    a, b = {"q": 1}, {"q": 2}
    assert rows_sha256([a, b]) != rows_sha256([b, a]) and rows_sha256([a]) == rows_sha256([{"q": 1}])


# --- formats ----------------------------------------------------------------

def test_mmlu_prompt_exact():
    p = mmlu_prompt(" What is 2+2? ", ["3", "4", "5", "6"])
    assert p == "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"
    ctx, choices, gold = mmlu_doc({"question": "Q", "choices": ["a", "b", "c", "d"], "answer": 2, "subject": "s"})
    assert choices == ["A", "B", "C", "D"] and gold == 2 and "<|im_start|>" not in ctx


def test_hellaswag_doc():
    row = {"activity_label": "Removing ice from car", "ctx_a": "Then, the man writes over the snow.",
           "ctx_b": "then", "endings": [" [title] Use a brush.", "b", "c", "d"], "label": "3"}
    ctx, choices, gold = hellaswag_doc(row)
    assert ctx == "Removing ice from car: Then, the man writes over the snow. Then"
    # Same as lm-eval: strip first, then " [title]" -> ". " (the leading tag is removed by the bracket rule)
    assert choices[0] == " Use a brush." and gold == 3
    assert hellaswag_preprocess("Go home [title] Rest.") == "Go home. Rest."
    assert hellaswag_preprocess(" a [header] b  c ") == "a b c"


def test_arc_doc_letter_and_digit_keys():
    row = {"question": "Which?", "choices": {"text": ["x", "y", "z", "w"], "label": ["A", "B", "C", "D"]},
           "answerKey": "C"}
    ctx, choices, gold = arc_doc(row)
    assert ctx == "Question: Which?\nAnswer:" and choices == ["x", "y", "z", "w"] and gold == 2
    row2 = {"question": "Q", "choices": {"text": ["p", "q", "r"], "label": ["1", "2", "3"]}, "answerKey": "2"}
    assert arc_doc(row2)[2] == 1


# --- scoring (fake log-likelihoods) -----------------------------------------

def test_choice_metrics_acc_vs_norm():
    # choice 0 is best by total; choice 1 is best after length normalisation
    m = choice_metrics([-4.0, -6.0, -9.0], [4, 12, 9], gold=1)
    assert m["pred"] == 0 and m["pred_norm"] == 1
    assert not m["correct"] and m["correct_norm"]


def test_choice_metrics_tie_first():
    assert choice_metrics([-1.0, -1.0], [1, 1], gold=0)["pred"] == 0


def test_score_choice_and_summary():
    raw = [
        {"idx": 0, "gold": 1, "logprobs": [-3.0, -1.0], "cont_chars": [1, 1], "error": None},
        {"idx": 1, "gold": 0, "logprobs": [-2.0, -1.0], "cont_chars": [10, 1], "error": None},
        {"idx": 2, "gold": 0, "logprobs": [None, -1.0], "cont_chars": [1, 1], "error": "OOM"},
    ]
    recs = eg.score_choice(raw)
    assert [r["correct"] for r in recs] == [True, False, False]
    assert [r["correct_norm"] for r in recs] == [True, True, False]
    s = eg.choice_summary("arc_c", recs)
    assert s["n"] == 3 and s["n_errors"] == 1 and s["acc"] == 1 / 3 and s["acc_norm"] == 2 / 3
    assert s["primary_metric"] == "acc_norm" and s["acc_ci95"][0] < 1 / 3 < s["acc_ci95"][1]


def test_mmlu_summary_by_subject():
    recs = [{"subject": "a", "correct": True, "correct_norm": True},
            {"subject": "a", "correct": False, "correct_norm": False},
            {"subject": "b", "correct": True, "correct_norm": True}]
    s = eg.choice_summary("mmlu", recs)
    assert s["primary_metric"] == "acc" and s["by_subject"] == {"a": {"n": 2, "acc": 0.5}, "b": {"n": 1, "acc": 1.0}}


class FakeLM:
    """Character-level tokenizer: each character is one token."""
    max_length = 8

    def encode(self, text):
        return [ord(c) for c in text]


def test_choice_request_split_and_truncation():
    inp, tgt = eg.choice_request(FakeLM(), "ab", " c")
    assert tgt == [ord(" "), ord("c")] and inp == [ord("a"), ord("b"), ord(" ")]
    inp, tgt = eg.choice_request(FakeLM(), "0123456789", " x")
    assert len(inp) == FakeLM.max_length and inp[-1] == ord(" ") and tgt == [ord(" "), ord("x")]


# --- sliding window and perplexity ------------------------------------------

@pytest.mark.parametrize("n, L, s", [(10000, 2048, 512), (2048, 2048, 512), (2049, 2048, 512),
                                     (100, 2048, 512), (7, 4, 2), (1, 4, 2)])
def test_plan_windows_each_token_once(n, L, s):
    wins = plan_windows(n, L, s)
    scored = Counter()
    for b, e, f in wins:
        assert 0 <= b < f <= e <= n and e - b <= L
        scored.update(range(f, e))
    assert set(scored) == set(range(1, n)) and all(c == 1 for c in scored.values())


def test_plan_windows_context_after_first():
    wins = plan_windows(10000, 2048, 512)
    assert wins[0] == (0, 2048, 1) and wins[1] == (512, 2560, 2048)
    for b, e, f in wins[1:]:
        assert f - b >= 2048 - 512  # every scored token has at least 1536 tokens of context


def test_perplexity_aggregation():
    w = [{"nll_sum": 2.0, "n_tokens": 2}, {"nll_sum": 4.0, "n_tokens": 2},
         {"nll_sum": None, "n_tokens": 5, "error": "x"}]
    p = perplexity(w)
    assert p["n_tokens"] == 4 and p["n_errors"] == 1 and p["ppl"] == pytest.approx(math.exp(1.5))
    assert perplexity([])["ppl"] is None


# --- batched execution: OOM and single-item errors ---------------------------

class OutOfMemoryError(RuntimeError):
    pass


def test_run_batched_halves_on_oom():
    calls, cleared = [], []

    def fn(batch):
        calls.append(len(batch))
        if len(batch) > 3:
            raise OutOfMemoryError("CUDA out of memory")
        return [x * 2 for x in batch]

    out, bs = run_batched(list(range(10)), fn, 16, on_oom=lambda: cleared.append(1), log=lambda m: None)
    assert out == [x * 2 for x in range(10)] and bs == 2 and calls[:3] == [10, 8, 4] and len(cleared) == 3


def test_run_batched_isolates_item_errors():
    def fn(batch):
        if 5 in batch:
            raise ValueError("corrupt example")
        return [x + 1 for x in batch]

    out, _ = run_batched(list(range(8)), fn, 4, log=lambda m: None)
    assert isinstance(out[5], ItemError) and "corrupt" in out[5].msg
    assert [o for i, o in enumerate(out) if i != 5] == [x + 1 for x in range(8) if x != 5]


def test_run_batched_oom_at_bs1_is_item_error():
    def fn(batch):
        if batch == [3]:
            raise OutOfMemoryError("out of memory")
        if len(batch) > 1:
            raise OutOfMemoryError("out of memory")
        return batch

    out, bs = run_batched(list(range(5)), fn, 4, log=lambda m: None)
    assert bs == 1 and isinstance(out[3], ItemError) and out[:3] == [0, 1, 2] and out[4] == 4
