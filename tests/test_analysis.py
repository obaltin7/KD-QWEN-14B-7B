import math

import numpy as np
import pytest

from kdq.analysis import (
    INCONSISTENT, NOT_SUPPORTED, OPPOSITE, PARTIAL, SUPPORTED, decide, gap_closed, holm, mcnemar_paired,
    paired_bootstrap_acc, paired_bootstrap_ppl, ppl,
)


# --- Holm --------------------------------------------------------------------

def test_holm_known_values():
    # sorted: 0.005·4 = 0.02; 0.01·3 = 0.03; 0.03·2 = 0.06; 0.04·1 = 0.04 -> monotonicity: 0.06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_single():
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert holm([0.03]) == pytest.approx([0.03])
    assert holm([]) == []


def test_holm_never_below_raw():
    rng = np.random.default_rng(0)
    p = list(rng.uniform(0, 0.2, size=12))
    assert all(a >= r for a, r in zip(holm(p), p))


# --- McNemar (paired) ----------------------------------------------------------

def test_mcnemar_paired_counts():
    a = [1, 1, 1, 0, 0, 1]
    b = [1, 0, 0, 1, 0, 1]
    r = mcnemar_paired(a, b)
    assert (r["b_only_a"], r["c_only_b"], r["n"]) == (2, 1, 6)
    assert r["diff_points"] == pytest.approx(100 * (4 / 6 - 3 / 6))
    assert r["p"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mcnemar_paired([1, 0], [1])


# --- bootstrap ----------------------------------------------------------------

def test_bootstrap_acc_identical_is_zero():
    a = np.random.default_rng(1).integers(0, 2, 500).astype(bool)
    assert paired_bootstrap_acc(a, a, n_boot=2000) == (0.0, 0.0)


def test_bootstrap_acc_constant_difference():
    # a always correct, b always wrong -> the difference is exactly 100 points in every resample
    assert paired_bootstrap_acc([1] * 50, [0] * 50, n_boot=500) == (100.0, 100.0)


def test_bootstrap_acc_known_input_contains_point_and_deterministic():
    rng = np.random.default_rng(3)
    b = rng.random(1319) < 0.77
    a = b.copy()
    flip = rng.choice(1319, 80, replace=False)
    a[flip] = True                               # a ≥ b; difference > 0
    diff = 100 * (a.mean() - b.mean())
    lo, hi = paired_bootstrap_acc(a, b)
    assert lo <= diff <= hi and lo > 0
    assert (lo, hi) == paired_bootstrap_acc(a, b)               # same seed -> same CI
    # normal approximation: width ~ 2·1.96·SE of the paired difference
    d = a.astype(float) - b.astype(float)
    se = 100 * d.std(ddof=1) / math.sqrt(d.size)
    assert (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.1)


def test_ppl_and_bootstrap_ppl():
    nll = np.array([2.0, 4.0, 6.0])
    tok = np.array([2.0, 2.0, 2.0])
    assert ppl(nll, tok) == pytest.approx(math.exp(2.0))
    r = paired_bootstrap_ppl(nll, nll, tok, n_boot=300)
    assert r["diff"] == 0 and r["ci95"] == [0.0, 0.0] and r["n_tokens"] == 6 and r["n_windows"] == 3
    # increase NLL by 0.1 per token in every window -> constant PPL ratio e^0.1; difference > 0 in every resample
    r2 = paired_bootstrap_ppl(nll + 0.1 * tok, nll, tok, n_boot=300)
    assert r2["ppl_a"] / r2["ppl_b"] == pytest.approx(math.exp(0.1))
    assert r2["ci95"][0] > 0 and r2["ci95"][0] <= r2["diff"] <= r2["ci95"][1]
    with pytest.raises(ValueError):
        paired_bootstrap_ppl([1.0], [1.0, 2.0], [1.0])


# --- gap closed ------------------------------------------------------------------

def test_gap_closed():
    assert gap_closed(0.82, 0.77, 0.87) == pytest.approx(0.5)
    assert gap_closed(0.87, 0.77, 0.87) == pytest.approx(1.0)
    assert gap_closed(0.72, 0.77, 0.87) == pytest.approx(-0.5)
    assert gap_closed(0.8, 0.8, 0.8) is None


# --- decision logic ----------------------------------------------------------------

def _t(sig, dir_):
    return {"significant": sig, "in_direction": dir_}


@pytest.mark.parametrize("tests, expected", [
    ([_t(True, True), _t(True, True)], SUPPORTED),
    ([_t(True, True), _t(False, True)], PARTIAL),
    ([_t(False, True), _t(True, True)], PARTIAL),
    ([_t(True, True), _t(True, False)], INCONSISTENT),
    ([_t(True, False), _t(True, True)], INCONSISTENT),
    ([_t(False, True), _t(False, False)], NOT_SUPPORTED),
    ([_t(False, False), _t(False, False)], NOT_SUPPORTED),
    ([_t(True, False), _t(False, True)], OPPOSITE),
    ([_t(True, False), _t(True, False)], OPPOSITE),
])
def test_decide(tests, expected):
    assert decide(tests) == expected
