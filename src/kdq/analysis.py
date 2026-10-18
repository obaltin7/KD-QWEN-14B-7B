"""Statistical core of the pre-registered analyses (prereg-v1.0 §6). No dependency besides numpy.

- Holm–Bonferroni correction
- Question-level paired bootstrap (accuracy difference, points)
- Window-level paired bootstrap (WikiText-2 PPL difference)
- Gap-closed ratio (Sx − S0) / (T − S0)
- Hypothesis decision logic (two seeds, Holm significance and direction)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .stats import mcnemar_exact

ALPHA = 0.05
N_BOOT = 10_000
BOOT_SEED = 42

SUPPORTED = "supported"
PARTIAL = "partially supported"
NOT_SUPPORTED = "not supported"
OPPOSITE = "not supported (significant in the opposite direction)"
INCONSISTENT = "inconsistent (not supported)"


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (in input order). p_(i) -> max_{j<=i} min(1, (m-j+1)·p_(j))."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def mcnemar_paired(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """a and b over the same questions (same order). b = only a correct, c = only b correct."""
    a = np.asarray(a, dtype=bool)
    b_ = np.asarray(b, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("paired inputs must have equal length")
    n10 = int((a & ~b_).sum())
    n01 = int((~a & b_).sum())
    return {"n": int(a.size), "acc_a": float(a.mean()), "acc_b": float(b_.mean()),
            "diff_points": 100.0 * float(a.mean() - b_.mean()),
            "b_only_a": n10, "c_only_b": n01, "p": mcnemar_exact(n10, n01)}


def paired_bootstrap_acc(a: Sequence[bool], b: Sequence[bool], n_boot: int = N_BOOT,
                         seed: int = BOOT_SEED) -> tuple[float, float]:
    """Question-level paired bootstrap 95% percentile CI for (acc_a − acc_b)·100."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = 100.0 * d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def ppl(nll: Sequence[float], tokens: Sequence[float]) -> float:
    return float(np.exp(np.sum(nll) / np.sum(tokens)))


def paired_bootstrap_ppl(nll_a: Sequence[float], nll_b: Sequence[float], tokens: Sequence[float],
                         n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """PPL_a − PPL_b; window-level paired bootstrap (in each resample PPL = exp(ΣNLL/Σtokens))."""
    na = np.asarray(nll_a, dtype=np.float64)
    nb = np.asarray(nll_b, dtype=np.float64)
    t = np.asarray(tokens, dtype=np.float64)
    if not (na.shape == nb.shape == t.shape):
        raise ValueError("window counts must be equal")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, t.size, size=(n_boot, t.size))
    tt = t[idx].sum(axis=1)
    boots = np.exp(na[idx].sum(axis=1) / tt) - np.exp(nb[idx].sum(axis=1) / tt)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pa, pb = ppl(na, t), ppl(nb, t)
    return {"ppl_a": pa, "ppl_b": pb, "diff": pa - pb, "ci95": [float(lo), float(hi)],
            "n_windows": int(t.size), "n_tokens": int(t.sum())}


def gap_closed(sx: float, s0: float, t: float) -> float | None:
    """Gap-closed ratio (Sx − S0) / (T − S0); None if T == S0."""
    return None if t == s0 else (sx - s0) / (t - s0)


def decide(tests: Sequence[dict]) -> str:
    """Two-seed hypothesis decision (prereg §6). Each test: {"significant": bool, "in_direction": bool}.

    significant in the hypothesised direction in both seeds -> supported; in only one -> partially
    supported; in none -> not supported; none in the hypothesised direction but at least one in the
    opposite direction -> not supported (significant in the opposite direction); one seed in the
    hypothesised direction and the other in the opposite direction -> inconsistent (not supported).
    """
    support = sum(1 for t in tests if t["significant"] and t["in_direction"])
    against = sum(1 for t in tests if t["significant"] and not t["in_direction"])
    if support >= 1 and against >= 1:
        return INCONSISTENT
    if support == len(tests) and support > 0:
        return SUPPORTED
    if support >= 1:
        return PARTIAL
    if against >= 1:
        return OPPOSITE
    return NOT_SUPPORTED
