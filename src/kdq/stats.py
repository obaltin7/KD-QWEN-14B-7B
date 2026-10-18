"""Confidence intervals and paired tests (dependency-free)."""

from __future__ import annotations

import math
from fractions import Fraction


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return (lo, hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact (binomial) two-sided McNemar p-value.

    b: only A correct, c: only B correct (discordant pairs).
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))
