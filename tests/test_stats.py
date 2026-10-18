import pytest

from kdq.stats import mcnemar_exact, wilson_ci


def test_wilson_known_value():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)


def test_wilson_edges():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and 0.65 < lo < 1
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # b=0, c=6: 2 * (1/64) = 0.03125
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert mcnemar_exact(6, 0) == mcnemar_exact(0, 6)
    assert mcnemar_exact(1000, 900) < 0.05
