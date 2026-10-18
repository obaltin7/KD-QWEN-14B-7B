"""GSM8K gold-answer / prediction extraction and numeric comparison.

Rules:
- Gold answer: the text after "####", with commas removed.
- Prediction: first number after the last "####"; otherwise the last number in the text.
  "$", thousands separators, decimals, negatives and sentence-final periods are supported.
- Comparison: numeric via Decimal (3 == 3.0 == 3.00), with a small tolerance.
Assumption: fractions ("1/2") and scientific notation are not supported; GSM8K answers are integers.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, localcontext

# Try comma-grouped numbers (1,234,567.5) before plain numbers (1234.5).
# Reject numbers that are part of a word or of a decimal (e.g. "x2", "1.2.3").
_NUM_RE = re.compile(
    r"(?<![\w.])(-)?\s?\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
)
_TOL = Decimal("1e-6")
# Digit strings longer than this (the model occasionally emits hundreds of digits) are
# invalid: _normalize returns None and the extraction step is treated as unanswered.
MAX_DIGITS = 30


def _normalize(sign: str | None, digits: str) -> str | None:
    """Converts a number to its canonical string; None if invalid/too long/infinite (never raises)."""
    raw = digits.replace(",", "")
    if sum(c.isdigit() for c in raw) > MAX_DIGITS:
        return None
    try:
        # The default context precision is 28 digits; quantize raises InvalidOperation on overflow.
        with localcontext() as ctx:
            ctx.prec = 2 * MAX_DIGITS + 10
            d = Decimal(raw)
            if not d.is_finite():
                return None
            if sign:
                d = -d
            # Strip trailing zeros: "3.50" -> "3.5", "3.0" -> "3"
            d = d.normalize()
            if d == d.to_integral_value():
                d = d.quantize(Decimal(1))
            out = format(d, "f")
    except (InvalidOperation, ValueError, OverflowError):
        return None
    return "0" if out == "-0" else out


def _parse(m: re.Match | None) -> str | None:
    return _normalize(m.group(1), m.group(2)) if m else None


def _numbers(text: str) -> list[str | None]:
    """Number tokens in the text; invalid ones remain in place as None."""
    return [_parse(m) for m in _NUM_RE.finditer(text)]


def extract_gold(answer: str) -> str:
    """Extracts the gold answer from the GSM8K 'answer' field ("#### 1,234" -> "1234")."""
    if "####" not in answer:
        raise ValueError("'####' not found in gold answer")
    tail = answer.split("####")[-1].strip()
    nums = _numbers(tail)
    if not nums or nums[0] is None:
        raise ValueError(f"no valid number after '####': {tail!r}")
    return nums[0]


def extract_pred(text: str | None) -> str | None:
    """Extracts the prediction from model output; None if no number is present."""
    if not text:
        return None
    if "####" in text:
        nums = _numbers(text.split("####")[-1])
        if nums:
            return nums[0]
    nums = _numbers(text)
    return nums[-1] if nums else None


# ---------------------------------------------------------------------------
# Answer extraction v2 (strict + flexible). Defined from raw generations after the
# baseline stage; frozen before training and applied identically to all configurations.
# extract_pred (v1) is kept only for backward compatibility.
# ---------------------------------------------------------------------------

# Markers of model-hallucinated new questions at line start, AFTER the first line:
# "Question", "[Question]", "Q:", "Q1:", "Problem:", "### Question" (case-insensitive).
_NEW_Q_RE = re.compile(
    r"(?<=\n)[ \t]*(?:#+[ \t]*)?"
    r"(?:\[[ \t]*(?:question|problem)[ \t]*\d*[ \t]*\]|(?:question|problem)\b|q[ \t]*\d*[ \t]*[:.])",
    re.IGNORECASE,
)
_ANSWER_IS_RE = re.compile(r"answer\s+is\s*:?\s*", re.IGNORECASE)


def truncate_generation(text: str) -> tuple[str, int | None]:
    """Truncates the generation at the first new-question marker and at the end of the first "####" line.

    Returns: (truncated text, cut position | None).
    """
    cut = len(text)
    m = _NEW_Q_RE.search(text)
    if m:
        cut = m.start() - 1  # also exclude the "\n" preceding the marker
    h = text.find("####")
    if h != -1 and h < cut:
        eol = text.find("\n", h)
        if eol != -1:
            cut = min(cut, eol)
    return text[:cut], (cut if cut < len(text) else None)


def _strict_match(text: str) -> re.Match | None:
    """The first "####" and the number token immediately following it (skipping whitespace)."""
    h = text.find("####")
    if h == -1:
        return None
    pos = h + 4
    while pos < len(text) and text[pos] in " \t":
        pos += 1
    return _NUM_RE.match(text, pos)


def extract_strict(text: str | None) -> str | None:
    """Only the number immediately after the first "####"; None if absent or invalid."""
    if not text:
        return None
    return _parse(_strict_match(text))


def extract_flexible(text: str | None) -> tuple[str | None, int | None]:
    """Truncate -> "#### number" -> "answer is (:) number" (first) -> last number.

    If a step finds a number token, that step's result is used; if the token is
    invalid (e.g. > MAX_DIGITS digits) the result is None (unanswered) and later
    steps are not tried.
    Returns: (prediction | None, cut position | None).
    """
    if not text:
        return None, None
    cut_text, cut_at = truncate_generation(text)
    m = _strict_match(cut_text)
    if m:
        return _parse(m), cut_at
    for a in _ANSWER_IS_RE.finditer(cut_text):
        m = _NUM_RE.match(cut_text, a.end())
        if m:
            return _parse(m), cut_at
    nums = _numbers(cut_text)
    return (nums[-1] if nums else None), cut_at


def is_correct(pred: str | None, gold: str) -> bool:
    """Numeric equality (tolerance 1e-6). False if pred is None."""
    if pred is None:
        return False
    try:
        return abs(Decimal(pred) - Decimal(gold)) <= _TOL
    except (InvalidOperation, ValueError):
        return False
