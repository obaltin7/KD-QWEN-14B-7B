"""GSM8K prompt format: a single, fixed, plain-text format; NO chat template.

Decision (final, docs/decisions.md): the same format for all models (including Instruct):
    prompt = "Question: {question}\nAnswer:"
    target = " {solution}"         (a GSM8K solution ends with "#### number")
k-shot: examples (prompt + target) and the query are joined with "\n\n".
Calculator annotations (<<48/2=24>>) are removed from solutions; the same cleaning
is applied to training targets and few-shot examples (clean_solution).
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

_CALC_RE = re.compile(r"<<[^>]*>>")
SHOT_SEP = "\n\n"


def clean_solution(solution: str) -> str:
    return _CALC_RE.sub("", solution).strip()


def format_question(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def format_target(solution: str) -> str:
    """Training target / few-shot answer, including the leading space."""
    return " " + clean_solution(solution)


def format_example(question: str, solution: str) -> str:
    return format_question(question) + format_target(solution)


def build_prompt(question: str, shots: Iterable[Mapping[str, str]] = ()) -> str:
    """shots: {"question", "answer"} dicts (GSM8K rows)."""
    parts = [format_example(ex["question"], ex["answer"]) for ex in shots]
    parts.append(format_question(question))
    return SHOT_SEP.join(parts)


def truncate_at_stop(text: str, stops: Sequence[str]) -> str:
    """Truncates the text at the first stop sequence (the sequence is removed)."""
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]
