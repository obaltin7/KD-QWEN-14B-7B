"""General benchmarks (forgetting axis): MMLU, HellaSwag, ARC-Challenge, WikiText-2.

Logic that requires no GPU/torch lives here (tested locally): subset sampling,
prompt/choice formats, scoring from log-likelihoods, sliding-window planning,
perplexity aggregation, and a runner that shrinks the batch size on out-of-memory errors.
Formats are zero-shot without a chat template (consistent with the plain GSM8K format).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

TASKS = ("mmlu", "hellaswag", "arc_c", "wikitext2")
LETTERS = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Datasets and subsets
# ---------------------------------------------------------------------------

def load_task_dataset(cfg: Mapping[str, Any], task: str):
    """Loads the split defined by general.datasets[task] in configs/base.yaml."""
    from datasets import load_dataset

    d = cfg["general"]["datasets"][task]
    try:
        return load_dataset(d["path"], d.get("name"), split=d["split"])
    except Exception:
        # Legacy script-based repositories (e.g. Rowan/hellaswag) may fail to load with datasets>=4;
        # HF's automatic parquet conversion is tried as a fallback.
        if not d.get("fallback_revision"):
            raise
        return load_dataset(d["path"], d.get("name"), split=d["split"], revision=d["fallback_revision"])


def rows_sha256(rows) -> str:
    """Order-sensitive content digest of rows (dicts)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def proportional_allocation(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Allocates n samples to groups proportionally via the largest-remainder method (sum exactly n)."""
    total = sum(counts.values())
    if not 0 < n <= total:
        raise ValueError(f"invalid n={n} (total {total})")
    quotas = {k: n * c / total for k, c in counts.items()}
    alloc = {k: int(math.floor(q)) for k, q in quotas.items()}
    rest = n - sum(alloc.values())
    # remainder to the largest fractional parts; ties broken by group name (deterministic)
    order = sorted(counts, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in order[:rest]:
        alloc[k] += 1
    return alloc


def stratified_sample(groups: Sequence[str], n: int, seed: int) -> list[int]:
    """groups[i] = group of example i. Returns a sorted, group-proportional index list."""
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    alloc = proportional_allocation({g: len(v) for g, v in by_group.items()}, n)
    rng = random.Random(seed)
    picked: list[int] = []
    for g in sorted(by_group):
        picked += rng.sample(by_group[g], alloc[g])
    return sorted(picked)


def simple_sample(n_total: int, n: int, seed: int) -> list[int]:
    if not 0 < n <= n_total:
        raise ValueError(f"invalid n={n} (total {n_total})")
    return sorted(random.Random(seed).sample(range(n_total), n))


def group_counts(groups: Sequence[str], idx: Sequence[int]) -> dict[str, int]:
    return dict(sorted(Counter(groups[i] for i in idx).items()))


# ---------------------------------------------------------------------------
# Prompt / choice formats
# ---------------------------------------------------------------------------

def mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    """"Question\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nAnswer:" — predicted via the " A".." D" tokens."""
    lines = [question.strip()] + [f"{L}. {c}" for L, c in zip(LETTERS, choices)]
    return "\n".join(lines) + "\nAnswer:"


def hellaswag_preprocess(text: str) -> str:
    """lm-eval-harness HellaSwag preprocessing."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag_doc(row: Mapping[str, Any]) -> tuple[str, list[str], int]:
    ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
    context = hellaswag_preprocess(row["activity_label"] + ": " + ctx)
    choices = [hellaswag_preprocess(e) for e in row["endings"]]
    return context, choices, int(row["label"])


def arc_doc(row: Mapping[str, Any]) -> tuple[str, list[str], int]:
    context = f"Question: {row['question'].strip()}\nAnswer:"
    choices = list(row["choices"]["text"])
    gold = list(row["choices"]["label"]).index(row["answerKey"])
    return context, choices, gold


def mmlu_doc(row: Mapping[str, Any]) -> tuple[str, list[str], int]:
    return mmlu_prompt(row["question"], row["choices"]), list(LETTERS), int(row["answer"])


DOC_BUILDERS: dict[str, Callable] = {"mmlu": mmlu_doc, "hellaswag": hellaswag_doc, "arc_c": arc_doc}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _argmax(xs: Sequence[float]) -> int:
    return max(range(len(xs)), key=lambda i: (xs[i], -i))  # ties go to the first choice


def choice_metrics(logprobs: Sequence[float], cont_chars: Sequence[int], gold: int) -> dict:
    """acc: total log-likelihood; acc_norm: normalised by character length (lm-eval)."""
    pred = _argmax(logprobs)
    norm = [lp / max(1, n) for lp, n in zip(logprobs, cont_chars)]
    pred_norm = _argmax(norm)
    return {"pred": pred, "pred_norm": pred_norm,
            "correct": pred == gold, "correct_norm": pred_norm == gold}


def perplexity(windows: Sequence[Mapping[str, Any]]) -> dict:
    """Token-level perplexity from window records (nll_sum, n_tokens)."""
    ok = [w for w in windows if not w.get("error")]
    nll = sum(w["nll_sum"] for w in ok)
    n = sum(w["n_tokens"] for w in ok)
    return {"ppl": math.exp(nll / n) if n else None, "nll_sum": nll, "n_tokens": n,
            "n_windows": len(windows), "n_errors": len(windows) - len(ok)}


def plan_windows(n_tokens: int, max_len: int, stride: int) -> list[tuple[int, int, int]]:
    """Sliding windows: (begin, end, score_from). Tokens in [score_from, end) are
    scored; every token 1..n_tokens-1 is scored exactly once (token 0 has no
    context). Same scheme as the HF perplexity guide."""
    if stride <= 0 or max_len <= 1 or stride > max_len:
        raise ValueError("invalid window/stride")
    wins, prev_end, begin = [], 0, 0
    while begin < n_tokens:
        end = min(begin + max_len, n_tokens)
        s = max(prev_end, begin + 1)
        if end > s:
            wins.append((begin, end, s))
        prev_end = end
        if end == n_tokens:
            break
        begin += stride
    return wins


# ---------------------------------------------------------------------------
# Batched execution: halve the batch size on OOM, isolate single-item errors
# ---------------------------------------------------------------------------

def is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


class ItemError:
    """Error of a single item (stored in place of its value in the result list)."""

    def __init__(self, msg: str):
        self.msg = msg[:300]

    def __repr__(self):
        return f"ItemError({self.msg!r})"


def run_batched(items: Sequence[Any], fn: Callable[[list], list], batch_size: int,
                on_oom: Callable[[], None] | None = None, log: Callable[[str], None] = print) -> tuple[list, int]:
    """Processes items in batches with fn; results are returned in item order.

    - OOM: on_oom() (e.g. emptying the CUDA cache), batch size halved, same batch retried.
      OOM at batch size 1 -> that item becomes an ItemError.
    - Other errors: the batch is retried item by item; failing items become ItemError, the run continues.
    Returns: (results, final batch size).
    """
    out: list = [None] * len(items)
    bs, i = max(1, batch_size), 0
    while i < len(items):
        idx = list(range(i, min(i + bs, len(items))))
        try:
            res = fn([items[j] for j in idx])
            for j, r in zip(idx, res):
                out[j] = r
            i += len(idx)
        except Exception as e:  # noqa: BLE001
            if is_oom(e) and bs > 1:
                if on_oom:
                    on_oom()
                bs = max(1, bs // 2)
                log(f"[WARN] out of memory; batch size -> {bs}")
                continue
            if on_oom and is_oom(e):
                on_oom()
            for j in idx:  # one by one
                try:
                    out[j] = fn([items[j]])[0]
                except Exception as e2:  # noqa: BLE001
                    if on_oom and is_oom(e2):
                        on_oom()
                    out[j] = ItemError(f"{type(e2).__name__}: {e2}")
            i += len(idx)
    return out, bs
