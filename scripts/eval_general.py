"""General benchmarks: log-likelihood based, zero-shot, bf16, batched, no chat template.

Tasks: mmlu (" A".." D" next token), hellaswag and arc_c (log-likelihood of the choice
continuations; acc and acc_norm), wikitext2 (sliding-window perplexity).

For each task, <out>/<task>.raw.jsonl (first), <task>.jsonl and <task>.summary.json
(last) are written. Tasks with a summary are skipped (rerun with --overwrite). A single
example's error does not fail the run (n_errors); on OOM the batch size shrinks automatically.

Example:
  python scripts/eval_general.py --model student_instruct --tasks all
  python scripts/eval_general.py --model student_instruct --tasks mmlu,arc_c --limit 20
  python scripts/eval_general.py --model student_instruct --tasks all --adapter $WORKSPACE/adapters/S2_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kdq.config import load_config, repo_path  # noqa: E402
from kdq.data import save_json  # noqa: E402
from kdq.evaluation import write_jsonl  # noqa: E402
from kdq.general import (  # noqa: E402
    DOC_BUILDERS, LETTERS, TASKS, ItemError, choice_metrics, load_task_dataset, perplexity,
    plan_windows, rows_sha256, run_batched,
)
from kdq.stats import wilson_ci  # noqa: E402
from kdq.utils import adapter_tag, run_metadata, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class LM:
    def __init__(self, repo: str, adapter: str | None, max_length: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model
        self.max_length = max_length

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def forward_last(self, seqs: list[list[int]], keep: int):
        """Left-padded batched forward pass; logits of the last `keep` positions (B x keep x V)."""
        torch = self.torch
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, L - len(s):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            out = self.model(input_ids=ids.cuda(), attention_mask=mask.cuda(),
                             position_ids=pos.cuda(), logits_to_keep=keep)
        return out.logits

    def empty_cache(self) -> None:
        self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multiple-choice tasks
# ---------------------------------------------------------------------------

def choice_request(lm: LM, context: str, cont: str) -> tuple[list[int], list[int]]:
    """(input, target) tokens; lm-eval style: whole = context + continuation, boundary from the context encoding."""
    ctx = lm.encode(context)
    whole = lm.encode(context + cont)
    if len(whole) > len(ctx) and whole[: len(ctx)] == ctx:
        tgt = whole[len(ctx):]
    else:  # encode separately if the token boundary shifted on concatenation
        tgt = lm.encode(cont)
        whole = ctx + tgt
    whole = whole[-(lm.max_length + 1):]
    return whole[:-1], tgt


def run_choice_task(task: str, lm: LM, rows: list[dict], src_idx: list[int], bs: int) -> tuple[list[dict], int]:
    torch = lm.torch
    docs = []
    for i, row in zip(src_idx, rows):
        rec = {"idx": i}
        if task == "mmlu":
            rec["subject"] = row.get("subject")
        try:
            ctx, choices, gold = DOC_BUILDERS[task](row)
            rec.update({"gold": gold, "_ctx": ctx, "_choices": choices, "error": None})
        except Exception as e:  # noqa: BLE001
            rec.update({"gold": None, "error": f"{type(e).__name__}: {e}"[:300]})
        docs.append(rec)
    valid = [d for d in docs if not d["error"]]

    if task == "mmlu":
        letter_ids = []
        for L in LETTERS:
            ids = lm.encode(" " + L)
            if len(ids) != 1:
                raise RuntimeError(f"' {L}' is not a single token: {ids}")
            letter_ids.append(ids[0])
        items = [(d, lm.encode(d["_ctx"])[-lm.max_length:]) for d in valid]
        items.sort(key=lambda x: -len(x[1]))

        def fn(batch):
            logits = lm.forward_last([seq for _, seq in batch], 1)
            lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            return lp[:, letter_ids].tolist()

        res, bs = run_batched(items, fn, bs, on_oom=lm.empty_cache)
        for (d, _), r in zip(items, res):
            if isinstance(r, ItemError):
                d["error"] = r.msg
            else:
                d["logprobs"], d["cont_chars"] = r, [1] * len(LETTERS)
    else:
        items = []
        for d in valid:
            for c_i, choice in enumerate(d["_choices"]):
                inp, tgt = choice_request(lm, d["_ctx"], " " + choice)
                items.append((d, c_i, inp, tgt))
            d["logprobs"] = [None] * len(d["_choices"])
            d["cont_chars"] = [len(c) for c in d["_choices"]]  # lm-eval acc_norm: choice text length
        items.sort(key=lambda x: -len(x[2]))

        def fn(batch):
            k = max(len(t) for *_, t in batch)
            logits = lm.forward_last([inp for _, _, inp, _ in batch], k)
            out = []
            for r, (_, _, _, tgt) in enumerate(batch):
                n = len(tgt)
                lp = torch.log_softmax(logits[r, k - n:, :].float(), dim=-1)
                t = torch.tensor(tgt, device=lp.device)
                out.append(float(lp.gather(1, t[:, None]).sum()))
            return out

        res, bs = run_batched(items, fn, bs, on_oom=lm.empty_cache)
        for (d, c_i, _, _), r in zip(items, res):
            if isinstance(r, ItemError):
                d["error"] = d["error"] or r.msg
            else:
                d["logprobs"][c_i] = r

    raw = [{k: v for k, v in d.items() if not k.startswith("_")} for d in docs]
    return raw, bs


def score_choice(raw: list[dict]) -> list[dict]:
    out = []
    for r in raw:
        r = dict(r)
        if r.get("error") or r.get("logprobs") is None or None in r["logprobs"]:
            r.update({"pred": None, "pred_norm": None, "correct": False, "correct_norm": False,
                      "error": r.get("error") or "missing log-likelihood"})
        else:
            r.update(choice_metrics(r["logprobs"], r["cont_chars"], r["gold"]))
        out.append(r)
    return out


def choice_summary(task: str, recs: list[dict]) -> dict:
    n = len(recs)
    k = sum(r["correct"] for r in recs)
    kn = sum(r["correct_norm"] for r in recs)
    s = {
        "task": task, "n": n, "n_errors": sum(bool(r.get("error")) for r in recs),
        "acc": k / n if n else None, "acc_ci95": list(wilson_ci(k, n)),
        "acc_norm": kn / n if n else None, "acc_norm_ci95": list(wilson_ci(kn, n)),
        "primary_metric": "acc" if task == "mmlu" else "acc_norm",
        "note": "failed examples count as incorrect (included in n)",
    }
    if task == "mmlu":
        s["acc_norm_note"] = "MMLU choices are single letters; acc_norm = acc"
        by: dict = {}
        for r in recs:
            b = by.setdefault(r.get("subject"), [0, 0])
            b[0] += 1
            b[1] += bool(r["correct"])
        s["by_subject"] = {sub: {"n": v[0], "acc": v[1] / v[0]} for sub, v in sorted(by.items(), key=lambda x: str(x[0]))}
    return s


# ---------------------------------------------------------------------------
# WikiText-2 perplexity
# ---------------------------------------------------------------------------

def run_wikitext(lm: LM, texts: list[str], window: int, stride: int, bs: int, limit: int | None):
    torch = lm.torch
    text = "\n\n".join(texts)
    ids = lm.encode(text)
    wins = plan_windows(len(ids), window, stride)
    if limit:
        wins = wins[:limit]
    items = [(w_i, ids[b:e], e - s, (b, e, s)) for w_i, (b, e, s) in enumerate(wins)]

    def fn(batch):
        k = max(n for _, _, n, _ in batch)
        logits = lm.forward_last([seq[:-1] for _, seq, _, _ in batch], k)
        out = []
        for r, (_, seq, n, _) in enumerate(batch):
            lp = torch.log_softmax(logits[r, k - n:, :].float(), dim=-1)
            t = torch.tensor(seq[-n:], device=lp.device)
            out.append(float(-lp.gather(1, t[:, None]).sum()))
        return out

    res, bs = run_batched(items, fn, bs, on_oom=lm.empty_cache)
    raw = []
    for (w_i, _, n, (b, e, s)), r in zip(items, res):
        err = r.msg if isinstance(r, ItemError) else None
        raw.append({"window": w_i, "begin": b, "end": e, "score_from": s, "n_tokens": n,
                    "nll_sum": None if err else r, "error": err})
    info = {"text_chars": len(text), "text_tokens": len(ids), "window": window, "stride": stride}
    return raw, info, bs


# ---------------------------------------------------------------------------

def load_subset(cfg, task: str, ds, limit: int | None) -> tuple[list[int], str | None]:
    """(indices, expected sha256). MMLU/HellaSwag: data/splits/general_<task>.json."""
    n = cfg["general"]["datasets"][task]["n"]
    if n == "all":
        idx, sha = list(range(len(ds))), None
    else:
        p = repo_path(cfg["paths"]["splits_dir"]) / f"general_{task}.json"
        if not p.exists():
            sys.exit(f"[ERROR] {p} not found; first run: python scripts/make_general_subsets.py")
        obj = json.loads(p.read_text(encoding="utf-8"))
        idx, sha = obj["indices"], obj["sha256"]
    return (idx[:limit] if limit else idx), (None if limit else sha)


def main() -> int:
    ap = argparse.ArgumentParser(description="General-benchmark evaluation (log-likelihood)")
    ap.add_argument("--model", required=True, help="config key: teacher | student_instruct | student_base")
    ap.add_argument("--tasks", default="all", help="comma-separated: mmlu,hellaswag,arc_c,wikitext2 or all")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory (peft)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N examples/windows per task")
    ap.add_argument("--out", default=None, help="output directory (default results/general/<model>[_adapter][_limitN])")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    g = cfg["general"]
    tasks = list(TASKS) if a.tasks == "all" else [t.strip() for t in a.tasks.split(",")]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        sys.exit(f"[ERROR] unknown task(s): {bad}; choices: {TASKS}")
    repo = cfg["models"][a.model]
    name = a.model + (f"_{adapter_tag(a.adapter)}" if a.adapter else "") + (f"_limit{a.limit}" if a.limit else "")
    out_dir = Path(a.out) if a.out else repo_path(g["out_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in tasks if a.overwrite or not (out_dir / f"{t}.summary.json").exists()]
    for t in tasks:
        if t not in todo:
            print(f"[general] SKIP {t} (summary exists)")
    if not todo:
        return 0

    set_seed(g["seed"])
    t0 = time.perf_counter()
    lm = LM(repo, a.adapter, g["max_length"])
    load_s = time.perf_counter() - t0
    print(f"[general] {a.model} ({repo}){' + ' + a.adapter if a.adapter else ''} loaded: {load_s:.0f} s", flush=True)
    bs0 = a.batch_size or g["batch_size"]

    for task in todo:
        t1 = time.perf_counter()
        ds = load_task_dataset(cfg, task)
        base = {"task": task, "model_key": a.model, "repo_id": repo, "adapter": a.adapter,
                "limit": a.limit, "chat_template": False, "dtype": "bfloat16", "load_s": round(load_s, 1)}
        if task == "wikitext2":
            w = g["wikitext"]
            bs = min(bs0, w.get("batch_size", 4))
            raw, info, bs = run_wikitext(lm, list(ds["text"]), w["window"], w["stride"], bs, a.limit)
            write_jsonl(raw, out_dir / f"{task}.raw.jsonl")
            write_jsonl(raw, out_dir / f"{task}.jsonl")
            summary = {**base, **info, **perplexity(raw), "primary_metric": "ppl"}
            msg = f"ppl={summary['ppl']:.4f} ({summary['n_tokens']} tokens, {summary['n_windows']} windows)"
        else:
            idx, sha = load_subset(cfg, task, ds, a.limit)
            sub = ds.select(idx)
            if sha and rows_sha256(sub) != sha:
                print(f"[ERROR] {task}: subset content does not match the recorded sha256; skipped.")
                continue
            raw, bs = run_choice_task(task, lm, list(sub), idx, bs0)
            write_jsonl(raw, out_dir / f"{task}.raw.jsonl")  # before scoring
            recs = score_choice(raw)
            write_jsonl(recs, out_dir / f"{task}.jsonl")
            summary = {**base, **choice_summary(task, recs)}
            msg = (f"acc={summary['acc']:.4f} acc_norm={summary['acc_norm']:.4f} "
                   f"n={summary['n']} errors={summary['n_errors']}")
        summary.update({"time_s": round(time.perf_counter() - t1, 1), "final_batch_size": bs,
                        "meta": run_metadata(cfg, script="eval_general.py", argv=sys.argv[1:])})
        save_json(summary, out_dir / f"{task}.summary.json")
        print(f"[general] {task}: {msg} | {summary['time_s']:.0f} s | batch={bs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
