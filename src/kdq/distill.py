"""Distillation training (S1/S2/S3): data preparation, losses, training loop.

Exactly as in docs/preregistration.md (prereg-v1.0):
  S1: L = CE
  S2: L = (1-α)·CE + α·τ²·KL(p_T^τ || p_S^τ)     (forward KL, fp32, first vocab_cut indices)
  S3: L = L_S2 + β·(1/|pairs|)·Σ MSE(LN(W·h_S^(l)), LN(h_T^(m)))
All losses are computed on answer tokens only and averaged at the token level over the
effective batch. Model loading lives in scripts/train.py; everything here is tested on CPU
with small fake models (tests/test_distill.py).

Hidden-state indexing (HF output_hidden_states): index 0 = embedding output, index l =
output of layer l, the LAST index (= number of layers) is the state AFTER the final norm
(model.norm); hence the (28, 48) pair matches the normalised final state in both models,
and hidden_states[-1] is the input to lm_head (logits are computed from it at selected positions).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prompts import format_question, format_target

IGNORE = -100


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def tokenize_example(tok, question: str, solution: str, max_len: int, eos_id: int) -> dict:
    """Input = prompt + target + eos; prompt tokens are IGNORE in the labels.

    Continuation tokens are taken as enc(prompt + target)[len(enc(prompt)):] (the same
    boundary as in evaluation tokenisation); on a prefix mismatch the target is encoded
    separately and counted.
    """
    prompt, target = format_question(question), format_target(solution)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full = tok(prompt + target, add_special_tokens=False)["input_ids"]
    prefix_ok = len(full) > len(p_ids) and full[: len(p_ids)] == p_ids
    if not prefix_ok:
        full = p_ids + tok(target, add_special_tokens=False)["input_ids"]
    ids = full + [eos_id]
    truncated = len(ids) > max_len
    ids = ids[:max_len]
    n_p = min(len(p_ids), len(ids))
    labels = [IGNORE] * n_p + ids[n_p:]
    return {"input_ids": ids, "labels": labels, "n_prompt": len(p_ids),
            "truncated": truncated, "prefix_mismatch": not prefix_ok}


def build_examples(rows: Sequence[Mapping[str, str]], tok, max_len: int, eos_id: int) -> tuple[list[dict], dict]:
    exs = [tokenize_example(tok, r["question"], r["answer"], max_len, eos_id) for r in rows]
    lens = [len(e["input_ids"]) for e in exs]
    stats = {
        "n": len(exs), "max_seq_len": max_len, "eos_id": eos_id,
        "n_truncated": sum(e["truncated"] for e in exs),
        "truncated_indices": [i for i, e in enumerate(exs) if e["truncated"]][:50],
        "n_prefix_mismatch": sum(e["prefix_mismatch"] for e in exs),
        "max_len_tokens": max(lens) if lens else 0,
        "mean_len_tokens": sum(lens) / len(lens) if lens else 0.0,
        "n_answer_tokens": sum(sum(l != IGNORE for l in e["labels"]) for e in exs),
    }
    return exs, stats


def collate(examples: Sequence[Mapping], pad_id: int) -> dict[str, torch.Tensor]:
    """Right padding."""
    B, L = len(examples), max(len(e["input_ids"]) for e in examples)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, L), IGNORE, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def epoch_orders(n: int, epochs: int, seed: int) -> list[list[int]]:
    """Per-epoch example order; depends only on the seed (S1/S2/S3 see the same order)."""
    rng = random.Random(seed)
    return [rng.sample(range(n), n) for _ in range(epochs)]


def orders_sha256(orders: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(json.dumps([list(o) for o in orders]).encode()).hexdigest()


def steps_per_epoch(n: int, effective_batch: int) -> int:
    return math.ceil(n / effective_batch)


# ---------------------------------------------------------------------------
# Positions and losses
# ---------------------------------------------------------------------------

def prediction_positions(labels: torch.Tensor):
    """CE/KL positions: t such that labels[t+1] != IGNORE (t predicts t+1).

    Returns: (b_idx, t_idx, targets). Prompt and padding tokens are never targets.
    """
    shifted = labels[:, 1:]
    b, t = (shifted != IGNORE).nonzero(as_tuple=True)
    return b, t, shifted[b, t]


def answer_positions(labels: torch.Tensor):
    """Hidden-state alignment positions: the answer tokens themselves (labels != IGNORE)."""
    return (labels != IGNORE).nonzero(as_tuple=True)


def kd_kl_sum(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float) -> torch.Tensor:
    """τ²·Σ_n KL(p_T^τ || p_S^τ) (forward KL), fp32. Inputs are vocabulary-truncated (N, V_cut)."""
    ls = F.log_softmax(student_logits.float() / tau, dim=-1)
    lt = F.log_softmax(teacher_logits.float() / tau, dim=-1)
    kl = (lt.exp() * (lt - ls)).sum(-1)
    return (tau * tau) * kl.sum()


class HiddenProjector(nn.Module):
    """W: d_S -> d_T linear projection (fp32). shared=True: a single W for all pairs."""

    def __init__(self, d_s: int, d_t: int, n_pairs: int, shared: bool = True):
        super().__init__()
        self.shared = shared
        self.proj = nn.ModuleList([nn.Linear(d_s, d_t) for _ in range(1 if shared else n_pairs)])
        self.float()

    def forward(self, pair_i: int, h: torch.Tensor) -> torch.Tensor:
        return self.proj[0 if self.shared else pair_i](h.float())


def hidden_loss_sum(student_hs, teacher_hs, pairs, projector: HiddenProjector, b_idx, t_idx) -> torch.Tensor:
    """(1/|pairs|)·Σ_pairs Σ_n MSE(LN(W·h_S), LN(h_T)); MSE = element-wise mean, LN without affine.

    Returns the SUM over positions; the caller divides by the number of positions in the effective batch.
    """
    total = torch.zeros((), dtype=torch.float32, device=b_idx.device)
    for i, (l, m) in enumerate(pairs):
        hs = student_hs[l][b_idx, t_idx]
        ht = teacher_hs[m][b_idx, t_idx].float()
        ps = F.layer_norm(projector(i, hs), (ht.shape[-1],))
        pt = F.layer_norm(ht, (ht.shape[-1],))
        total = total + F.mse_loss(ps, pt, reduction="none").mean(-1).sum()
    return total / len(pairs)


def forward_selected(model, batch: Mapping[str, torch.Tensor], b_idx, t_idx, need_hidden: bool):
    """To save memory, logits only at selected positions: lm_head(hidden_states[-1][positions]).

    hidden_states[-1] is the state after the final norm (lm_head input); logits_to_keep=1
    skips the model's own full logit computation.
    """
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                output_hidden_states=True, logits_to_keep=1)
    h_last = out.hidden_states[-1]
    logits = model.get_output_embeddings()(h_last[b_idx, t_idx])
    return logits, (out.hidden_states if need_hidden else None)


@dataclass
class LossSpec:
    kd: bool
    hidden: bool
    tau: float
    alpha: float
    beta: float
    vocab_cut: int
    pairs: list


def micro_losses(student, teacher, batch, spec: LossSpec, projector: HiddenProjector | None) -> dict:
    """Loss SUMS for one micro-batch (the caller averages at the effective-batch level)."""
    labels = batch["labels"]
    b, t, targets = prediction_positions(labels)
    s_logits, s_hs = forward_selected(student, batch, b, t, spec.hidden)
    s_logits = s_logits[:, : spec.vocab_cut].float()   # CE and KL on the same support
    out = {"ce_sum": F.cross_entropy(s_logits, targets, reduction="sum"), "n_pred": int(targets.numel())}
    if spec.kd or spec.hidden:
        with torch.no_grad():
            t_logits, t_hs = forward_selected(teacher, batch, b, t, spec.hidden)
            t_logits = t_logits[:, : spec.vocab_cut].float()
        if spec.kd:
            out["kl_sum"] = kd_kl_sum(s_logits, t_logits, spec.tau)
        if spec.hidden:
            ab, at = answer_positions(labels)
            out["feat_sum"] = hidden_loss_sum(s_hs, t_hs, spec.pairs, projector, ab, at)
            out["n_ans"] = int(ab.numel())
    return out


def combine_losses(m: Mapping[str, Any], spec: LossSpec, n_pred_total: int, n_ans_total: int) -> torch.Tensor:
    """Converts micro-batch sums into their contribution to the effective-batch mean (for backward)."""
    if spec.kd:
        loss = ((1 - spec.alpha) * m["ce_sum"] + spec.alpha * m["kl_sum"]) / n_pred_total
    else:
        loss = m["ce_sum"] / n_pred_total
    if spec.hidden:
        loss = loss + spec.beta * m["feat_sum"] / max(1, n_ans_total)
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def count_tokens(examples: Sequence[Mapping]) -> tuple[int, int]:
    n_pred = sum(sum(l != IGNORE for l in e["labels"][1:]) for e in examples)
    n_ans = sum(sum(l != IGNORE for l in e["labels"]) for e in examples)
    return n_pred, n_ans


def write_jsonl_line(path: Path, rec: Mapping) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def latest_checkpoint(out_dir: Path) -> Path | None:
    d = out_dir / "checkpoints"
    if not d.exists():
        return None
    cks = sorted((p for p in d.iterdir() if p.name.startswith("step") and (p / "state.pt").exists()),
                 key=lambda p: int(p.name[4:]))
    return cks[-1] if cks else None


def save_checkpoint(path: Path, student, projector, opt, sched, step: int) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    student.save_pretrained(tmp / "adapter")
    state = {"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "projector": projector.state_dict() if projector is not None else None,
             "rng_cpu": torch.get_rng_state(),
             "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    torch.save(state, tmp / "state.pt")
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)


def load_checkpoint(path: Path, student, projector, opt, sched) -> int:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(str(path / "adapter" / "adapter_model.safetensors"))
    set_peft_model_state_dict(student, sd)
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    if projector is not None and state["projector"] is not None:
        projector.load_state_dict(state["projector"])
    torch.set_rng_state(state["rng_cpu"])
    if state["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return int(state["step"])


def train_run(*, student, teacher, examples: Sequence[Mapping], spec: LossSpec, tcfg: Mapping[str, Any],
              seed: int, lr: float, epochs: int, out_dir: Path, device, pad_id: int,
              max_steps: int | None = None, resume: bool = False, micro_batch: int | None = None,
              ckpt_every: int | None = None, log: Callable[[str], None] = print) -> dict:
    """Training loop. student: PeftModel (LoRA trainable); teacher: frozen or None."""
    from transformers import get_cosine_schedule_with_warmup

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(examples)
    eb = int(tcfg["effective_batch"])
    mb = int(micro_batch or tcfg["micro_batch"])
    if eb % mb != 0:
        raise ValueError(f"effective_batch {eb} is not divisible by micro_batch {mb}")
    spe = steps_per_epoch(n, eb)
    total_steps = epochs * spe
    orders = epoch_orders(n, epochs, seed)
    warmup = int(round(tcfg["warmup_ratio"] * total_steps))

    projector = None
    if spec.hidden:
        h = tcfg["hidden"]
        projector = HiddenProjector(h["student_dim"], h["teacher_dim"], len(spec.pairs),
                                    shared=(h.get("projection", "shared") == "shared")).to(device)
    params = [p for p in student.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in params)
    if projector is not None:
        params += list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(tcfg["weight_decay"]))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    log_path = out_dir / "train_log.jsonl"
    start_step = 0
    if resume:
        ck = latest_checkpoint(out_dir)
        if ck is not None:
            start_step = load_checkpoint(ck, student, projector, opt, sched)
            log(f"[train] resuming: {ck} (step {start_step})")
            if log_path.exists():  # drop duplicated lines
                keep = [l for l in log_path.read_text(encoding="utf-8").splitlines()
                        if l.strip() and json.loads(l)["step"] <= start_step]
                log_path.write_text("".join(x + "\n" for x in keep), encoding="utf-8")
        else:
            log("[train] no checkpoint found; starting from scratch")
    elif log_path.exists():
        log_path.unlink()

    use_cuda = str(device).startswith("cuda")
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_cuda else contextlib.nullcontext
    clip = float(tcfg["max_grad_norm"])
    end_step = min(total_steps, max_steps) if max_steps else total_steps
    student.train()
    if teacher is not None:
        teacher.eval()
    t_start = time.perf_counter()
    peak = 0.0
    for step in range(start_step, end_step):
        t0 = time.perf_counter()
        epoch, bi = divmod(step, spe)
        idx = orders[epoch][bi * eb:(bi + 1) * eb]
        batch_ex = [examples[i] for i in idx]
        n_pred_total, n_ans_total = count_tokens(batch_ex)
        n_tokens = sum(len(e["input_ids"]) for e in batch_ex)
        acc = {"ce": 0.0, "kl": 0.0, "feat": 0.0, "loss": 0.0}
        for k in range(0, len(batch_ex), mb):
            batch = {kk: v.to(device) for kk, v in collate(batch_ex[k:k + mb], pad_id).items()}
            with autocast():
                m = micro_losses(student, teacher, batch, spec, projector)
                loss = combine_losses(m, spec, n_pred_total, n_ans_total)
            loss.backward()
            acc["loss"] += float(loss.detach())
            acc["ce"] += float(m["ce_sum"].detach()) / n_pred_total
            if spec.kd:
                acc["kl"] += float(m["kl_sum"].detach()) / n_pred_total
            if spec.hidden:
                acc["feat"] += float(m["feat_sum"].detach()) / max(1, n_ans_total)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, clip))
        lr_used = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        if use_cuda:
            peak = torch.cuda.max_memory_allocated() / 1e9
        rec = {"step": step + 1, "epoch": epoch, "ce": acc["ce"], "kl": acc["kl"] if spec.kd else None,
               "feat": acc["feat"] if spec.hidden else None, "loss": acc["loss"],
               "lr": lr_used, "grad_norm": gnorm, "n_pred": n_pred_total,
               "tokens": n_tokens, "tokens_per_s": n_tokens / dt, "step_time_s": dt,
               "peak_mem_gb": peak if use_cuda else None}
        write_jsonl_line(log_path, rec)
        if (step + 1) % max(1, spe // 10) == 0 or step + 1 == end_step:
            log(f"[train] step {step + 1}/{total_steps} ep={epoch} loss={acc['loss']:.4f} ce={acc['ce']:.4f}"
                f"{' kl=%.4f' % acc['kl'] if spec.kd else ''}{' feat=%.4f' % acc['feat'] if spec.hidden else ''}"
                f" lr={rec['lr']:.2e} gn={gnorm:.2f} {n_tokens / dt:.0f} tok/s"
                f"{' mem=%.1fGB' % peak if use_cuda else ''}")
        if not math.isfinite(acc["loss"]):
            log("[train][WARN] non-finite loss (NaN/inf)")
        done = step + 1
        if done % spe == 0 or (ckpt_every and done % ckpt_every == 0):
            save_checkpoint(out_dir / "checkpoints" / f"step{done}", student, projector, opt, sched, done)

    student.save_pretrained(out_dir / "adapter")
    if projector is not None:
        torch.save(projector.state_dict(), out_dir / "projector.pt")
    recs = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    tail = recs[-10:]
    summary = {
        "steps_done": len(recs), "total_steps": total_steps, "steps_per_epoch": spe, "epochs": epochs,
        "effective_batch": eb, "micro_batch": mb, "warmup_steps": warmup, "lr": lr, "seed": seed,
        "n_examples": n, "n_trainable_lora": n_lora,
        "n_projector_params": sum(p.numel() for p in projector.parameters()) if projector else 0,
        "data_order_sha256": orders_sha256(orders),
        "final": {k: (sum(r[k] for r in tail) / len(tail) if tail and tail[0][k] is not None else None)
                  for k in ("loss", "ce", "kl", "feat")},
        "any_nonfinite": any(not math.isfinite(r["loss"]) for r in recs),
        "wall_time_s": time.perf_counter() - t_start, "peak_mem_gb": peak if use_cuda else None,
        "mean_step_time_s": (sum(r["step_time_s"] for r in recs[2:]) / len(recs[2:])) if len(recs) > 2 else None,
        "mean_tokens_per_s": (sum(r["tokens_per_s"] for r in recs[2:]) / len(recs[2:])) if len(recs) > 2 else None,
        "completed": len(recs) >= total_steps,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
