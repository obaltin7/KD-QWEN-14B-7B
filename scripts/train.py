"""LoRA distillation training: S1 (CE), S2 (CE+KD), S3 (CE+KD+hidden state). Exactly as in prereg-v1.0.

Outputs:
  <runs_dir>/<name>/adapter/           PEFT adapter (loaded via vLLM --adapter)
  <runs_dir>/<name>/projector.pt       S3 only: W (not merged into the student)
  <runs_dir>/<name>/checkpoints/stepN/ end-of-epoch checkpoints (--resume)
  <runs_dir>/<name>/train_log.jsonl, run_meta.json, summary.json
  <results_dir>/<name>/                copy of the same small files (committed)
name = <cfg>_seed<seed>[_lr<lr>] (if lr is given on the command line). Skipped if summary.json exists.

Example:
  python scripts/train.py --cfg S2 --seed 42 --lr 1e-4
  python scripts/train.py --cfg S3 --seed 42 --lr 1e-4 --max-steps 20 --out $WORKSPACE/runs/smoke_S3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kdq.config import load_config, repo_path  # noqa: E402
from kdq.data import load_gsm8k, load_split  # noqa: E402
from kdq.distill import LossSpec, build_examples, train_run  # noqa: E402
from kdq.utils import lr_tag, set_seed, write_run_metadata  # noqa: E402


def loss_spec(tcfg: dict, cfg_name: str) -> LossSpec:
    c = tcfg["configs"][cfg_name]
    return LossSpec(kd=bool(c["kd"]), hidden=bool(c["hidden"]), tau=float(tcfg["kd"]["tau"]),
                    alpha=float(tcfg["kd"]["alpha"]), beta=float(tcfg["hidden"]["beta"]),
                    vocab_cut=int(tcfg["kd"]["vocab_cut"]), pairs=[list(p) for p in tcfg["hidden"]["pairs"]])


def load_models(cfg: dict, tcfg: dict, spec: LossSpec):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    student_repo = cfg["models"][cfg["student"]]
    tok = AutoTokenizer.from_pretrained(student_repo)
    if len(tok) != spec.vocab_cut:
        sys.exit(f"[ERROR] len(tokenizer)={len(tok)} != train.kd.vocab_cut={spec.vocab_cut}")
    kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation=tcfg["attn_implementation"])
    student = AutoModelForCausalLM.from_pretrained(student_repo, **kw)
    student.config.use_cache = False
    if tcfg["gradient_checkpointing"]:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.enable_input_require_grads()
    lcfg = tcfg["lora"]
    student = get_peft_model(student, LoraConfig(
        r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]), lora_dropout=float(lcfg["dropout"]),
        target_modules=list(lcfg["target_modules"]), bias="none", task_type="CAUSAL_LM"))
    # get_peft_model default (autocast_adapter_dtype=True): LoRA weights fp32, base bf16.
    teacher = None
    if spec.kd or spec.hidden:
        teacher = AutoModelForCausalLM.from_pretrained(cfg["models"]["teacher"], **kw)
        teacher.config.use_cache = False
        teacher.requires_grad_(False)
        teacher.eval()
        h = tcfg["hidden"]
        assert teacher.config.hidden_size == h["teacher_dim"] and student.config.hidden_size == h["student_dim"]
        assert max(p[1] for p in spec.pairs) <= teacher.config.num_hidden_layers
        assert max(p[0] for p in spec.pairs) <= student.config.num_hidden_layers
    return tok, student, teacher


def main() -> int:
    ap = argparse.ArgumentParser(description="S1/S2/S3 LoRA distillation training")
    ap.add_argument("--cfg", required=True, choices=["S1", "S2", "S3"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=None, help="if omitted, configs/base.yaml train.lr (frozen)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=None, help="run directory (default <runs_dir>/<name>)")
    ap.add_argument("--results-dir", default=None, help="copy of the small outputs (default <results_dir>/<name>)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    tcfg = cfg["train"]
    if a.lr is None and tcfg.get("lr") is None:
        sys.exit("[ERROR] Learning rate is not frozen: pass --lr or set configs/base.yaml train.lr after the pilot.")
    lr = float(a.lr if a.lr is not None else tcfg["lr"])
    epochs = int(a.epochs or tcfg["epochs"])
    name = f"{a.cfg}_seed{a.seed}" + (f"_lr{lr_tag(lr)}" if a.lr is not None else "")
    out_dir = Path(a.out) if a.out else Path(tcfg["runs_dir"]) / name
    res_dir = Path(a.results_dir) if a.results_dir else repo_path(tcfg["results_dir"]) / name
    if (out_dir / "summary.json").exists() and not a.resume:
        print(f"[train] SKIP {out_dir} (summary.json exists)")
        return 0

    spec = loss_spec(tcfg, a.cfg)
    set_seed(a.seed)
    tok, student, teacher = load_models(cfg, tcfg, spec)
    eos_id = tok.convert_tokens_to_ids(tcfg["eos_token"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    ds = load_gsm8k(cfg)["train"]
    rows = [ds[i] for i in load_split(repo_path(cfg["paths"]["splits_dir"]), "train_rest")]
    examples, data_stats = build_examples(rows, tok, int(tcfg["max_seq_len"]), eos_id)
    print(f"[train] {name}: n={data_stats['n']} max_len={data_stats['max_len_tokens']} "
          f"mean={data_stats['mean_len_tokens']:.0f} truncated={data_stats['n_truncated']} "
          f"prefix_mismatch={data_stats['n_prefix_mismatch']} | lr={lr} epochs={epochs}", flush=True)
    student.print_trainable_parameters()

    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir / "run_meta.json", cfg, script="train.py", argv=sys.argv[1:], run_name=name,
                       cfg_name=a.cfg, seed=a.seed, lr=lr, epochs=epochs, data_stats=data_stats,
                       loss_spec=spec.__dict__, student=cfg["models"][cfg["student"]],
                       teacher=cfg["models"]["teacher"] if teacher is not None else None, eos_id=eos_id)
    summary = train_run(student=student, teacher=teacher, examples=examples, spec=spec, tcfg=tcfg,
                        seed=a.seed, lr=lr, epochs=epochs, out_dir=out_dir, device="cuda", pad_id=pad_id,
                        max_steps=a.max_steps, resume=a.resume, micro_batch=a.micro_batch,
                        ckpt_every=tcfg.get("ckpt_every_steps"))
    res_dir.mkdir(parents=True, exist_ok=True)
    for f in ("train_log.jsonl", "run_meta.json", "summary.json"):
        shutil.copy2(out_dir / f, res_dir / f)
    print(f"[train] done: {json.dumps(summary['final'])} | {summary['wall_time_s']:.0f} s | "
          f"adapter: {out_dir / 'adapter'} | log: {res_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
