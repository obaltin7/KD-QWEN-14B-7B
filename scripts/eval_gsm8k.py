"""GSM8K evaluation (vllm or hf engine), greedy decoding.

Outputs (in order):
  <out>.raw.jsonl + <out>.raw.meta.json  raw generations, written BEFORE scoring
  <out>.jsonl                            per-question scores (answer extraction v2: strict + flexible)
  <out>.summary.json                     summary + metadata; written last, its presence marks
                                         the run as complete.

Example:
  python scripts/eval_gsm8k.py --model teacher --shots 4
  python scripts/eval_gsm8k.py --model student_base --shots 0 --limit 50 --engine hf
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kdq.answer import extract_gold  # noqa: E402
from kdq.config import load_config, repo_path  # noqa: E402
from kdq.data import load_gsm8k, load_split, save_json  # noqa: E402
from kdq.evaluation import write_jsonl, write_scored  # noqa: E402
from kdq.prompts import build_prompt, truncate_at_stop  # noqa: E402
from kdq.utils import adapter_tag, run_metadata, set_seed  # noqa: E402


def default_out(cfg, a, split: str) -> Path:
    name = f"{a.model}_{a.shots}shot_{a.engine}"
    if a.limit:
        name += f"_n{a.limit}"
    if a.adapter:
        name += f"_{adapter_tag(a.adapter)}"
    if split != cfg["eval"]["split"]:
        name += f"_{split}"
    return repo_path(cfg["eval"]["out_dir"]) / name


def stop_token_ids(repo: str, names) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    ids = [tok.convert_tokens_to_ids(t) for t in names]
    return [i for i in ids if i is not None and i != tok.unk_token_id]


def gen_vllm(repo, prompts, cfg, adapter, stop_ids):
    from vllm import LLM, SamplingParams

    e, v = cfg["eval"], cfg["eval"]["vllm"]
    lora_kw = {}
    if adapter:
        lora_kw = {"enable_lora": True, "max_lora_rank": max(16, int(cfg["train"]["lora"]["r"]))}
    t0 = time.perf_counter()
    llm = LLM(
        model=repo,
        dtype="bfloat16",
        seed=e["seed"],
        gpu_memory_utilization=v["gpu_memory_utilization"],
        max_model_len=v["max_model_len"],
        enable_prefix_caching=v["enable_prefix_caching"],
        # The model's generation_config (e.g. Instruct: repetition_penalty=1.05,
        # temperature=0.7) is ignored; identical greedy settings for all models.
        generation_config="vllm",
        **lora_kw,
    )
    load_s = time.perf_counter() - t0
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=e["max_new_tokens"],
        stop=e["stop_sequences"],
        stop_token_ids=stop_ids,
        repetition_penalty=1.0,
    )
    lora_req = None
    if adapter:
        from vllm.lora.request import LoRARequest

        lora_req = LoRARequest("adapter", 1, str(adapter))
    t1 = time.perf_counter()
    outs = llm.generate(prompts, sp, lora_request=lora_req)
    gen_s = time.perf_counter() - t1
    res = [
        {"text": o.outputs[0].text, "n_gen_tokens": len(o.outputs[0].token_ids),
         "finish_reason": o.outputs[0].finish_reason}
        for o in outs
    ]
    return res, load_s, gen_s


def gen_hf(repo, prompts, cfg, adapter, stop_ids):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    e = cfg["eval"]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    load_s = time.perf_counter() - t0

    bs = e["hf"]["batch_size"]
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))  # similar lengths in the same batch
    res: list[dict | None] = [None] * len(prompts)
    t1 = time.perf_counter()
    for start in range(0, len(order), bs):
        batch = order[start:start + bs]
        enc = tok([prompts[i] for i in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.0,
                max_new_tokens=e["max_new_tokens"],
                eos_token_id=stop_ids,
                pad_token_id=tok.pad_token_id,
                stop_strings=e["stop_sequences"],
                tokenizer=tok,
            )
        new = out[:, enc["input_ids"].shape[1]:].tolist()
        for ids, i in zip(new, batch):
            n, finish = len(ids), "length"
            for j, t in enumerate(ids):
                if t in stop_ids:
                    n, finish = j, "stop"
                    break
            text = tok.decode(ids[:n], skip_special_tokens=True)
            if truncate_at_stop(text, e["stop_sequences"]) != text:
                finish = "stop"
            res[i] = {"text": text, "n_gen_tokens": n, "finish_reason": finish}
        print(f"  hf {min(start + bs, len(order))}/{len(order)}", flush=True)
    gen_s = time.perf_counter() - t1
    return res, load_s, gen_s


def main() -> int:
    ap = argparse.ArgumentParser(description="GSM8K evaluation")
    ap.add_argument("--model", required=True, help="config key: teacher | student_base | student_instruct")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory (optional)")
    ap.add_argument("--split", default=None, help="default: config eval.split (test)")
    ap.add_argument("--out", default=None, help="output prefix (.jsonl / .summary.json are appended)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    e = cfg["eval"]
    repo = cfg["models"][a.model]
    split = a.split or e["split"]
    out = Path(a.out) if a.out else default_out(cfg, a, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    set_seed(e["seed"])

    ds = load_gsm8k(cfg)
    if split == "val":
        rows = [ds["train"][i] for i in load_split(repo_path(cfg["paths"]["splits_dir"]), "val")]
    else:
        rows = list(ds[split])
    if a.limit:
        rows = rows[: a.limit]
    shots = []
    if a.shots:
        k = cfg["split"]["fewshot_k"]
        if a.shots > k:
            sys.exit(f"[ERROR] --shots {a.shots} > fewshot_k {k}")
        idx = load_split(repo_path(cfg["paths"]["splits_dir"]), f"fewshot_{k}shot")[: a.shots]
        shots = [ds["train"][i] for i in idx]
    prompts = [build_prompt(r["question"], shots) for r in rows]
    golds = [extract_gold(r["answer"]) for r in rows]

    stop_ids = stop_token_ids(repo, e["stop_tokens"])
    print(f"[eval] {a.model} ({repo}) | {split} n={len(rows)} | {a.shots}-shot | {a.engine}"
          f"{' | adapter=' + a.adapter if a.adapter else ''}", flush=True)
    gen = gen_vllm if a.engine == "vllm" else gen_hf
    gens, load_s, gen_s = gen(repo, prompts, cfg, a.adapter, stop_ids)

    raw = [{"idx": i, "gold": gold, "generation": truncate_at_stop(g["text"], e["stop_sequences"]),
            "n_gen_tokens": g["n_gen_tokens"], "finish_reason": g["finish_reason"]}
           for i, (g, gold) in enumerate(zip(gens, golds))]
    total_tok = sum(r["n_gen_tokens"] for r in raw)
    base = {
        "model_key": a.model, "repo_id": repo, "engine": a.engine, "shots": a.shots,
        "split": split, "limit": a.limit, "adapter": a.adapter,
        "load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
        "tokens_per_s": round(total_tok / gen_s, 1) if gen_s > 0 else None,
        "decoding": {"greedy": True, "max_new_tokens": e["max_new_tokens"],
                     "stop_sequences": e["stop_sequences"], "stop_token_ids": stop_ids,
                     "repetition_penalty": 1.0, "chat_template": False},
        "example_prompt": prompts[0] if prompts else None,
        "records": out.name + ".jsonl",
        "raw_records": out.name + ".raw.jsonl",
        "meta": run_metadata(cfg, script="eval_gsm8k.py", argv=sys.argv[1:],
                             cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
    }
    # 1) Preserve GPU work: raw generations and run info go to disk BEFORE scoring.
    #    If scoring crashes, python scripts/rescore.py scores from these files.
    write_jsonl(raw, out.with_name(out.name + ".raw.jsonl"))
    save_json(base, out.with_name(out.name + ".raw.meta.json"))
    print(f"[eval] raw generations written: {out.name}.raw.jsonl", flush=True)

    # 2) Scoring (answer extraction v2: strict + flexible); per-question errors are caught.
    s = write_scored(out, raw, base)
    print(f"[eval] flexible {s['accuracy_flexible']:.4f} {fmt_ci(s['ci95_flexible'])} | "
          f"strict {s['accuracy_strict']:.4f} {fmt_ci(s['ci95_strict'])} | n={s['n']} | "
          f"format error {s['format_error_rate']:.3f} | no answer {s['no_answer_rate']:.3f} | "
          f"truncated at 512 {s['truncated_rate']:.3f} | scoring errors {s['n_score_errors']} | "
          f"{gen_s:.0f} s, {base['tokens_per_s']} tok/s")
    return 0


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


if __name__ == "__main__":
    sys.exit(main())
