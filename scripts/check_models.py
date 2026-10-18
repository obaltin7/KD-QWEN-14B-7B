"""Model/tokenizer compatibility check (does NOT load weights; offline).

Checks:
  - Each model: layers, hidden size, attention heads, config.vocab_size, len(tokenizer), license.
  - Teacher <-> each student: get_vocab() equality, added/special tokens,
    eos/pad/bos, token IDs on the first 20 GSM8K train examples.
  - Whether the layer mapping (base.yaml) is consistent with the model dimensions.

Exit code: 0 = critical checks passed, 1 = critical mismatch, 2 = missing from cache.
Critical: vocab, added tokens, token IDs, len(tokenizer), layers/dimensions.
Warning (critical with --strict): eos/pad/bos differences. The Qwen2.5 base model's eos is
"<|endoftext|>", the Instruct models' is "<|im_end|>"; this does not affect the KL
(same vocabulary) but must be accounted for in generation/stopping settings.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from kdq.config import load_config, repo_path  # noqa: E402
from kdq.data import load_gsm8k, save_json  # noqa: E402
from kdq.utils import run_metadata  # noqa: E402

N_TEXTS = 20


def read_license(repo_id: str) -> dict:
    """Read the license from the YAML front matter of the cached README.md."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id, "README.md", local_files_only=True)
    except Exception as e:  # report if not cached; not counted as an error
        return {"license": None, "note": f"README not in cache ({type(e).__name__})"}
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {"license": None, "note": "README has no front matter"}
    front = yaml.safe_load(text.split("---", 2)[1]) or {}
    return {k: front.get(k) for k in ("license", "license_name", "license_link")}


def model_summary(repo_id: str, cfg, tok) -> dict:
    tc = getattr(cfg, "text_config", cfg)
    return {
        "num_hidden_layers": tc.num_hidden_layers,
        "hidden_size": tc.hidden_size,
        "num_attention_heads": tc.num_attention_heads,
        "num_key_value_heads": getattr(tc, "num_key_value_heads", None),
        "intermediate_size": getattr(tc, "intermediate_size", None),
        "config_vocab_size": tc.vocab_size,
        "len_tokenizer": len(tok),
        "tokenizer_vocab_size": tok.vocab_size,
        "vocab_pad_rows": tc.vocab_size - len(tok),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
        "bos": [tok.bos_token, tok.bos_token_id],
        "eos": [tok.eos_token, tok.eos_token_id],
        "pad": [tok.pad_token, tok.pad_token_id],
        "license": read_license(repo_id),
    }


def compare_tokenizers(t_tok, s_tok, texts: list[str]) -> dict:
    t_added = {i: str(a) for i, a in t_tok.added_tokens_decoder.items()}
    s_added = {i: str(a) for i, a in s_tok.added_tokens_decoder.items()}
    id_mismatch = []
    for k, txt in enumerate(texts):
        a = t_tok(txt, add_special_tokens=False)["input_ids"]
        b = s_tok(txt, add_special_tokens=False)["input_ids"]
        if a != b:
            id_mismatch.append(k)
    t_vocab, s_vocab = t_tok.get_vocab(), s_tok.get_vocab()
    critical = {
        "vocab_identical": t_vocab == s_vocab,
        "len_tokenizer_equal": len(t_tok) == len(s_tok),
        "added_tokens_identical": t_added == s_added,
        "special_tokens_identical": sorted(t_tok.all_special_tokens) == sorted(s_tok.all_special_tokens),
        "token_ids_identical_on_texts": not id_mismatch,
    }
    soft = {
        "eos_equal": (t_tok.eos_token, t_tok.eos_token_id) == (s_tok.eos_token, s_tok.eos_token_id),
        "pad_equal": (t_tok.pad_token, t_tok.pad_token_id) == (s_tok.pad_token, s_tok.pad_token_id),
        "bos_equal": (t_tok.bos_token, t_tok.bos_token_id) == (s_tok.bos_token, s_tok.bos_token_id),
    }
    detail = {
        "vocab_sizes": [len(t_vocab), len(s_vocab)],
        "vocab_symmetric_diff": len(set(t_vocab.items()) ^ set(s_vocab.items())),
        "n_texts": len(texts),
        "mismatched_text_indices": id_mismatch,
    }
    return {"critical": critical, "soft": soft, "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/day1/model_check.json")
    ap.add_argument("--strict", action="store_true", help="also treat eos/pad/bos differences as errors")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    cfg = load_config(args.config)
    roles = {k: cfg["models"][k] for k in ("teacher", "student_base", "student_instruct")}

    try:
        ds = load_gsm8k(cfg)["train"]
    except Exception as e:
        print(f"[ERROR] GSM8K not in cache: {e}\n  -> run step 5 of scripts/setup_pod.sh.")
        return 2
    texts = [ds[i]["question"] + "\n" + ds[i]["answer"] for i in range(N_TEXTS)]

    toks, models = {}, {}
    for role, repo in roles.items():
        try:
            mcfg = AutoConfig.from_pretrained(repo)
            toks[role] = AutoTokenizer.from_pretrained(repo)
        except OSError as e:
            print(f"[ERROR] {repo} not found in cache: {e}")
            return 2
        models[role] = {"repo_id": repo, **model_summary(repo, mcfg, toks[role])}

    pairs = {
        f"teacher__{s}": compare_tokenizers(toks["teacher"], toks[s], texts)
        for s in ("student_base", "student_instruct")
    }

    # Layer-mapping consistency (for S3)
    h = cfg["train"]["hidden"]
    h = {**h, "student_layers": [p[0] for p in h["pairs"]], "teacher_layers": [p[1] for p in h["pairs"]]}
    h = {**h, "student_layers": [p[0] for p in h["pairs"]], "teacher_layers": [p[1] for p in h["pairs"]]}
    t, s = models["teacher"], models["student_base"]
    layer_check = {
        "student_layers_in_range": max(h["student_layers"]) <= s["num_hidden_layers"],
        "teacher_layers_in_range": max(h["teacher_layers"]) <= t["num_hidden_layers"],
        "student_dim_matches": h["student_dim"] == s["hidden_size"],
        "teacher_dim_matches": h["teacher_dim"] == t["hidden_size"],
        "config_vocab_size_equal": t["config_vocab_size"] == s["config_vocab_size"],
    }

    failures, warnings = [], []
    for name, p in pairs.items():
        failures += [f"{name}.{k}" for k, v in p["critical"].items() if not v]
        (failures if args.strict else warnings).extend(
            f"{name}.{k}" for k, v in p["soft"].items() if not v
        )
    failures += [f"layer_check.{k}" for k, v in layer_check.items() if not v]

    kl_dim = {
        "len_tokenizer": t["len_tokenizer"],
        "config_vocab_size": t["config_vocab_size"],
        "pad_rows": t["vocab_pad_rows"],
        "note": "The lm_head output has config_vocab_size columns; the last pad_rows rows "
                "correspond to no token. Whether the KL is computed over the first "
                "len_tokenizer logits or the full dimension is decided after the baseline stage.",
    }

    report = {
        "models": models,
        "pairs": pairs,
        "layer_check": layer_check,
        "kl_vocab_dim": kl_dim,
        "critical_ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "meta": run_metadata(None, script="check_models.py", strict=args.strict),
    }
    out = repo_path(args.out)
    save_json(report, out)

    # Human-readable summary
    print(f"{'role':18s} {'layers':>6s} {'hidden':>6s} {'heads':>6s} {'kv':>4s} "
          f"{'cfg_vocab':>9s} {'len_tok':>8s} {'diff':>5s}  license")
    for role, m in models.items():
        print(f"{role:18s} {m['num_hidden_layers']:6d} {m['hidden_size']:6d} "
              f"{m['num_attention_heads']:6d} {m['num_key_value_heads'] or 0:4d} "
              f"{m['config_vocab_size']:9d} {m['len_tokenizer']:8d} {m['vocab_pad_rows']:5d}  "
              f"{m['license'].get('license')}")
        print(f"{'':18s} eos={m['eos']} pad={m['pad']}")
    for name, p in pairs.items():
        flags = " ".join(f"{k}={'OK' if v else 'DIFF'}" for k, v in {**p['critical'], **p['soft']}.items())
        print(f"{name}: {flags}")
    print("layer_check:", " ".join(f"{k}={'OK' if v else 'DIFF'}" for k, v in layer_check.items()))
    for w in warnings:
        print(f"[WARNING] {w}")
    for f_ in failures:
        print(f"[ERROR] {f_}")
    print(f"-> {out}  |  critical: {'PASSED' if not failures else 'FAILED'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
