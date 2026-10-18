"""Distillation training tests: small fake Qwen2 models + fake WordLevel tokenizer, CPU."""

import json
import math
import random

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from kdq.distill import (
    IGNORE, HiddenProjector, LossSpec, answer_positions, build_examples, collate, combine_losses,
    epoch_orders, hidden_loss_sum, kd_kl_sum, micro_losses, orders_sha256, prediction_positions,
    steps_per_epoch, tokenize_example, train_run,
)
from kdq.prompts import format_question, format_target
from kdq.utils import adapter_tag, lr_tag

PAIRS = [[1, 2], [2, 4]]          # (student 2 layers, teacher 4 layers): analogue of 7/14/21/28 <-> 12/24/36/48
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TCFG = {"effective_batch": 4, "micro_batch": 2, "warmup_ratio": 0.03, "weight_decay": 0.0,
        "max_grad_norm": 1.0, "hidden": {"student_dim": 32, "teacher_dim": 64, "projection": "shared"}}


def fake_rows(n=24, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        rows.append({"question": f"Tom has {a} apples and gets {b} more . How many ?",
                     "answer": f"{a} + {b} = <<{a}+{b}={a + b}>>{a + b}\n#### {a + b}"})
    return rows


def make_tokenizer(rows):
    words = set()
    for r in rows:
        words.update((format_question(r["question"]) + format_target(r["answer"])).split())
    vocab = {"<|endoftext|>": 0, "<|im_end|>": 1, "[UNK]": 2}
    for w in sorted(words):
        vocab[w] = len(vocab)
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|endoftext|>",
                                   pad_token="<|endoftext|>", unk_token="[UNK]")


def make_models(vocab_total, seed=0):
    torch.manual_seed(seed)
    common = dict(vocab_size=vocab_total, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=256, tie_word_embeddings=False, use_cache=False)
    student = Qwen2ForCausalLM(Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, **common))
    teacher = Qwen2ForCausalLM(Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4, **common))
    teacher.requires_grad_(False)
    teacher.eval()
    return student, teacher


def wrap_lora(student, grad_ckpt=True):
    if grad_ckpt:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.enable_input_require_grads()
    return get_peft_model(student, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                              target_modules=TARGETS, task_type="CAUSAL_LM"))


def spec_for(cfg, cut):
    return LossSpec(kd=cfg != "S1", hidden=cfg == "S3", tau=2.0, alpha=0.5, beta=1.0, vocab_cut=cut, pairs=PAIRS)


@pytest.fixture(scope="module")
def setup():
    rows = fake_rows()
    tok = make_tokenizer(rows)
    cut = len(tok)
    examples, stats = build_examples(rows, tok, max_len=64, eos_id=tok.eos_token_id)
    return {"rows": rows, "tok": tok, "cut": cut, "vocab_total": cut + 8, "examples": examples, "stats": stats}


# --- data and mask ------------------------------------------------------------

def test_tokenize_masks_prompt_and_appends_eos(setup):
    tok = setup["tok"]
    r = setup["rows"][0]
    e = tokenize_example(tok, r["question"], r["answer"], 64, tok.eos_token_id)
    n_p = e["n_prompt"]
    assert e["labels"][:n_p] == [IGNORE] * n_p and e["labels"][n_p:] == e["input_ids"][n_p:]
    assert e["input_ids"][-1] == tok.eos_token_id and not e["truncated"] and not e["prefix_mismatch"]
    assert e["input_ids"][:n_p] == tok(format_question(r["question"]), add_special_tokens=False)["input_ids"]
    assert e["input_ids"][n_p:-1] == tok(format_target(r["answer"]), add_special_tokens=False)["input_ids"]
    assert "<<" not in tok.decode(e["input_ids"][n_p:-1])  # calculator annotations removed
    assert setup["stats"]["n_truncated"] == 0 and setup["stats"]["n_prefix_mismatch"] == 0


def test_truncation_reported(setup):
    tok = setup["tok"]
    r = setup["rows"][0]
    e = tokenize_example(tok, r["question"], r["answer"], 8, tok.eos_token_id)
    assert e["truncated"] and len(e["input_ids"]) == 8 and len(e["labels"]) == 8


def test_positions_exclude_prompt_and_padding(setup):
    exs = setup["examples"][:3]
    batch = collate(exs, pad_id=0)
    b, t, targets = prediction_positions(batch["labels"])
    n_ans = sum(sum(l != IGNORE for l in e["labels"]) for e in exs)
    assert targets.numel() == n_ans == answer_positions(batch["labels"])[0].numel()
    for bi, ti, tg in zip(b.tolist(), t.tolist(), targets.tolist()):
        e = exs[bi]
        assert ti + 1 >= e["n_prompt"] and ti + 1 < len(e["input_ids"])      # outside prompt and padding
        assert batch["attention_mask"][bi, ti + 1] == 1 and e["input_ids"][ti + 1] == tg
    # The first prediction position of each example is the last prompt token (Answer:)
    firsts = {bi: ti for bi, ti in reversed(list(zip(b.tolist(), t.tolist())))}
    assert all(firsts[i] == exs[i]["n_prompt"] - 1 for i in range(3))


def test_padding_tokens_get_no_loss(setup):
    """Padding does not change the loss: an example alone and in a padded batch gives the same CE."""
    student, teacher = make_models(setup["vocab_total"])
    student = wrap_lora(student, grad_ckpt=False).eval()
    spec = spec_for("S1", setup["cut"])
    exs = setup["examples"]
    short = min(exs, key=lambda e: len(e["input_ids"]))
    long_ = max(exs, key=lambda e: len(e["input_ids"]))
    alone = micro_losses(student, None, collate([short], 0), spec, None)
    padded = micro_losses(student, None, collate([short, long_], 0), spec, None)
    only_long = micro_losses(student, None, collate([long_], 0), spec, None)
    assert torch.allclose(alone["ce_sum"] + only_long["ce_sum"], padded["ce_sum"], atol=1e-4)
    assert padded["n_pred"] == alone["n_pred"] + only_long["n_pred"]


# --- KL -----------------------------------------------------------------------

def test_kl_zero_for_identical():
    x = torch.randn(5, 7)
    assert float(kd_kl_sum(x, x.clone(), 2.0)) == pytest.approx(0.0, abs=1e-6)


def test_kl_matches_manual_and_tau_scale():
    s = torch.tensor([[0.5, -1.0, 2.0], [1.0, 1.0, 0.0]])
    t = torch.tensor([[1.0, 0.0, -1.0], [0.3, 0.2, 0.1]])
    for tau in (1.0, 2.0):
        ref = 0.0
        for i in range(2):
            ps = [math.exp(v / tau) for v in s[i].tolist()]
            pt = [math.exp(v / tau) for v in t[i].tolist()]
            ps = [p / sum(ps) for p in ps]
            pt = [p / sum(pt) for p in pt]
            ref += sum(pt[j] * math.log(pt[j] / ps[j]) for j in range(3))
        assert float(kd_kl_sum(s, t, tau)) == pytest.approx(tau * tau * ref, rel=1e-5)
    assert float(kd_kl_sum(s, t, 2.0)) != pytest.approx(float(kd_kl_sum(s, t, 1.0)))
    # forward direction: KL(p_T || p_S), not symmetric
    assert float(kd_kl_sum(s, t, 1.0)) != pytest.approx(float(kd_kl_sum(t, s, 1.0)))


def test_vocab_cut_applied_to_both_sides(setup):
    cut, V = setup["cut"], setup["vocab_total"]
    student, teacher = make_models(V)
    student = wrap_lora(student, grad_ckpt=False).eval()
    batch = collate(setup["examples"][:2], 0)
    spec = spec_for("S2", cut)
    before = micro_losses(student, teacher, batch, spec, None)
    with torch.no_grad():  # corrupt the padding rows (>= cut): no effect if truncation is applied
        student.get_output_embeddings().weight[cut:] = 50.0
        teacher.get_output_embeddings().weight[cut:] = -50.0
    after = micro_losses(student, teacher, batch, spec, None)
    assert torch.allclose(before["ce_sum"], after["ce_sum"]) and torch.allclose(before["kl_sum"], after["kl_sum"])
    nocut = micro_losses(student, teacher, batch, spec_for("S2", V), None)   # without truncation: changes
    assert not torch.allclose(before["kl_sum"], nocut["kl_sum"])


# --- hidden states and layer matching ------------------------------------------

def test_hidden_states_indexing_and_lm_head(setup):
    student, teacher = make_models(setup["vocab_total"])
    batch = collate(setup["examples"][:2], 0)
    for model, n_layers in ((student, 2), (teacher, 4)):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
        hs = out.hidden_states
        assert len(hs) == n_layers + 1
        emb = model.get_input_embeddings()(batch["input_ids"])
        assert torch.allclose(hs[0], emb)                                   # index 0 = embedding output
        logits = model.get_output_embeddings()(hs[-1])
        assert torch.allclose(logits, out.logits, atol=1e-5)                # hs[-1] = normalised final state = lm_head input
        assert hs[-1].shape[-1] == model.config.hidden_size
    assert max(p[0] for p in PAIRS) == student.config.num_hidden_layers
    assert max(p[1] for p in PAIRS) == teacher.config.num_hidden_layers


def test_projector_shapes_and_hidden_loss_manual():
    proj = HiddenProjector(32, 64, len(PAIRS), shared=True)
    assert len(proj.proj) == 1 and proj.proj[0].weight.shape == (64, 32) and proj.proj[0].weight.dtype == torch.float32
    assert len(HiddenProjector(32, 64, len(PAIRS), shared=False).proj) == len(PAIRS)
    torch.manual_seed(0)
    s_hs = [torch.randn(2, 5, 32) for _ in range(3)]
    t_hs = [torch.randn(2, 5, 64) for _ in range(5)]
    b, t = torch.tensor([0, 1, 1]), torch.tensor([1, 2, 4])
    got = hidden_loss_sum(s_hs, t_hs, PAIRS, proj, b, t)
    ref = 0.0
    for (l, m) in PAIRS:
        for bi, ti in zip(b.tolist(), t.tolist()):
            ps = F.layer_norm(proj.proj[0](s_hs[l][bi, ti]), (64,))
            pt = F.layer_norm(t_hs[m][bi, ti], (64,))
            ref += float(((ps - pt) ** 2).mean())
    assert float(got) == pytest.approx(ref / len(PAIRS), rel=1e-5)


def test_combine_losses_formulas():
    m = {"ce_sum": torch.tensor(8.0), "kl_sum": torch.tensor(4.0), "feat_sum": torch.tensor(6.0)}
    s1 = combine_losses(m, spec_for("S1", 10), n_pred_total=4, n_ans_total=3)
    s2 = combine_losses(m, spec_for("S2", 10), 4, 3)
    s3 = combine_losses(m, spec_for("S3", 10), 4, 3)
    assert float(s1) == pytest.approx(2.0)
    assert float(s2) == pytest.approx((0.5 * 8 + 0.5 * 4) / 4)
    assert float(s3) == pytest.approx(float(s2) + 1.0 * 6 / 3)


# --- data order ----------------------------------------------------------------

def test_epoch_orders_depend_only_on_seed():
    a, b = epoch_orders(7173, 2, 42), epoch_orders(7173, 2, 42)
    assert a == b and sorted(a[0]) == list(range(7173)) and a[0] != a[1]
    assert epoch_orders(7173, 2, 1234) != a
    assert steps_per_epoch(7173, 16) == 449


# --- end to end --------------------------------------------------------------

def _run(setup, cfg, out, seed=1, **kw):
    torch.manual_seed(seed)
    student, teacher = make_models(setup["vocab_total"], seed=seed)
    student = wrap_lora(student)
    return train_run(student=student, teacher=teacher if cfg != "S1" else None, examples=setup["examples"],
                     spec=spec_for(cfg, setup["cut"]), tcfg=TCFG, seed=42, lr=5e-3, epochs=2, out_dir=out,
                     device="cpu", pad_id=0, log=lambda m: None, **kw)


@pytest.mark.slow
@pytest.mark.parametrize("cfg", ["S1", "S2", "S3"])
def test_end_to_end_loss_decreases(setup, tmp_path, cfg):
    out = tmp_path / cfg
    s = _run(setup, cfg, out)
    recs = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert s["completed"] and s["steps_done"] == 12 == s["total_steps"] and len(recs) == 12
    first, last = [r["loss"] for r in recs[:3]], [r["loss"] for r in recs[-3:]]
    assert sum(last) / 3 < sum(first) / 3
    assert all(math.isfinite(r["loss"]) and r["grad_norm"] >= 0 and r["lr"] >= 0 for r in recs)
    assert recs[0]["lr"] == pytest.approx(5e-3) and recs[-1]["lr"] > 0   # lr used at the step (cosine > 0 at the last step)
    assert (recs[0]["kl"] is None) == (cfg == "S1") and (recs[0]["feat"] is None) == (cfg != "S3")
    assert (out / "adapter" / "adapter_model.safetensors").exists()
    assert (out / "projector.pt").exists() == (cfg == "S3")
    assert (out / "checkpoints" / "step6").exists() and (out / "checkpoints" / "step12").exists()
    assert s["data_order_sha256"] == orders_sha256(epoch_orders(24, 2, 42))


@pytest.mark.slow
def test_same_seed_same_data_order_across_configs(setup, tmp_path):
    shas = {cfg: _run(setup, cfg, tmp_path / cfg, max_steps=1)["data_order_sha256"] for cfg in ("S1", "S2", "S3")}
    assert len(set(shas.values())) == 1


@pytest.mark.slow
def test_checkpoint_resume_matches_uninterrupted(setup, tmp_path):
    a = _run(setup, "S3", tmp_path / "ab", max_steps=6)        # stops at the end of epoch 1 (checkpoint step6)
    assert a["steps_done"] == 6 and not a["completed"]
    b = _run(setup, "S3", tmp_path / "ab", resume=True)        # new model objects + checkpoint
    c = _run(setup, "S3", tmp_path / "c")                      # uninterrupted
    recs = [json.loads(l)["step"] for l in (tmp_path / "ab" / "train_log.jsonl").read_text().splitlines()]
    assert recs == list(range(1, 13)) and b["completed"]
    wb = load_file(str(tmp_path / "ab" / "adapter" / "adapter_model.safetensors"))
    wc = load_file(str(tmp_path / "c" / "adapter" / "adapter_model.safetensors"))
    assert wb.keys() == wc.keys() and all(torch.allclose(wb[k], wc[k], atol=1e-5) for k in wb)
    pb, pc = torch.load(tmp_path / "ab" / "projector.pt"), torch.load(tmp_path / "c" / "projector.pt")
    assert all(torch.allclose(pb[k], pc[k], atol=1e-5) for k in pb)


# --- helpers ---------------------------------------------------------------------

def test_tags():
    assert lr_tag(1e-4) == "1e-4" and lr_tag(2e-4) == "2e-4" and lr_tag(5e-5) == "5e-5" and lr_tag(1.5e-4) == "1.5e-4"
    assert adapter_tag("/workspace/runs/S2_seed42/adapter") == "S2_seed42"
    assert adapter_tag("/workspace/runs/S2_seed42_lr1e-4/adapter/") == "S2_seed42_lr1e-4"
    assert adapter_tag("/x/my_adapter") == "my_adapter"
