"""Analysis: pre-registered tests (prereg-v1.0 §6), secondary/exploratory analyses, figures, tables.

Runs locally (no GPU/model); reads only the records under results/. Everything is regenerated from
scratch and the output is deterministic (bootstrap seed 42, no timestamps are written).

Figures and tables are written in two languages from the same numbers; all language-dependent text
lives in LABELS.

Outputs: results/analysis/{findings.md, analysis.json}
         results/analysis/tables-{en,tr}/*.csv|md
         results/analysis/figures-{en,tr}/*.png|pdf|txt

  python scripts/analyze.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kdq.analysis import (  # noqa: E402
    ALPHA, BOOT_SEED, INCONSISTENT, N_BOOT, NOT_SUPPORTED, OPPOSITE, PARTIAL, SUPPORTED, decide,
    gap_closed, holm, mcnemar_paired, paired_bootstrap_acc, paired_bootstrap_ppl, ppl,
)
from kdq.config import load_config  # noqa: E402
from kdq.stats import wilson_ci  # noqa: E402

RES = ROOT / "results"
OUT = RES / "analysis"
LANGS = ("en", "tr")
CFGS = ("S1", "S2", "S3")
SEEDS = (42, 1234)
RUNS = [f"{c}_seed{s}" for s in SEEDS for c in CFGS]
MODELS = ["T", "S0", *RUNS]
SHOTS = (0, 4)
GEN_TASKS = {"mmlu": "correct", "hellaswag": "correct_norm", "arc_c": "correct_norm"}  # pre-registered metric
GEN_LABEL = {"mmlu": "MMLU", "hellaswag": "HellaSwag", "arc_c": "ARC-C", "wikitext2": "WikiText-2"}
GEN_METRIC = {"mmlu": "acc", "hellaswag": "acc_norm", "arc_c": "acc_norm"}
EXPECT_N = {"gsm8k": 1319, "mmlu": 1000, "hellaswag": 1000, "arc_c": 1172}
EXPLORATORY = "exploratory, not pre-registered"

# Language-dependent text for tables and figures. Keys are identical in both languages.
LABELS = {
    "en": {
        "decimal": ".", "thousands": ",", "ci_sep": ", ",
        "seed": "seed", "yes": "yes", "no": "no",
        "favourable": "favourable", "unfavourable": "unfavourable", "descriptive": " (descriptive)",
        "exploratory": "Exploratory, not pre-registered",
        "decisions": {SUPPORTED: SUPPORTED, PARTIAL: PARTIAL, NOT_SUPPORTED: NOT_SUPPORTED,
                      OPPOSITE: OPPOSITE, INCONSISTENT: INCONSISTENT},
        "tables": {"T1": "T1_conditions", "T2": "T2_main_results", "T3": "T3_H1_H2", "T4": "T4_H3",
                   "T5": "T5_gap_closed", "T6": "T6_secondary_metrics", "T7": "T7_vs_S0_exploratory",
                   "T8": "T8_format_exploratory"},
        "figures": {"F1": "F1_gsm8k_primary", "F2": "F2_learning_forgetting", "F3": "F3_general_benchmark_change",
                    "F4": "F4_gap_closed", "F5": "F5_training_curves", "F6": "F6_format_usage_exploratory"},
        "item_value": ["Item", "Value"],
        "teacher": "Teacher (T)", "teacher_v": "{m} (frozen, bf16)",
        "student": "Student (S)", "student_v": "{m} + LoRA",
        "train_data": "Training data", "train_data_v": "GSM8K train_rest, n = {n}",
        "longest": "Longest example / truncated", "longest_v": "{a} tokens / {b}",
        "test": "Test", "test_v": "GSM8K test, n = 1319",
        "prompt": "Prompt", "prompt_v": "Question: {question}\\nAnswer: — no chat template",
        "lora_v": "r = {r}, alpha = {a}, dropout = {d}, targets: {t}",
        "lora_params": "Trainable LoRA parameters",
        "optim": "Optimisation",
        "optim_v": "AdamW (wd = 0), lr = {lr}, 3% linear warm-up + cosine, gradient clipping {c}",
        "batch": "Batch", "batch_v": "effective {e} (micro {m})",
        "epochs": "Epochs / steps", "epochs_v": "{e} / {s} (warm-up {w})",
        "seqlen": "Sequence length", "seqlen_v": "at most {n}",
        "kd_v": "τ = {tau}, α = {alpha}, forward KL, first {v} vocabulary indices",
        "hidden": "Hidden state (S3)", "hidden_v": "β = {b}, pairs {p}, W: {ds}→{dt} (shared)",
        "seeds": "Seeds",
        "time_mem": "Time / peak memory — {m}", "time_mem_v": "{t} min / {g} GB",
        "model": "Model", "mean_row": "**{c} (mean)**",
        "t2_note": "Accuracies in %; GSM8K test n = 1319, MMLU n = 1000, HellaSwag n = 1000, ARC-C n = 1172. "
                   "Primary metric: GSM8K 0-shot flexible.",
        "t3_hdr": ["Hypothesis", "Comparison", "Seed", "Acc. A %", "Acc. B %", "Difference (points)",
                   "Bootstrap 95% CI", "b (only A)", "c (only B)", "p (raw)", "p (Holm)", "Significant (α = 0.05)",
                   "Direction", "Decision"],
        "t3_note": "Primary metric: GSM8K test 0-shot flexible. Exact binomial McNemar test (two-sided), family of "
                   "4 tests, Holm. Bootstrap: question-level paired, {n} resamples, seed {s}.",
        "t4_hdr": ["Task", "Metric", "Comparison", "Seed", "A", "B", "Difference", "95% CI", "b", "c",
                   "p (raw)", "p (Holm)", "Significant", "Decision"],
        "t4_note": "Accuracy tasks: values in %, difference in points; exact binomial McNemar test, family of 12 "
                   "tests, Holm. Hypothesised direction: A − S1 > 0 (smaller drop than S1). WikiText-2: PPL and PPL "
                   "difference; window-level paired bootstrap ({n}, seed {s}); significant = 95% CI excludes 0; "
                   "hypothesised direction: difference < 0.",
        "t5_primary": "primary (0-shot flexible)", "t5_secondary": "secondary (4-shot flexible)",
        "t5_hdr": ["Metric", "Configuration", "Seed 42 %", "Seed 1234 %", "Mean %", "Range %"],
        "t5_note": "Gap-closed ratio = (Sx − S0) / (T − S0) × 100. Negative values lie below S0.",
        "metric": "Metric",
        "t6_note": "Secondary metrics (prereg §5). For each metric 4 tests in the H1/H2 structure, Holm within "
                   "the metric; decisions are descriptive (the pre-registered decision uses the primary metric).",
        "t7_hdr": ["Run", "Sx %", "S0 %", "Difference (points)", "Bootstrap 95% CI", "b", "c",
                   "p (raw, no Holm)"],
        "t7_note": "{e}. Primary metric; exact binomial McNemar test, no correction.",
        "t8_hdr": ["Model", "\"####\" usage %", "n (used)", "n (not used)", "Flexible acc. % (used)",
                   "Flexible acc. % (not used)"],
        "t8_note": "{e}. GSM8K test 0-shot; \"####\" usage = 1 − format error (a strict answer exists).",
        "f1_y": "GSM8K accuracy (%, 0-shot, flexible)",
        "f1_cap": "GSM8K test (n = 1319) 0-shot flexible accuracy. Bars: seed mean; markers: seed 42 (filled) and "
                  "1234 (open), error bars 95% Wilson confidence intervals. Dashed line: S0 (untrained student); "
                  "dash-dotted line: T (teacher).",
        "f2_x": "Mean drop on general benchmarks (points, vs. S0)",
        "f2_y": "GSM8K primary gain (points, vs. S0)",
        "f2_cap": "Learning–forgetting plane. x: mean drop relative to S0 in MMLU (acc), HellaSwag (acc_norm) and "
                  "ARC-C (acc_norm) accuracy (points; positive = drop). y: gain relative to S0 in GSM8K test 0-shot "
                  "flexible accuracy (points). Small markers: single runs (two seeds); large markers: configuration "
                  "means; star: S0 (0, 0).",
        "f3_ya": "Change vs. S0 (points)", "f3_yb": "WikiText-2 PPL change (vs. S0)",
        "f3_cap": "(a) Accuracy change relative to S0 on the general benchmarks (points; negative = drop): MMLU "
                  "(n = 1000, acc), HellaSwag (n = 1000, acc_norm), ARC-C (n = 1172, acc_norm). (b) WikiText-2 test "
                  "perplexity change (Sx − S0; positive = worse). Bars: mean of two seeds; error bars: seed range "
                  "(min–max).",
        "f4_y": "Gap-closed ratio (%)",
        "f4_cap": "Gap-closed ratio (Sx − S0) / (T − S0), primary metric (GSM8K test 0-shot flexible). Bars: seed "
                  "mean; markers: seed 42 (filled) and 1234 (open). 0% = S0 level (dashed), 100% = T level "
                  "(dash-dotted).",
        "f5_kl": "KL (τ²-scaled)", "f5_feat": "Feature loss", "f5_x": "Step",
        "f5_cap": "Training curves (25-step moving average; solid line seed 42, dashed seed 1234). (a) CE on answer "
                  "tokens (S1, S2, S3). (b) Forward KL scaled by τ² (S2, S3). (c) Intermediate-layer feature loss "
                  "(S3; mean over the four layer pairs, layer-normalised MSE).",
        "f6_y": "\"####\" usage rate (%, 0-shot)",
        "f6_cap": "Exploratory (not pre-registered): rate of using the \"#### number\" format in GSM8K test 0-shot "
                  "generations (1 − format error). For S1–S3 the bar is the seed mean and the markers are the seed "
                  "values.",
    },
    "tr": {
        "decimal": ",", "thousands": ".", "ci_sep": "; ",
        "seed": "tohum", "yes": "evet", "no": "hayır",
        "favourable": "lehte", "unfavourable": "aleyhte", "descriptive": " (betimsel)",
        "exploratory": "Keşif amaçlı, ön kayıt dışı",
        "decisions": {SUPPORTED: "desteklendi", PARTIAL: "kısmen desteklendi", NOT_SUPPORTED: "desteklenmedi",
                      OPPOSITE: "desteklenmedi (ters yönde anlamlı)", INCONSISTENT: "tutarsız (desteklenmedi)"},
        "tables": {"T1": "C1_kosullar", "T2": "C2_ana_sonuclar", "T3": "C3_H1_H2", "T4": "C4_H3",
                   "T5": "C5_kapanan_bosluk", "T6": "C6_ikincil_olcutler", "T7": "C7_S0a_karsi_kesif",
                   "T8": "C8_bicim_kesif"},
        "figures": {"F1": "S1_gsm8k_birincil", "F2": "S2_ogrenme_unutma", "F3": "S3_genel_olcut_degisim",
                    "F4": "S4_kapanan_bosluk", "F5": "S5_egitim_egrileri", "F6": "S6_bicim_kullanimi_kesif"},
        "item_value": ["Öğe", "Değer"],
        "teacher": "Öğretmen (T)", "teacher_v": "{m} (dondurulmuş, bf16)",
        "student": "Öğrenci (S)", "student_v": "{m} + LoRA",
        "train_data": "Eğitim verisi", "train_data_v": "GSM8K train_rest, n = {n}",
        "longest": "En uzun örnek / kırpılan", "longest_v": "{a} belirteç / {b}",
        "test": "Test", "test_v": "GSM8K test, n = 1319",
        "prompt": "İstem", "prompt_v": "Question: {soru}\\nAnswer: — sohbet şablonu yok",
        "lora_v": "r = {r}, alfa = {a}, dropout = {d}, hedef: {t}",
        "lora_params": "Eğitilebilir LoRA parametresi",
        "optim": "Optimizasyon",
        "optim_v": "AdamW (wd = 0), lr = {lr}, %3 doğrusal ısınma + kosinüs, gradyan kırpma {c}",
        "batch": "Yığın", "batch_v": "etkin {e} (mikro {m})",
        "epochs": "Dönem / adım", "epochs_v": "{e} / {s} (ısınma {w})",
        "seqlen": "Dizi uzunluğu", "seqlen_v": "en fazla {n}",
        "kd_v": "τ = {tau}, α = {alpha}, ileri KL, ilk {v} sözlük indeksi",
        "hidden": "Gizli durum (S3)", "hidden_v": "β = {b}, çiftler {p}, W: {ds}→{dt} (paylaşılan)",
        "seeds": "Tohumlar",
        "time_mem": "Süre / en yüksek bellek — {m}", "time_mem_v": "{t} dk / {g} GB",
        "model": "Model", "mean_row": "**{c} (ortalama)**",
        "t2_note": "Doğruluklar %; GSM8K test n = 1319, MMLU n = 1000, HellaSwag n = 1000, ARC-C n = 1172. "
                   "Birincil ölçüt: GSM8K 0-shot flexible.",
        "t3_hdr": ["Hipotez", "Karşılaştırma", "Tohum", "Doğr. A %", "Doğr. B %", "Fark (puan)", "Bootstrap %95 GA",
                   "b (yalnız A)", "c (yalnız B)", "p (ham)", "p (Holm)", "Anlamlı (α = 0,05)", "Yön", "Karar"],
        "t3_note": "Birincil ölçüt: GSM8K test 0-shot flexible. Kesin binom McNemar (iki yönlü), aile 4 test, Holm. "
                   "Bootstrap: soru düzeyinde eşleştirilmiş, {n} yeniden örnekleme, seed {s}.",
        "t4_hdr": ["Görev", "Ölçüt", "Karşılaştırma", "Tohum", "A", "B", "Fark", "%95 GA", "b", "c",
                   "p (ham)", "p (Holm)", "Anlamlı", "Karar"],
        "t4_note": "Doğruluk görevleri: değerler %, fark puan; kesin binom McNemar, aile 12 test, Holm. Hipotez yönü: "
                   "A − S1 > 0 (S1'den az düşüş). WikiText-2: PPL ve PPL farkı; pencere düzeyinde eşleştirilmiş "
                   "bootstrap ({n}, seed {s}); anlamlı = %95 GA 0'ı dışlar; hipotez yönü: fark < 0.",
        "t5_primary": "birincil (0-shot flexible)", "t5_secondary": "ikincil (4-shot flexible)",
        "t5_hdr": ["Ölçüt", "Yapılandırma", "Tohum 42 %", "Tohum 1234 %", "Ortalama %", "Aralık %"],
        "t5_note": "Kapanan boşluk oranı = (Sx − S0) / (T − S0) × 100. Negatif değer S0'ın altını gösterir.",
        "metric": "Ölçüt",
        "t6_note": "İkincil ölçütler (prereg §5). Her ölçüt için H1/H2 yapısında 4 test, ölçüt içinde Holm; "
                   "kararlar betimseldir (ön kayıtlı karar birincil ölçüttedir).",
        "t7_hdr": ["Koşu", "Sx %", "S0 %", "Fark (puan)", "Bootstrap %95 GA", "b", "c", "p (ham, Holm'suz)"],
        "t7_note": "{e}. Birincil ölçüt; kesin binom McNemar, düzeltme yok.",
        "t8_hdr": ["Model", "\"####\" kullanım %", "n (kullanılan)", "n (kullanılmayan)",
                   "Flexible doğr. % (kullanılan)", "Flexible doğr. % (kullanılmayan)"],
        "t8_note": "{e}. GSM8K test 0-shot; \"####\" kullanımı = 1 − format hatası (strict cevap var).",
        "f1_y": "GSM8K doğruluğu (%, 0-shot, flexible)",
        "f1_cap": "GSM8K test (n = 1319) 0-shot flexible doğruluğu. Çubuk: tohum ortalaması; işaretler: tohum 42 "
                  "(dolu) ve 1234 (boş) değerleri, hata çubukları %95 Wilson güven aralığı. Kesikli çizgi: S0 "
                  "(eğitimsiz öğrenci); kesik-noktalı çizgi: T (öğretmen).",
        "f2_x": "Genel ölçütlerde ortalama düşüş (puan, S0'a göre)",
        "f2_y": "GSM8K birincil kazanç (puan, S0'a göre)",
        "f2_cap": "Öğrenme–unutma düzlemi. x: MMLU (acc), HellaSwag (acc_norm) ve ARC-C (acc_norm) doğruluklarındaki "
                  "S0'a göre düşüşün ortalaması (puan; pozitif = düşüş). y: GSM8K test 0-shot flexible doğruluğunda "
                  "S0'a göre kazanç (puan). Küçük işaretler tek koşular (iki tohum), büyük işaretler yapılandırma "
                  "ortalamaları; yıldız S0 (0, 0).",
        "f3_ya": "S0'a göre değişim (puan)", "f3_yb": "WikiText-2 PPL değişimi (S0'a göre)",
        "f3_cap": "(a) Genel ölçütlerde S0'a göre doğruluk değişimi (puan; negatif = düşüş): MMLU (n = 1000, acc), "
                  "HellaSwag (n = 1000, acc_norm), ARC-C (n = 1172, acc_norm). (b) WikiText-2 test perplexity "
                  "değişimi (Sx − S0; pozitif = kötüleşme). Çubuk: iki tohumun ortalaması; hata çubuğu: tohum "
                  "aralığı (en küçük–en büyük).",
        "f4_y": "Kapanan boşluk oranı (%)",
        "f4_cap": "Kapanan boşluk oranı (Sx − S0) / (T − S0), birincil ölçüt (GSM8K test 0-shot flexible). Çubuk: "
                  "tohum ortalaması; işaretler: tohum 42 (dolu) ve 1234 (boş). %0 = S0 düzeyi (kesikli), %100 = T "
                  "düzeyi (kesik-noktalı).",
        "f5_kl": "KL (τ² ölçekli)", "f5_feat": "Öznitelik kaybı", "f5_x": "Adım",
        "f5_cap": "Eğitim eğrileri (25 adımlık hareketli ortalama; düz çizgi tohum 42, kesikli tohum 1234). (a) Cevap "
                  "belirteçlerinde CE (S1, S2, S3). (b) τ² ile ölçeklenmiş ileri KL (S2, S3). (c) Ara katman "
                  "öznitelik kaybı (S3; dört katman çiftinin ortalaması, katman normalize MSE).",
        "f6_y": "\"####\" kullanım oranı (%, 0-shot)",
        "f6_cap": "Keşif amaçlı (ön kayıt dışı): GSM8K test 0-shot üretimlerinde \"#### sayı\" biçimini kullanma "
                  "oranı (1 − format hatası). S1–S3 için çubuk tohum ortalaması, işaretler tohum değerleri.",
    },
}


class IntegrityError(RuntimeError):
    pass


def cfg_of(run: str) -> str:
    return run.split("_")[0]


def seed_of(run: str) -> int:
    return int(run.split("seed")[1])


# ---------------------------------------------------------------------------
# Inputs and integrity
# ---------------------------------------------------------------------------

def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        raise IntegrityError(f"missing file: {p.relative_to(ROOT)}")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def gsm_path(model: str, shots: int) -> Path:
    if model == "S0":
        return RES / f"day1/eval/student_instruct_{shots}shot_vllm.jsonl"
    if model == "T":
        return RES / f"day1/eval/teacher_{shots}shot_vllm.jsonl"
    return RES / f"train/{model}/gsm8k_test_{shots}shot.jsonl"


def gen_dir(model: str) -> Path:
    return RES / "general" / {"S0": "student_instruct", "T": "teacher"}.get(model, f"student_instruct_{model}")


def load_all() -> tuple[dict, list[str]]:
    checks: list[str] = []
    gsm: dict = {m: {} for m in MODELS}
    for k in SHOTS:
        ref_idx = ref_gold = None
        for m in MODELS:
            recs = read_jsonl(gsm_path(m, k))
            if len(recs) != EXPECT_N["gsm8k"]:
                raise IntegrityError(f"GSM8K {k}-shot {m}: n={len(recs)} != {EXPECT_N['gsm8k']}")
            missing = [f for f in ("correct_flexible", "correct_strict", "format_error") if f not in recs[0]]
            if missing:
                raise IntegrityError(f"GSM8K {k}-shot {m}: v2 fields missing {missing} (run rescore.py)")
            idx, gold = [r["idx"] for r in recs], [r["gold"] for r in recs]
            if ref_idx is None:
                ref_idx, ref_gold = idx, gold
            elif idx != ref_idx or gold != ref_gold:
                raise IntegrityError(f"GSM8K {k}-shot {m}: idx/gold order differs from T")
            gsm[m][k] = {"flexible": np.array([bool(r["correct_flexible"]) for r in recs]),
                         "strict": np.array([bool(r["correct_strict"]) for r in recs]),
                         "format_error": np.array([bool(r["format_error"]) for r in recs]),
                         "n_score_errors": sum(bool(r.get("score_error")) for r in recs)}
        checks.append(f"GSM8K {k}-shot: {len(MODELS)} models × n=1319, identical idx and gold-answer order")

    gen: dict = {m: {} for m in MODELS}
    for task, key in GEN_TASKS.items():
        ref = None
        for m in MODELS:
            recs = read_jsonl(gen_dir(m) / f"{task}.jsonl")
            if len(recs) != EXPECT_N[task]:
                raise IntegrityError(f"{task} {m}: n={len(recs)} != {EXPECT_N[task]}")
            sig = ([r["idx"] for r in recs], [r["gold"] for r in recs])
            if ref is None:
                ref = sig
            elif sig != ref:
                raise IntegrityError(f"{task} {m}: idx/gold order differs from T")
            n_err = sum(bool(r.get("error")) for r in recs)
            gen[m][task] = {"correct": np.array([bool(r[key]) for r in recs]), "n_errors": n_err,
                            "acc": np.array([bool(r["correct"]) for r in recs]).mean(),
                            "acc_norm": np.array([bool(r["correct_norm"]) for r in recs]).mean()}
        checks.append(f"{GEN_LABEL[task]}: {len(MODELS)} models × n={EXPECT_N[task]}, identical idx and gold order")
    ref = None
    for m in MODELS:
        recs = read_jsonl(gen_dir(m) / "wikitext2.jsonl")
        sig = [(r["window"], r["begin"], r["end"], r["score_from"], r["n_tokens"]) for r in recs]
        if ref is None:
            ref = sig
        elif sig != ref:
            raise IntegrityError(f"WikiText-2 {m}: window/token layout differs from T")
        bad = [r["window"] for r in recs if r.get("error") or r.get("nll_sum") is None]
        if bad:
            raise IntegrityError(f"WikiText-2 {m}: faulty windows {bad[:5]}")
        gen[m]["wikitext2"] = {"nll": np.array([r["nll_sum"] for r in recs], dtype=np.float64),
                               "tokens": np.array([r["n_tokens"] for r in recs], dtype=np.float64)}
    checks.append(f"WikiText-2: {len(MODELS)} models × {len(ref)} windows, {int(sum(s[4] for s in ref))} "
                  "scored tokens; identical window boundaries, no faulty windows")

    train: dict = {}
    for run in RUNS:
        d = RES / "train" / run
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        if not s.get("completed"):
            raise IntegrityError(f"{run}: training incomplete ({s.get('steps_done')}/{s.get('total_steps')})")
        train[run] = {"summary": s, "meta": meta, "log": read_jsonl(d / "train_log.jsonl")}
    for seed in SEEDS:
        shas = {train[f"{c}_seed{seed}"]["summary"]["data_order_sha256"] for c in CFGS}
        if len(shas) != 1:
            raise IntegrityError(f"seed {seed}: S1/S2/S3 data-order digests differ")
    steps = {train[r]["summary"]["total_steps"] for r in RUNS}
    if len(steps) != 1:
        raise IntegrityError(f"step counts differ: {steps}")
    checks.append(f"Training: 6 runs completed, all with {steps.pop()} steps; identical S1/S2/S3 data-order digest "
                  "within each seed")
    n_score = {m: gsm[m][k]["n_score_errors"] for m in MODELS for k in SHOTS if gsm[m][k]["n_score_errors"]}
    n_gerr = {f"{m}/{t}": gen[m][t]["n_errors"] for m in MODELS for t in GEN_TASKS if gen[m][t]["n_errors"]}
    checks.append(f"Scoring errors: GSM8K {n_score or 'none'}; general-benchmark example errors {n_gerr or 'none'}")
    return {"gsm": gsm, "gen": gen, "train": train}, checks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def acc_test(a: np.ndarray, b: np.ndarray, label: str, seed: int, direction: str = "greater") -> dict:
    t = mcnemar_paired(a, b)
    t["ci95_points"] = list(paired_bootstrap_acc(a, b))
    t.update({"label": label, "seed": seed,
              "in_direction": t["diff_points"] > 0 if direction == "greater" else t["diff_points"] < 0})
    return t


def apply_holm(tests: list[dict]) -> None:
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
        t["significant"] = p < ALPHA


def hypothesis_family(data: dict, shots: int, metric: str) -> dict:
    """H1 (S2 > S1) and H2 (S3 > S2), 2 seeds, Holm (4 tests)."""
    g = data["gsm"]
    tests = []
    for seed in SEEDS:
        for h, (a, b) in (("H1", ("S2", "S1")), ("H2", ("S3", "S2"))):
            t = acc_test(g[f"{a}_seed{seed}"][shots][metric], g[f"{b}_seed{seed}"][shots][metric], f"{a}−{b}", seed)
            t["hypothesis"] = h
            tests.append(t)
    apply_holm(tests)
    decisions = {h: decide([t for t in tests if t["hypothesis"] == h]) for h in ("H1", "H2")}
    return {"shots": shots, "metric": metric, "tests": tests, "decisions": decisions}


def h3(data: dict) -> dict:
    gen = data["gen"]
    tests = []
    for task in GEN_TASKS:
        for cx in ("S2", "S3"):
            for seed in SEEDS:
                t = acc_test(gen[f"{cx}_seed{seed}"][task]["correct"], gen[f"S1_seed{seed}"][task]["correct"],
                             f"{cx}−S1", seed)
                t["task"] = task
                tests.append(t)
    apply_holm(tests)
    decisions = {f"{task}:{cx}−S1": decide([t for t in tests if t["task"] == task and t["label"] == f"{cx}−S1"])
                 for task in GEN_TASKS for cx in ("S2", "S3")}
    # WikiText-2: window bootstrap; significant = 95% CI excludes 0; hypothesised direction: PPL difference < 0
    wiki = []
    w = {m: gen[m]["wikitext2"] for m in MODELS}
    tok = w["S0"]["tokens"]
    for cx in ("S2", "S3"):
        for seed in SEEDS:
            r = paired_bootstrap_ppl(w[f"{cx}_seed{seed}"]["nll"], w[f"S1_seed{seed}"]["nll"], tok)
            lo, hi = r["ci95"]
            r.update({"label": f"{cx}−S1", "seed": seed, "significant": lo > 0 or hi < 0, "in_direction": r["diff"] < 0})
            wiki.append(r)
    for cx in ("S2", "S3"):
        decisions[f"wikitext2:{cx}−S1"] = decide([r for r in wiki if r["label"] == f"{cx}−S1"])
    vs_s0_ppl = []
    for run in RUNS:
        r = paired_bootstrap_ppl(w[run]["nll"], w["S0"]["nll"], tok)
        r.update({"run": run})
        vs_s0_ppl.append(r)
    # Drop relative to S0 (points) and the mean over the three tasks
    drops = {}
    for run in RUNS:
        d = {task: 100.0 * (gen["S0"][task]["correct"].mean() - gen[run][task]["correct"].mean()) for task in GEN_TASKS}
        d["mean3"] = float(np.mean([d[t] for t in GEN_TASKS]))
        drops[run] = d
    return {"tests": tests, "wikitext": wiki, "wikitext_vs_s0": vs_s0_ppl, "drops": drops, "decisions": decisions}


def as4(data: dict) -> dict:
    g = data["gsm"]
    out = {}
    for shots in SHOTS:
        s0, t = g["S0"][shots]["flexible"].mean(), g["T"][shots]["flexible"].mean()
        per = {run: gap_closed(g[run][shots]["flexible"].mean(), s0, t) for run in RUNS}
        by_cfg = {}
        for c in CFGS:
            v = [per[f"{c}_seed{s}"] for s in SEEDS]
            by_cfg[c] = {"seeds": dict(zip(SEEDS, v)), "mean": float(np.mean(v)), "min": min(v), "max": max(v)}
        out[f"{shots}shot_flexible"] = {"S0": float(s0), "T": float(t), "per_run": per, "by_cfg": by_cfg}
    return out


def vs_s0(data: dict) -> list[dict]:
    g = data["gsm"]
    return [acc_test(g[run][0]["flexible"], g["S0"][0]["flexible"], f"{run}−S0", seed_of(run)) | {"run": run}
            for run in RUNS]


def format_analysis(data: dict) -> dict:
    out = {}
    for m in MODELS:
        d = data["gsm"][m][0]
        used = ~d["format_error"]
        out[m] = {"usage": float(used.mean()), "n_used": int(used.sum()),
                  "acc_used": float(d["flexible"][used].mean()) if used.any() else None,
                  "acc_unused": float(d["flexible"][~used].mean()) if (~used).any() else None}
    return out


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def num(x, nd=1, lang="en") -> str:
    """Fixed-point number with the language's decimal separator and a typographic minus; None -> —."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", LABELS[lang]["decimal"]).replace("-", "−")


def pval(p, lang="en") -> str:
    if p is None:
        return "—"
    dec = LABELS[lang]["decimal"]
    if p < 1e-4:
        m, e = f"{p:.1e}".split("e")
        return f"{m.replace('.', dec)}·10^{int(e)}"
    return f"{p:.4f}".replace(".", dec)


def ci_str(ci, nd=1, lang="en") -> str:
    return f"[{num(ci[0], nd, lang)}{LABELS[lang]['ci_sep']}{num(ci[1], nd, lang)}]"


def grouped(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", LABELS[lang]["thousands"])


def model_label(m: str, lang: str) -> str:
    return m if m in ("T", "S0") else f"{cfg_of(m)} ({LABELS[lang]['seed']} {seed_of(m)})"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_table(lang: str, key: str, headers: list[str], rows: list[list], md_rows: list[list[str]],
                note: str | None = None) -> None:
    """CSV: raw values (point decimal); MD: language-specific formatting."""
    d = OUT / f"tables-{lang}"
    d.mkdir(parents=True, exist_ok=True)
    stem = LABELS[lang]["tables"][key]
    with open(d / f"{stem}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(headers)
        for r in rows:
            w.writerow(["" if v is None else (f"{v:.6g}" if isinstance(v, float) else v) for v in r])
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in md_rows]
    if note:
        lines += ["", note]
    (d / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def tables(data: dict, R: dict, cfg: dict, lang: str) -> None:
    L = LABELS[lang]
    dec = L["decisions"]
    g, gen, train = data["gsm"], data["gen"], data["train"]
    tc = cfg["train"]

    # T1 — conditions and hyperparameters
    s_any = train[RUNS[0]]["summary"]
    ds = train[RUNS[0]]["meta"].get("data_stats", {})
    kv = [
        (L["teacher"], L["teacher_v"].format(m=cfg["models"]["teacher"])),
        (L["student"], L["student_v"].format(m=cfg["models"][cfg["student"]])),
        (L["train_data"], L["train_data_v"].format(n=s_any["n_examples"])),
        (L["longest"], L["longest_v"].format(a=ds.get("max_len_tokens", "—"), b=ds.get("n_truncated", "—"))),
        (L["test"], L["test_v"]),
        (L["prompt"], L["prompt_v"]),
        ("LoRA", L["lora_v"].format(r=tc["lora"]["r"], a=tc["lora"]["alpha"], d=num(tc["lora"]["dropout"], 2, lang),
                                    t=", ".join(tc["lora"]["target_modules"]))),
        (L["lora_params"], grouped(s_any["n_trainable_lora"], lang)),
        (L["optim"], L["optim_v"].format(lr=f"{tc['lr']:g}", c=num(tc["max_grad_norm"], 1, lang))),
        (L["batch"], L["batch_v"].format(e=s_any["effective_batch"], m=s_any["micro_batch"])),
        (L["epochs"], L["epochs_v"].format(e=s_any["epochs"], s=s_any["total_steps"], w=s_any["warmup_steps"])),
        (L["seqlen"], L["seqlen_v"].format(n=tc["max_seq_len"])),
        ("KD (S2, S3)", L["kd_v"].format(tau=num(tc["kd"]["tau"], 0, lang), alpha=num(tc["kd"]["alpha"], 1, lang),
                                         v=grouped(tc["kd"]["vocab_cut"], lang))),
        (L["hidden"], L["hidden_v"].format(b=num(tc["hidden"]["beta"], 0, lang),
                                           p=", ".join(f"{a}↔{b}" for a, b in tc["hidden"]["pairs"]),
                                           ds=tc["hidden"]["student_dim"], dt=tc["hidden"]["teacher_dim"])),
        (L["seeds"], ", ".join(map(str, SEEDS))),
    ]
    rows = [[k, v] for k, v in kv]
    for run in RUNS:
        s = train[run]["summary"]
        rows.append([L["time_mem"].format(m=model_label(run, lang)),
                     L["time_mem_v"].format(t=num(s["wall_time_s"] / 60, 1, lang), g=num(s["peak_mem_gb"], 1, lang))])
    csv_rows = [[k, v] for k, v in kv] + [[f"wall_time_min/peak_mem_gb {run}",
                                           f"{train[run]['summary']['wall_time_s'] / 60:.2f}/{train[run]['summary']['peak_mem_gb']:.2f}"]
                                          for run in RUNS]
    write_table(lang, "T1", L["item_value"], csv_rows, rows)

    # T2 — main results
    def row_vals(m):
        return [100 * g[m][0]["flexible"].mean(), 100 * g[m][0]["strict"].mean(),
                100 * g[m][4]["flexible"].mean(), 100 * g[m][4]["strict"].mean(),
                100 * gen[m]["mmlu"]["acc"], 100 * gen[m]["hellaswag"]["acc_norm"], 100 * gen[m]["arc_c"]["acc_norm"],
                ppl(gen[m]["wikitext2"]["nll"], gen[m]["wikitext2"]["tokens"])]
    hdr = [L["model"], "GSM8K 0-shot flex", "GSM8K 0-shot strict", "GSM8K 4-shot flex", "GSM8K 4-shot strict",
           "MMLU acc", "HellaSwag acc_norm", "ARC-C acc_norm", "WikiText-2 PPL"]
    csv_rows, md_rows = [], []
    order = ["T", "S0"] + [r for c in CFGS for r in (f"{c}_seed42", f"{c}_seed1234", f"{c}_mean")]
    for m in order:
        if m.endswith("_mean"):
            c = m.split("_")[0]
            vals = list(np.mean([row_vals(f"{c}_seed{s}") for s in SEEDS], axis=0))
            label = L["mean_row"].format(c=c)
        else:
            vals, label = row_vals(m), model_label(m, lang)
        csv_rows.append([m] + [float(v) for v in vals])
        md_rows.append([label] + [num(v, 1, lang) for v in vals[:-1]] + [num(vals[-1], 3, lang)])
    write_table(lang, "T2", hdr, csv_rows, md_rows, L["t2_note"])

    # T3 — H1/H2
    def test_rows(fam, extra=""):
        cr, mr = [], []
        for t in fam["tests"]:
            direction = L["favourable"] if t["in_direction"] else L["unfavourable"]
            decision = dec[fam["decisions"][t["hypothesis"]]]
            cr.append([t["hypothesis"], t["label"], t["seed"], 100 * t["acc_a"], 100 * t["acc_b"], t["diff_points"],
                       t["ci95_points"][0], t["ci95_points"][1], t["b_only_a"], t["c_only_b"], t["p"], t["p_holm"],
                       direction, decision])
            mr.append([t["hypothesis"], t["label"], str(t["seed"]), num(100 * t["acc_a"], 1, lang),
                       num(100 * t["acc_b"], 1, lang), num(t["diff_points"], 1, lang), ci_str(t["ci95_points"], 1, lang),
                       str(t["b_only_a"]), str(t["c_only_b"]), pval(t["p"], lang), pval(t["p_holm"], lang),
                       L["yes"] if t["significant"] else L["no"], direction, decision + extra])
        return cr, mr
    hdr3 = L["t3_hdr"]
    cr, mr = test_rows(R["primary"])
    write_table(lang, "T3", hdr3, cr, mr, L["t3_note"].format(n=N_BOOT, s=BOOT_SEED))

    # T4 — H3
    cr, mr = [], []
    for t in R["h3"]["tests"]:
        decision = dec[R["h3"]["decisions"][f"{t['task']}:{t['label']}"]]
        cr.append([GEN_LABEL[t["task"]], GEN_METRIC[t["task"]], t["label"], t["seed"], 100 * t["acc_a"], 100 * t["acc_b"],
                   t["diff_points"], t["ci95_points"][0], t["ci95_points"][1], t["b_only_a"], t["c_only_b"], t["p"],
                   t["p_holm"], decision])
        mr.append([GEN_LABEL[t["task"]], GEN_METRIC[t["task"]], t["label"], str(t["seed"]), num(100 * t["acc_a"], 1, lang),
                   num(100 * t["acc_b"], 1, lang), num(t["diff_points"], 1, lang), ci_str(t["ci95_points"], 1, lang),
                   str(t["b_only_a"]), str(t["c_only_b"]), pval(t["p"], lang), pval(t["p_holm"], lang),
                   L["yes"] if t["significant"] else L["no"], decision])
    for r in R["h3"]["wikitext"]:
        decision = dec[R["h3"]["decisions"][f"wikitext2:{r['label']}"]]
        cr.append(["WikiText-2", "PPL", r["label"], r["seed"], r["ppl_a"], r["ppl_b"], r["diff"], r["ci95"][0],
                   r["ci95"][1], None, None, None, None, decision])
        mr.append(["WikiText-2", "PPL", r["label"], str(r["seed"]), num(r["ppl_a"], 3, lang), num(r["ppl_b"], 3, lang),
                   num(r["diff"], 3, lang), ci_str(r["ci95"], 3, lang), "—", "—", "—", "—",
                   L["yes"] if r["significant"] else L["no"], decision])
    write_table(lang, "T4", L["t4_hdr"], cr, mr, L["t4_note"].format(n=N_BOOT, s=BOOT_SEED))

    # T5 — gap closed
    cr, mr = [], []
    for key, lab in (("0shot_flexible", L["t5_primary"]), ("4shot_flexible", L["t5_secondary"])):
        a = R["as4"][key]
        for c in CFGS:
            b = a["by_cfg"][c]
            cr.append([lab, c, b["seeds"][42], b["seeds"][1234], b["mean"], b["min"], b["max"]])
            mr.append([lab, c, num(100 * b["seeds"][42], 1, lang), num(100 * b["seeds"][1234], 1, lang),
                       num(100 * b["mean"], 1, lang), ci_str([100 * b["min"], 100 * b["max"]], 1, lang)])
    write_table(lang, "T5", L["t5_hdr"], cr, mr, L["t5_note"])

    # T6 — secondary metrics
    cr, mr = [], []
    for fam in R["secondary"]:
        lab = f"{fam['shots']}-shot {fam['metric']}"
        c6, m6 = test_rows(fam, L["descriptive"])
        cr += [[lab] + r for r in c6]
        mr += [[lab] + r for r in m6]
    write_table(lang, "T6", [L["metric"]] + hdr3, cr, mr, L["t6_note"])

    # T7 — against S0 (exploratory)
    cr, mr = [], []
    for t in R["vs_s0"]:
        cr.append([t["run"], 100 * t["acc_a"], 100 * t["acc_b"], t["diff_points"], t["ci95_points"][0],
                   t["ci95_points"][1], t["b_only_a"], t["c_only_b"], t["p"]])
        mr.append([model_label(t["run"], lang), num(100 * t["acc_a"], 1, lang), num(100 * t["acc_b"], 1, lang),
                   num(t["diff_points"], 1, lang), ci_str(t["ci95_points"], 1, lang), str(t["b_only_a"]),
                   str(t["c_only_b"]), pval(t["p"], lang)])
    write_table(lang, "T7", L["t7_hdr"], cr, mr, L["t7_note"].format(e=L["exploratory"]))

    # T8 — format (exploratory)
    cr, mr = [], []
    for m in MODELS:
        f = R["format"][m]
        cr.append([m, f["usage"], f["n_used"], EXPECT_N["gsm8k"] - f["n_used"], f["acc_used"], f["acc_unused"]])
        mr.append([model_label(m, lang), num(100 * f["usage"], 1, lang), str(f["n_used"]),
                   str(EXPECT_N["gsm8k"] - f["n_used"]),
                   num(None if f["acc_used"] is None else 100 * f["acc_used"], 1, lang),
                   num(None if f["acc_unused"] is None else 100 * f["acc_unused"], 1, lang)])
    write_table(lang, "T8", L["t8_hdr"], cr, mr, L["t8_note"].format(e=L["exploratory"]))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COL = {"S0": "#7F7F7F", "S1": "#0072B2", "S2": "#E69F00", "S3": "#009E73", "T": "#CC79A7"}  # Okabe–Ito
MRK = {"S1": "o", "S2": "s", "S3": "^"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    names = {f.name for f in font_manager.fontManager.ttflist}
    rcParams.update({
        "font.family": "serif",
        "font.serif": (["Times New Roman"] if "Times New Roman" in names else []) + ["DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "kdq", "axes.unicode_minus": True,
    })
    return "Times New Roman" in names


def decimal_axis(ax, lang, x=True, y=True, nd=None):
    from matplotlib.ticker import FuncFormatter

    def f(v, _):
        s = f"{v:.{nd}f}" if nd is not None else f"{v:g}"
        return s.replace(".", LABELS[lang]["decimal"]).replace("-", "−")
    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(f))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(f))


def save_fig(fig, lang: str, key: str, caption: str) -> None:
    import matplotlib.pyplot as plt
    d = OUT / f"figures-{lang}"
    d.mkdir(parents=True, exist_ok=True)
    name = LABELS[lang]["figures"][key]
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    (d / f"{name}.txt").write_text(caption.strip() + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def figures(data: dict, R: dict, lang: str) -> None:
    import matplotlib.pyplot as plt
    L = LABELS[lang]
    g, train = data["gsm"], data["train"]
    acc0 = {m: g[m][0]["flexible"].mean() for m in MODELS}
    n = EXPECT_N["gsm8k"]

    # F1
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xs = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}
    k0 = int(g["S0"][0]["flexible"].sum())
    lo, hi = wilson_ci(k0, n)
    ax.bar(0, 100 * acc0["S0"], width=0.55, color=COL["S0"], alpha=0.35, edgecolor=COL["S0"])
    ax.errorbar(0, 100 * acc0["S0"], yerr=[[100 * (acc0["S0"] - lo)], [100 * (hi - acc0["S0"])]], fmt="D",
                color=COL["S0"], capsize=3, ms=4)
    lows = [lo]
    for c in CFGS:
        vals = [acc0[f"{c}_seed{s}"] for s in SEEDS]
        ax.bar(xs[c], 100 * np.mean(vals), width=0.55, color=COL[c], alpha=0.35, edgecolor=COL[c])
        for j, s in enumerate(SEEDS):
            k = int(g[f"{c}_seed{s}"][0]["flexible"].sum())
            lo, hi = wilson_ci(k, n)
            lows.append(lo)
            ax.errorbar(xs[c] + (-0.12 if j == 0 else 0.12), 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]],
                        fmt=MRK[c], color=COL[c], capsize=3, ms=4, mfc="white" if j else COL[c])
    ax.axhline(100 * acc0["S0"], color=COL["S0"], ls="--", lw=0.9)
    ax.axhline(100 * acc0["T"], color=COL["T"], ls="-.", lw=0.9)
    ax.text(3.68, 100 * acc0["T"] + 0.2, "T", va="bottom", ha="right", color=COL["T"])
    ax.text(3.68, 100 * acc0["S0"] + 0.2, "S0", va="bottom", ha="right", color=COL["S0"])
    ax.set_xticks(list(xs.values()), list(xs.keys()))
    ax.set_xlim(-0.5, 3.7)
    ax.set_ylim(np.floor(100 * min(lows)) - 2, 100 * acc0["T"] + 3)
    ax.set_ylabel(L["f1_y"])
    decimal_axis(ax, lang, x=False)
    save_fig(fig, lang, "F1", L["f1_cap"])

    # F2
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    drops = R["h3"]["drops"]
    for c in CFGS:
        xs_, ys_ = [], []
        for s in SEEDS:
            run = f"{c}_seed{s}"
            x, y = drops[run]["mean3"], 100 * (acc0[run] - acc0["S0"])
            xs_.append(x)
            ys_.append(y)
            ax.scatter(x, y, marker=MRK[c], color=COL[c], s=28, zorder=3)
        ax.scatter(np.mean(xs_), np.mean(ys_), marker=MRK[c], color=COL[c], s=110, edgecolor="black", lw=0.8,
                   zorder=4, label=c)
    ax.scatter(0, 0, marker="*", color=COL["S0"], s=120, zorder=4, label="S0")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(0, color="0.6", lw=0.7)
    ax.set_xlabel(L["f2_x"])
    ax.set_ylabel(L["f2_y"])
    ax.legend(frameon=False, loc="best")
    decimal_axis(ax, lang)
    save_fig(fig, lang, "F2", L["f2_cap"])

    # F3
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 3.2), gridspec_kw={"width_ratios": [3, 1.2]})
    w = 0.25
    for i, c in enumerate(CFGS):
        means, lo_e, hi_e = [], [], []
        for task in GEN_TASKS:
            v = [-drops[f"{c}_seed{s}"][task] for s in SEEDS]
            means.append(np.mean(v))
            lo_e.append(np.mean(v) - min(v))
            hi_e.append(max(v) - np.mean(v))
        x = np.arange(len(GEN_TASKS)) + (i - 1) * w
        ax.bar(x, means, width=w, color=COL[c], label=c, yerr=[lo_e, hi_e], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(np.arange(len(GEN_TASKS)), [f"{GEN_LABEL[t]}\n({GEN_METRIC[t]})" for t in GEN_TASKS])
    ax.set_ylabel(L["f3_ya"])
    ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, va="top")
    decimal_axis(ax, lang, x=False)
    vs0 = {r["run"]: r["diff"] for r in R["h3"]["wikitext_vs_s0"]}
    for i, c in enumerate(CFGS):
        v = [vs0[f"{c}_seed{s}"] for s in SEEDS]
        bx.bar(i, np.mean(v), width=0.6, color=COL[c], yerr=[[np.mean(v) - min(v)], [max(v) - np.mean(v)]], capsize=2,
               error_kw={"lw": 0.8})
    bx.axhline(0, color="black", lw=0.7)
    bx.set_xticks(range(3), CFGS)
    bx.set_ylabel(L["f3_yb"])
    bx.text(0.03, 0.98, "(b)", transform=bx.transAxes, va="top")
    decimal_axis(bx, lang, x=False, nd=2)
    fig.tight_layout()
    save_fig(fig, lang, "F3", L["f3_cap"])

    # F4
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    a = R["as4"]["0shot_flexible"]["by_cfg"]
    for i, c in enumerate(CFGS):
        ax.bar(i, 100 * a[c]["mean"], width=0.55, color=COL[c], alpha=0.35, edgecolor=COL[c])
        for j, s in enumerate(SEEDS):
            ax.scatter(i + (-0.1 if j == 0 else 0.1), 100 * a[c]["seeds"][s], marker=MRK[c], color=COL[c],
                       facecolor="white" if j else COL[c], s=26, zorder=3)
    ax.axhline(0, color=COL["S0"], ls="--", lw=0.9)
    ax.axhline(100, color=COL["T"], ls="-.", lw=0.9)
    ax.set_xticks(range(3), CFGS)
    ax.set_ylabel(L["f4_y"])
    decimal_axis(ax, lang, x=False)
    save_fig(fig, lang, "F4", L["f4_cap"])

    # F5
    def smooth(v, k=25):
        v = np.asarray(v, dtype=float)
        return np.convolve(v, np.ones(k) / k, mode="valid"), k
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    panels = [("ce", CFGS, "CE"), ("kl", ("S2", "S3"), L["f5_kl"]), ("feat", ("S3",), L["f5_feat"])]
    for ax, (key, cfgs, lab), tag in zip(axs, panels, "abc"):
        for c in cfgs:
            for j, s in enumerate(SEEDS):
                y, k = smooth([r[key] for r in train[f"{c}_seed{s}"]["log"]])
                ax.plot(np.arange(len(y)) + k, y, color=COL[c], lw=0.9, ls="-" if j == 0 else "--",
                        label=f"{c}, {L['seed']} {s}" if key == "ce" else None)
        ax.set_xlabel(L["f5_x"])
        ax.set_ylabel(lab)
        ax.text(0.97, 0.97, f"({tag})", transform=ax.transAxes, va="top", ha="right")
        decimal_axis(ax, lang, x=False, nd=2)
    axs[0].legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    fig.tight_layout()
    save_fig(fig, lang, "F5", L["f5_cap"])

    # F6 (exploratory)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    fmt = R["format"]
    labels = ["S0", "S1", "S2", "S3", "T"]
    for i, m in enumerate(labels):
        if m in CFGS:
            v = [fmt[f"{m}_seed{s}"]["usage"] for s in SEEDS]
            ax.bar(i, 100 * np.mean(v), width=0.55, color=COL[m], alpha=0.35, edgecolor=COL[m])
            for j, s in enumerate(SEEDS):
                ax.scatter(i + (-0.1 if j == 0 else 0.1), 100 * v[j], marker=MRK[m], color=COL[m],
                           facecolor="white" if j else COL[m], s=24, zorder=3)
        else:
            ax.bar(i, 100 * fmt[m]["usage"], width=0.55, color=COL[m], alpha=0.6, edgecolor=COL[m])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(L["f6_y"])
    ax.set_ylim(0, 105)
    decimal_axis(ax, lang, x=False)
    save_fig(fig, lang, "F6", L["f6_cap"])


# ---------------------------------------------------------------------------
# Findings report (English)
# ---------------------------------------------------------------------------

def findings(R: dict, checks: list[str]) -> str:
    def f(x, nd=1):
        return num(x, nd, "en")

    def c(ci, nd=1):
        return ci_str(ci, nd, "en")

    def p(x):
        return pval(x, "en")

    L = ["# Findings — KD-QWEN-14B-7B (analysis output)", "",
         "This document is generated by `scripts/analyze.py` and contains no interpretation. Decisions follow "
         "mechanically from the rule in docs/preregistration.md (prereg-v1.0) §6.", "", "## Input integrity", ""]
    L += [f"- {x}" for x in checks]
    L += ["", "## Pre-registered analyses", "",
          "### H1 (S2 > S1) and H2 (S3 > S2) — primary metric: GSM8K test, 0-shot, flexible", ""]
    fam = R["primary"]
    for t in fam["tests"]:
        L.append(f"- {t['hypothesis']} · {t['label']} · seed {t['seed']}: {f(100 * t['acc_a'])} − {f(100 * t['acc_b'])} "
                 f"= {f(t['diff_points'])} points, bootstrap 95% CI {c(t['ci95_points'])}; b = {t['b_only_a']}, "
                 f"c = {t['c_only_b']}; p = {p(t['p'])}, Holm p = {p(t['p_holm'])} "
                 f"({'significant' if t['significant'] else 'not significant'}, "
                 f"{'in the hypothesised direction' if t['in_direction'] else 'opposite to the hypothesis'})")
    L += ["", f"**H1: {fam['decisions']['H1']}** · **H2: {fam['decisions']['H2']}**", "",
          "### H3 — general benchmarks (S2 − S1, S3 − S1)", "",
          "McNemar (12 tests, Holm):", ""]
    for t in R["h3"]["tests"]:
        L.append(f"- {GEN_LABEL[t['task']]} ({GEN_METRIC[t['task']]}) · {t['label']} · seed {t['seed']}: "
                 f"{f(100 * t['acc_a'])} − {f(100 * t['acc_b'])} = {f(t['diff_points'])} points, CI {c(t['ci95_points'])}; "
                 f"b = {t['b_only_a']}, c = {t['c_only_b']}; p = {p(t['p'])}, Holm p = {p(t['p_holm'])}")
    L += ["", "WikiText-2 (window-level paired bootstrap, PPL difference):", ""]
    for r in R["h3"]["wikitext"]:
        L.append(f"- {r['label']} · seed {r['seed']}: {f(r['ppl_a'], 3)} − {f(r['ppl_b'], 3)} = {f(r['diff'], 3)}, "
                 f"95% CI {c(r['ci95'], 3)} ({'excludes 0' if r['significant'] else 'includes 0'})")
    L += ["", "Decisions (per task):", ""]
    for k, v in R["h3"]["decisions"].items():
        task, comp = k.split(":")
        L.append(f"- {GEN_LABEL[task]} · {comp}: **{v}**")
    L += ["", "Drop relative to S0 (points; positive = drop):", "",
          "| Run | MMLU | HellaSwag | ARC-C | Mean of three tasks | WikiText-2 PPL − S0 [95% CI] |",
          "|---|---:|---:|---:|---:|---|"]
    vs0 = {r["run"]: r for r in R["h3"]["wikitext_vs_s0"]}
    for run in RUNS:
        d = R["h3"]["drops"][run]
        L.append(f"| {model_label(run, 'en')} | {f(d['mmlu'])} | {f(d['hellaswag'])} | {f(d['arc_c'])} | {f(d['mean3'])} | "
                 f"{f(vs0[run]['diff'], 3)} {c(vs0[run]['ci95'], 3)} |")
    for cf in CFGS:
        m3 = np.mean([R["h3"]["drops"][f"{cf}_seed{s}"]["mean3"] for s in SEEDS])
        L.append(f"| **{cf} (mean)** | | | | {f(m3)} | {f(np.mean([vs0[f'{cf}_seed{s}']['diff'] for s in SEEDS]), 3)} |")
    L += ["", "### AS4 — gap-closed ratio (descriptive)", ""]
    for key, lab in (("0shot_flexible", "Primary (0-shot flexible)"), ("4shot_flexible", "Secondary (4-shot flexible)")):
        a = R["as4"][key]
        L.append(f"- {lab}: S0 = {f(100 * a['S0'])}, T = {f(100 * a['T'])}")
        for cf in CFGS:
            b = a["by_cfg"][cf]
            L.append(f"  - {cf}: seed 42 = {f(100 * b['seeds'][42])}%, seed 1234 = {f(100 * b['seeds'][1234])}%; "
                     f"mean {f(100 * b['mean'])}%, range [{f(100 * b['min'])}%, {f(100 * b['max'])}%]")
    L += ["", "## Secondary metrics (prereg §5; descriptive)", ""]
    for fam in R["secondary"]:
        L += [f"### GSM8K {fam['shots']}-shot {fam['metric']}", ""]
        for t in fam["tests"]:
            L.append(f"- {t['hypothesis']} · {t['label']} · seed {t['seed']}: {f(t['diff_points'])} points, "
                     f"CI {c(t['ci95_points'])}; p = {p(t['p'])}, Holm p (within metric) = {p(t['p_holm'])}")
        L += [f"- Rule applied (descriptive): H1 {fam['decisions']['H1']}; H2 {fam['decisions']['H2']}", ""]
    L += [f"## Exploratory analyses ({EXPLORATORY})", "", "### Each run − S0 (primary metric, no Holm)", ""]
    for t in R["vs_s0"]:
        L.append(f"- {model_label(t['run'], 'en')}: {f(t['diff_points'])} points, CI {c(t['ci95_points'])}; "
                 f"b = {t['b_only_a']}, c = {t['c_only_b']}; p = {p(t['p'])}")
    L += ["", "### Format: 0-shot \"####\" usage", ""]
    for m in MODELS:
        fm = R["format"][m]
        L.append(f"- {model_label(m, 'en')}: usage {f(100 * fm['usage'])}%; flexible accuracy — "
                 f"used (n = {fm['n_used']}): {f(None if fm['acc_used'] is None else 100 * fm['acc_used'])}, "
                 f"not used (n = {EXPECT_N['gsm8k'] - fm['n_used']}): "
                 f"{f(None if fm['acc_unused'] is None else 100 * fm['acc_unused'])}")
    L += ["", "## Methodological notes", "",
          f"- McNemar: exact binomial, two-sided (src/kdq/stats.py). Holm, α = {f(ALPHA, 2)}.",
          f"- Bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}; paired at the question level for accuracy differences "
          "and at the window level for WikiText-2, percentile 95% CI; each comparison is generated independently "
          "with the same seed.",
          "- Hypothesis decision: Holm-significant in the hypothesised direction in both seeds → supported; in only "
          "one → partially supported; in none → not supported; none in the hypothesised direction but significant "
          "in the opposite direction → not supported (significant in the opposite direction); one seed in the "
          "hypothesised direction and the other significant in the opposite direction → inconsistent (not "
          "supported). For WikiText-2, significance means that the 95% CI excludes 0.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------

def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main() -> int:
    cfg = load_config()
    try:
        data, checks = load_all()
    except IntegrityError as e:
        print(f"[ERROR] Integrity check failed: {e}")
        return 2
    for c in checks:
        print(f"[integrity] {c}")
    R = {
        "primary": hypothesis_family(data, 0, "flexible"),
        "secondary": [hypothesis_family(data, 0, "strict"), hypothesis_family(data, 4, "flexible"),
                      hypothesis_family(data, 4, "strict")],
        "h3": h3(data),
        "as4": as4(data),
        "vs_s0": vs_s0(data),
        "format": format_analysis(data),
    }
    for lang in LANGS:
        for sub in (f"figures-{lang}", f"tables-{lang}"):
            if (OUT / sub).exists():
                shutil.rmtree(OUT / sub)
    OUT.mkdir(parents=True, exist_ok=True)
    has_tnr = setup_mpl()
    for lang in LANGS:
        tables(data, R, cfg, lang)
        figures(data, R, lang)
    (OUT / "findings.md").write_text(findings(R, checks), encoding="utf-8", newline="\n")
    with open(OUT / "analysis.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(to_jsonable({"checks": checks, "results": R, "font_times_new_roman": has_tnr,
                               "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "alpha": ALPHA}), f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[analysis] H1: {R['primary']['decisions']['H1']} | H2: {R['primary']['decisions']['H2']}")
    for k, v in R["h3"]["decisions"].items():
        print(f"[analysis] H3 {k}: {v}")
    print(f"[analysis] outputs: {OUT.relative_to(ROOT)} (Times New Roman: {'yes' if has_tnr else 'no, serif fallback'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
