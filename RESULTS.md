# Results

Summary of the pre-registered knowledge-distillation study (teacher `Qwen/Qwen2.5-14B-Instruct`, student
`Qwen/Qwen2.5-7B-Instruct` + LoRA, GSM8K). All numbers are produced by the scripts in this repository from the
records under `results/`; the complete, uninterpreted output is `results/analysis/findings.md`, and the
figures and tables are available in English (`figures-en/`, `tables-en/`) and Turkish (`figures-tr/`,
`tables-tr/`). Accuracies are in %, differences in percentage points.

## 1. Baseline and student selection

GSM8K test (n = 1319), greedy decoding, flexible answer extraction:

| Model | 0-shot | 4-shot |
|---|---:|---:|
| T (14B-Instruct) | 87.4 | 92.8 |
| S0 (7B-Instruct) | 76.9 | 81.7 |
| 7B base | 80.7 | 82.2 |

Under the pre-specified rule (the smaller of the 0- and 4-shot teacher–student gaps; ≥ 8 points sufficient,
5–8 borderline), the instruct student leaves a gap of 10.5 points (*sufficient*) and the base model 6.7 points
(*borderline*); `Qwen2.5-7B-Instruct` was therefore used as the student (`results/day1/gap_report.md`).

## 2. Hyperparameter pilot

The learning rate was selected with S1 (CE) on the 300-question validation split only: lr 1e-4 → 85.7
[81.2, 89.2], lr 2e-4 → 84.0 [79.4, 87.7]; lr = 1e-4 was frozen. No run produced NaN/inf values or
diverged, and the largest initial β·L_feat / CE ratio of S3 was 2.11 (< 10), so the distillation
hyperparameters τ = 2, α = 0.5, β = 1 remained at their pre-registered defaults
(`results/train/pilot/pilot_report.md`). All six main runs completed 898 steps with identical data order
within each seed.

## 3. Main results

| Model | GSM8K 0-shot flex | 0-shot strict | 4-shot flex | MMLU acc | HellaSwag acc_norm | ARC-C acc_norm | WikiText-2 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| T | 87.4 | 0.0 | 92.8 | 78.7 | 84.5 | 62.4 | 4.975 |
| S0 | 76.9 | 0.0 | 81.7 | 70.9 | 80.6 | 55.0 | 6.508 |
| S1 (mean of 2 seeds) | 76.8 | 76.3 | 76.6 | 70.0 | 80.0 | 52.9 | 6.381 |
| S2 (mean of 2 seeds) | 83.1 | 34.9 | 81.7 | 71.1 | 79.5 | 56.5 | 6.670 |
| S3 (mean of 2 seeds) | 82.2 | 33.2 | 79.1 | 70.4 | 79.6 | 56.1 | 6.674 |

Per-seed values: `results/analysis/tables-en/T2_main_results.md`.

## 4. Pre-registered hypotheses

Seed-paired exact McNemar tests with Holm correction (α = 0.05); a hypothesis is *supported* when it is
significant in the hypothesised direction for both seeds.

- **H1 (S2 > S1, primary metric): supported.** S2 − S1 = +5.7 points (seed 42; 95% bootstrap CI [3.5, 7.9],
  Holm p = 1.9·10⁻⁶) and +6.7 points (seed 1234; [4.5, 9.1], Holm p = 6.1·10⁻⁸). Logit distillation clearly
  outperforms CE fine-tuning on the same data.
- **H2 (S3 > S2, primary metric): not supported.** S3 − S2 = −1.0 points ([−2.7, 0.7]) and −0.7 points
  ([−2.4, 1.1]), Holm p = 0.57 for both. Adding intermediate-layer alignment did not improve on logit
  distillation; the point estimates are slightly negative.
- **H3 (S2 and S3 forget less than S1):** task-dependent.
  - ARC-C: **supported** for S2 − S1 (+4.0, +3.3) and S3 − S1 (+3.3, +3.2), all Holm p ≤ 0.009.
  - MMLU and HellaSwag: **not supported** (differences between −0.8 and +1.4 points, none Holm-significant).
  - WikiText-2: **not supported, significant in the opposite direction** — S2 and S3 have a higher
    perplexity than S1 (+0.29 in every seed-paired comparison, 95% CIs excluding 0). Relative to S0, S1
    *lowers* perplexity (−0.13) whereas S2/S3 raise it (+0.16 to +0.17).
  - Mean accuracy drop over the three accuracy tasks relative to S0: S1 1.1–1.3 points, S2 −0.3 to 0.0,
    S3 0.0 to 0.3.

## 5. Gap closed (descriptive)

Gap-closed ratio (Sx − S0) / (T − S0) on the primary metric (0-shot flexible): S1 −0.4% (range −5.0 to
4.3), **S2 58.6%** (58.3 to 59.0), **S3 50.7%** (48.9 to 52.5). Under 4-shot evaluation (secondary), S1
falls below the untrained student (−45.6%), S2 is at the S0 level on average (0.3%) and S3 below it (−22.8%).

## 6. Secondary and exploratory observations

- **Answer format.** Neither T nor S0 uses the `#### number` format in 0-shot generation. CE training on
  GSM8K solutions makes S1 adopt it almost always (99.5%), whereas S2 and S3 use it in only 42–44% of the
  answers and otherwise answer in the teacher's style. This explains the low 0-shot *strict* accuracy of
  S2/S3 (≈ 34%) despite their higher flexible accuracy, and indicates that logit distillation transfers the
  teacher's output style together with its solutions (exploratory, `T8_format_exploratory`).
- **Few-shot evaluation.** With 4-shot prompts the seed means are S2 81.7 (equal to S0), S3 79.1 and S1 76.6;
  the benefit of distillation is therefore concentrated in the 0-shot setting used as the primary metric.
- **Against S0 (exploratory, uncorrected).** S2 and S3 improve on S0 by 5.1–6.2 points in the primary
  metric, while S1 stays within ±0.6 points of S0.

## 7. Deviations and limitations

- One deviation from the pre-registration is documented in `docs/decisions.md`: after the analysis, the
  decision rule was extended to the case "significant in the hypothesised direction in one seed and in the
  opposite direction in the other" (→ *inconsistent, not supported*). This case does not occur in the data
  and changes no decision.
- Two seeds per configuration, a single reasoning dataset, LoRA rather than full fine-tuning, and a single
  layer mapping for S3; the conclusions on H2 and H3 are limited to this setting.
