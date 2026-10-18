# Pre-registration — KD-QWEN-14B-7B

Version 1.0 · 12 October 2026 · This document was committed BEFORE the training runs and tagged `prereg-v1.0`.
Every later deviation is recorded with its date in docs/decisions.md and reported together with the results.

## 1. Models
- Teacher (T): Qwen/Qwen2.5-14B-Instruct — frozen, bf16, forward passes only.
- Student (S): Qwen/Qwen2.5-7B-Instruct — trained with LoRA.
- Selection rationale (pre-specified rule): the smaller of the 0-shot and 4-shot T − S0 gaps. 7B-Instruct: 10.5 points ("sufficient"); 7B base: 6.7 points ("borderline"). The target data remain GSM8K (gap ≥ 5 points).

## 2. Data and format
- Training: GSM8K train_rest (7,173 questions). Validation (val, 300 questions) is used for the pilot only. Test (1,319 questions) is used only in the final evaluation.
- The splits are fixed under data/splits/ with sha256 digests.
- Prompt: "Question: {question}\nAnswer:"; target: " {solution}" + the student's eos token. The <<...>> calculator annotations are removed from the solutions. No chat template is used for any model.

## 3. Configurations
- S0: untrained student. T: teacher.
- S1: L = CE.
- S2: L = (1 − α)·CE + α·τ²·KL(p_T^τ ‖ p_S^τ). Forward KL; only over real tokens (the first 151,665 vocabulary indices); only at answer tokens.
- S3: L = L_S2 + β·Σ_l MSE(norm(W·h_S^(l)), norm(h_T^(m))). Layer mapping 7/14/21/28 ↔ 12/24/36/48; W: learned linear projection 3584→5120 (discarded after training).
- Seeds: 42 and 1234. In total 3 × 2 = 6 training runs.
- Data, order, number of epochs, LoRA configuration, learning rate, sequence length and effective batch size are identical in all runs; the loss function is the only difference.

## 4. Hyperparameter pilot (validation set only)
- The shared hyperparameters (learning rate, number of epochs) are selected with S1 (CE) only, based on validation accuracy, so that the selection is not biased in favour of distillation.
- The distillation-specific values are fixed at their defaults: τ = 2, α = 0.5, β = 1. They are NOT tuned on validation performance. They may only be changed for training stability (NaN, divergence, or an initial feature loss exceeding CE by more than an order of magnitude); in that case the change and its rationale are recorded.
- At the end of the pilot all values are frozen and written to docs/decisions.md; the main runs start afterwards.

## 5. Evaluation
- Primary metric: GSM8K test accuracy, 0-shot, flexible answer extraction (v2), vLLM, greedy decoding, at most 512 new tokens.
- Secondary: GSM8K strict and 4-shot.
- Forgetting metrics (zero-shot, log-likelihood, HF, bf16): MMLU 1,000 (stratified proportionally by subject) — acc; HellaSwag 1,000 — acc_norm; ARC-Challenge 1,172 — acc_norm; WikiText-2 test — perplexity (window 2,048, stride 512).
- Trained models are evaluated with their LoRA adapter, using the same code and settings as the untrained models.

## 6. Hypotheses and statistical tests
- H1: S2 > S1 (primary metric).
- H2: S3 > S2 (primary metric). A negative result is also reported as a finding.
- H3: the drop of S2 and S3 relative to S0 on the general benchmarks is smaller than that of S1.
- AS4 (descriptive): gap-closed ratio = (Sx − S0) / (T − S0) on the primary metric; seed mean and range.
- Tests are seed-paired (e.g. S2-seed42 vs. S1-seed42).
  - H1–H2: question-level exact binomial McNemar test. 2 hypotheses × 2 seeds = 4 tests, Holm correction, α = 0.05. A hypothesis counts as "supported" if it is significant in the same direction in both seeds, and as "partially supported" if only in one.
  - H3: S2–S1 and S3–S1 McNemar tests for MMLU, HellaSwag and ARC-C (3 tasks × 2 comparisons × 2 seeds = 12 tests, Holm). For WikiText-2, a window-level paired bootstrap (10,000 resamples), 95% CI. Reported per task; the mean of the three accuracy drops is also given as a summary.
- All results (positive or negative) are reported; there is no selective reporting.

## 7. Exclusion and re-run rules
- No test question or example is excluded.
- A crashed run is restarted from scratch with the same seed; this is recorded.
- No early stopping or model selection is performed by looking at the test set.

## 8. Baseline values (before training)
GSM8K, flexible, % (0-shot / 4-shot): T 87.4 / 92.8 · S0-Instruct 76.9 / 81.7 · 7B base 80.7 / 82.2.
General benchmarks (T / S0-Instruct): MMLU 78.7 / 70.9 · HellaSwag 84.5 / 80.6 · ARC-C 62.4 / 55.0 · WikiText-2 PPL 4.98 / 6.51.
Details: results/day1/gap_report.md and results/general/general_report.md.
