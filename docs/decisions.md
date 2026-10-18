# Decisions and assumptions

The design decisions of the study (models, losses, layer mapping, metrics) are fixed. This log records the
defaults chosen where the design left details open, the values frozen after the pilot, and every deviation
from the pre-registration (docs/preregistration.md, `prereg-v1.0`), each with its date.

## Baseline stage

- **Package versions:** `vllm==0.10.2` (pinned to torch==2.8.0, CUDA 12.8 wheel). The other packages are
  pinned to versions compatible with it. `setup_pod.sh` constrains torch/torchvision/torchaudio/triton to
  their installed versions (`-c env/constraints_torch.txt`) so that pip cannot replace torch.
- **Splits:** Python `random.Random(42).sample` (independent of the numpy version). val = 300,
  train_rest = 7173. The 4-shot exemplars are `random.Random(42).sample(train_rest, 4)`; the returned order is
  the order in the prompt.
- **Answer extraction:** the first number after the last `####`, otherwise the last number in the text.
  Fractions and scientific notation are not supported. Comparison with `Decimal`, tolerance 1e-6.
- **Prompt format:** plain text `Question: … / Answer: …`, reference solutions with the calculator
  annotations (`<<…>>`) removed.
- **check_models eos/pad:** an eos difference between base (`<|endoftext|>`) and Instruct (`<|im_end|>`) is
  expected; since it does not affect the KL it is reported as a warning (an error with `--strict`).
- **Token-ID comparison:** the first 20 examples of GSM8K **train** (question + solution) are used; the test
  split is not touched.
- **Layer indexing:** in the HF `output_hidden_states` output, index 0 is the embedding layer; the output of
  "layer 7" is `hidden_states[7]`.

## Baseline stage — resolved open questions (2026-10-12)

1. **Split files:** `data/splits/*.json` are produced on the pod with `make_splits.py` and committed.
2. **KL dimension:** the KL is computed only over real tokens, i.e. over the first `len(tokenizer)` = 151665
   logit indices. The padding rows 151665–152063 (`config.vocab_size` = 152064) are excluded.
3. **Prompt format (final):** all models (teacher, students, reference) use the plain-text `Question/Answer`
   format; no chat template is used (a single, fixed prompt). `src/kdq/prompts.py` implements this format.
4. **eos difference:** during training the student's eos (`<|endoftext|>`) is appended to the target
   sequence. During evaluation generation is cut at the stop sequences (the model's eos is not relied upon).

## Baseline stage — evaluation assumptions

- **Prompt:** `"Question: {question}\nAnswer:"`, target `" {solution}"` (`<<…>>` removed). k-shot exemplars
  and the query are joined with `"\n\n"`. No chat template, including for the Instruct models. An earlier
  instruction line was removed.
- **Stopping:** stop sequences `"\n\nQuestion:"` and `"\nQuestion:"`, stop tokens `<|endoftext|>` (151643)
  and `<|im_end|>` (151645), identical in both engines. The output is additionally cut at the same sequences
  in post-processing.
- **The model's generation_config is ignored:** the Instruct defaults (repetition_penalty=1.05,
  temperature=0.7) are not applied (vllm `generation_config="vllm"`, explicit parameters in hf). Pure greedy
  decoding for all models.
- **Format error:** no `####` in the output. **Unanswered:** no number could be extracted. **Truncated:** the
  512-token limit was reached. Outputs with a format error are still scored with the "last number" rule.
- **Decision rule (gap_report):** for each S0, the **smaller** of the 0- and 4-shot T − S0 gaps is used.
  ≥ 8 points *sufficient*, 5–8 *borderline*, < 5 *insufficient*. The candidate with the best status is
  recommended; ties go to `student_base`.
- **McNemar:** exact binomial, two-sided, informational only.

## Decision rule and reporting (2026-10-12)

1. **Decision rule (confirmed):** for each student the **smaller** of the 0-shot and 4-shot T − S0 gaps is
   used (conservative reading). ≥ 8 points *sufficient*, 5–8 *borderline*, < 5 *insufficient*; within the same
   category `student_base` is preferred. `gap_report` writes this as a **RECOMMENDATION** only.
   The **primary evaluation setting** (0-shot / 4-shot) is fixed in the pre-registration after the baseline
   results; until then both are reported.
2. **0-shot format error:** without `####` the last-number rule is kept; the format-error rate is reported
   next to the accuracy for every model × shot in the `gap_report` table.
3. **JSONL outputs:** committed under `results/day1` (1–2 MB per file).
4. **Engine cross-check:** a warning is issued if answer agreement is < 90%; the pipeline does not stop.
   `engine_crosscheck.json` also contains the accuracy difference between the two engines in points
   (`accuracy_diff_points` = vllm − hf).

## Environment notes

- The RunPod image sets `HF_HUB_ENABLE_HF_TRANSFER=1`; since `hf_transfer` is not installed in the venv,
  downloads raise `ValueError`. All shell scripts set `HF_HUB_ENABLE_HF_TRANSFER=0`.
- `run_day1_baseline.sh` is self-contained: it changes to the repository root, activates `$WORKSPACE/venv`
  and sets `HF_HOME=$WORKSPACE/hf_cache` (default `WORKSPACE=/workspace`); it can be started with `nohup`.

## Answer extraction v2 (strict + flexible)

Answer extraction v2 was defined after inspecting the raw generations of the baseline measurement; it was
frozen before training and is applied identically to all configurations.

- **Rationale:** in the cross-check (student_base, 0-shot, vllm, n = 50) v1 gave an accuracy of 0.14 with a
  format-error rate of 0.98. The model mostly solved the problem correctly and wrote "The answer is 18.",
  then continued with self-generated questions introduced by `[Question]`, `Q:`, `A:`. The "last number" rule
  of v1 picked up the numbers of these hallucinated questions.
- **strict:** only the first `####` and the number immediately following it; otherwise unanswered
  (= format error).
- **flexible:** the generation is cut (i) after the first line, at the first new-question marker at a line
  start (`Question`, `[Question]`, `Q:`/`Q1:`, `Problem:`, `### Question`; case-insensitive, word-bounded) and
  (ii) at the end of the first `####` line. In the cut text, in order: `#### number` → first
  `answer is (:) number` → last number.
- **Record fields:** `pred_strict`, `correct_strict`, `pred_flexible`, `correct_flexible`, `truncated_at`.
  `pred`/`correct` equal the flexible values for backward compatibility. `format_error` = strict unanswered.
- **Reporting:** `gap_report` and `engine_crosscheck` report both modes separately. The decision rule is
  applied to **flexible**, with strict shown alongside.
- **Generation settings unchanged:** stop sequences, stop tokens and decoding settings are identical to the
  baseline run. Existing outputs are re-scored from the raw generations with `scripts/rescore.py`; the v1
  summaries are kept as `*.summary.v1.json`.

## Scoring robustness (2026-10-12)

- **Bug:** on the pod, the student_base 4-shot run crashed during scoring with `decimal.InvalidOperation`
  (the model generated a very long digit string; `quantize` exceeded the 28-digit Decimal precision).
- **Number validity:** numbers longer than `MAX_DIGITS = 30` digits, non-finite or unparsable numbers become
  `None` (no exception). If an extraction step (`####` / `answer is` / last number) hits an invalid number,
  the result of that step is `None`, i.e. the question counts as unanswered/wrong; no other number is tried.
- **Raw generations first:** `eval_gsm8k.py` writes the raw generations (`*.raw.jsonl`) and the run
  information (`*.raw.meta.json`) to disk **before** scoring. If scoring crashes, `rescore.py` completes it
  from these files.
- **Question-level error handling:** a scoring error for a single question does not abort the run. The
  question counts as wrong/unanswered, the error is stored in the record's `score_error` field and the count
  in the summary's `n_score_errors` field.
- **Pipeline:** `run_day1_baseline.sh` always runs `rescore.py` before `gap_report`, so that older outputs
  scored with v1 are brought to v2.

## End of the baseline stage (2026-10-12)

1. **Student = `student_instruct` (Qwen/Qwen2.5-7B-Instruct).** Under the pre-specified rule (smaller of the
   0/4-shot gaps, flexible) the minimum T − S0 gap is 10.5 points for Instruct (*sufficient*) and 6.7 points
   for the base model (*borderline*). `configs/base.yaml`: `student: student_instruct`.
2. **GSM8K is sufficient;** no switch to MATH.
3. **Primary GSM8K metric: 0-shot + flexible.** Strict and 4-shot are reported as secondary metrics.
   `configs/base.yaml`: `primary_metric`.
4. **General benchmarks (forgetting axis):** MMLU 1000 (stratified proportionally by subject, seed 42),
   HellaSwag 1000 (validation, seed 42), ARC-Challenge full test set (1172), WikiText-2 full test set.

## General-benchmark evaluation — assumptions

- **Method:** HF transformers, log-likelihood based, zero-shot, bf16, no chat template (consistent with the
  plain format used for GSM8K). The same code evaluates the LoRA models (S1/S2/S3) via `--adapter`.
- **MMLU:** `"{question}\nA. …\nB. …\nC. …\nD. …\nAnswer:"`. The lm-eval header "The following are multiple
  choice questions…" is **not** added. The next-token log-likelihoods of `" A"…" D"` are compared (a single
  forward pass). Since the options are single letters, acc_norm = acc; acc is primary.
- **Stratified sampling:** quotas proportional to subject via the largest-remainder method (exactly 1000 in
  total); `random.Random(42)` within a subject, subjects processed in name order.
- **HellaSwag:** lm-eval preprocessing (`activity_label: ctx_a + " " + ctx_b.capitalize()`, removal of
  `[title]`/bracketed text). **ARC-C:** `"Question: {question}\nAnswer:"`. Continuation = `" " + option`.
  **acc** = total log-likelihood, **acc_norm** = divided by the character length of the option text
  (lm-eval: option length without the leading space). acc_norm is primary. Ties go to the first option.
- **Token boundary:** continuation tokens `enc(context + continuation)[len(enc(context)):]` (lm-eval). On a
  prefix mismatch they are encoded separately. Requests are left-truncated to at most 4096 tokens.
- **Logit dimension:** log_softmax is taken over the model's full output vocabulary (152064), as in lm-eval.
  The "first 151665" choice for the KL applies to the training loss only.
- **WikiText-2:** `Salesforce/wikitext` (current name of the former `wikitext` repository),
  `wikitext-2-raw-v1` test. Lines are joined with `"\n\n"`. Sliding window 2048 / stride 512, HF perplexity
  scheme: every token (except the first) is scored exactly once, with at least 1536 tokens of context outside
  the first window. Per window, `nll_sum` and `n_tokens` are stored (for the paired bootstrap). Perplexity is
  token-level (Qwen tokenizer). Since the tokenizer is identical for all models, the models are comparable.
- **HellaSwag repository:** `Rowan/hellaswag` may be script-based (datasets ≥ 4 does not load scripts). If it
  cannot be loaded, HF's automatic parquet conversion (`refs/convert/parquet`) is used.
- **Robustness:** raw results (`*.raw.jsonl`) are written before scoring. A single example error does not
  abort the run; the example counts as wrong (`n_errors`). On out-of-memory errors the batch size is halved
  (16 → … → 1). The batch size for WikiText windows is 4.
- **Subset integrity:** `eval_general.py` compares the sha256 of the selected rows with
  `data/splits/general_*.json` and skips the task on a mismatch.

## Training code — assumptions and implementation details (2026-10-12)

prereg-v1.0 is binding; the following are implementation details that the pre-registration does not specify.

- **Config:** the `future_training` placeholder became the `train` section (same values). `train.lr: null`
  was to be frozen after the pilot; `run_main.sh` and `train.py` (without `--lr`) stop with an explicit error
  while it is null.
- **Target eos:** `<|endoftext|>` (151643), as decided at the baseline stage. Although the `eos_token` of the
  Qwen2.5-7B-Instruct tokenizer is `<|im_end|>`, evaluation stops at both.
- **Tokenisation:** continuation tokens `enc(prompt + target)[len(enc(prompt)):]` (same boundary as in
  evaluation); on a prefix mismatch the target is encoded separately, counted in `run_meta.json` →
  `data_stats.n_prefix_mismatch`. Examples longer than 512 tokens are truncated at the end (the eos is lost)
  and reported in `data_stats.n_truncated`.
- **Loss positions:** CE/KL at the positions t with `labels[t+1] != -100` (positions predicting the answer +
  eos tokens). Hidden-state alignment at the positions of the answer tokens themselves (`labels != -100`).
- **Token-level averaging:** divided by the total number of answer tokens in the effective batch (16
  examples); micro-batches are weighted by tokens rather than equally (exact token mean).
- **Vocabulary truncation:** CE is computed on the same support as the KL (the first 151,665 logits), in all
  configurations. `len(tokenizer) == 151665` is verified at start-up.
- **Memory:** logits are computed only at the selected positions via `lm_head(hidden_states[-1][positions])`
  (`logits_to_keep=1` skips the model's own full logit computation). A test verifies that `hidden_states[-1]`
  is the normalised final state and the lm_head input.
- **W (S3):** in line with the singular "W" of the pre-registration, a **single, shared** 3584→5120 linear
  projection (with bias) is used for all four pairs (as in TinyBERT). `train.hidden.projection: shared`; a
  `per_layer` option exists in the code but is not used. It is saved as `projector.pt` at the end of training
  and is not attached to the student.
- **LN:** `F.layer_norm` without affine parameters, at the teacher dimension (5120), fp32.
- **Number of steps:** 7173 / 16 = 448.3 → 449 steps per epoch (the last batch has 5 examples and is not
  dropped) → 2 epochs = 898 steps; warm-up round(0.03 · 898) = 27 steps. Identical in all configurations.
- **Data order:** a per-epoch permutation with `random.Random(seed)`; depends only on the run seed and can be
  verified via `summary.json` → `data_order_sha256` (identical for S1/S2/S3 with the same seed).
- **Micro-batch:** 4 (× 4 accumulation = 16). If memory is insufficient, `--micro-batch 2`; the effective
  batch does not change.
- **Checkpoints:** at the end of every epoch (`checkpoints/stepN`: adapter + optimizer + scheduler + W + RNG).
  `--resume` continues from the latest checkpoint; duplicated log steps are dropped. In the tests, the adapter
  weights of interrupted and uninterrupted runs match exactly.
- **Smoke tests:** `smoke_S{1,2,3}_seed42` (20 steps, lr 1e-4) are kept in directories separate from the
  main runs.
- **Pilot recommendation rule:** the lr with the highest val 0-shot flexible accuracy; ties go to the smaller
  lr. Recommendation only.
- **Output names:** `<cfg>_seed<seed>` (lr from the config) or `<cfg>_seed<seed>_lr<lr>` (lr from the command
  line). `eval_gsm8k.py`/`eval_general.py` use `<name>` if the adapter path is `.../<name>/adapter`; `_val`
  is appended to the names of validation outputs.
- **Local test environment:** since the torch 2.8.0 CPU wheel could not be installed on Windows (long
  temporary path), a project-local `.venv` (gitignored) with torch **2.9.1+cpu**, transformers 4.56.2 and
  peft 0.17.1 was used. The pod keeps torch 2.8.0. The training loop, including the GPU/bf16 autocast path,
  was tested on CPU in fp32 only.

## Pilot result and freezing (prereg §4)

12 October 2026 — Pilot (prereg §4): S1 seed 42, val 0-shot flexible: lr 1e-4 → 85.7 [81.2, 89.2],
lr 2e-4 → 84.0 [79.4, 87.7]. By the rule, lr = 1e-4 was frozen. Stability rule: no NaN/divergence,
β·L_feat/CE at most 2.11 (< 10) → τ = 2, α = 0.5, β = 1 unchanged. Epochs = 2.

- `configs/base.yaml` → `train.lr: 0.0001`. The main runs (`run_main.sh`) read this value.
- The slow end-to-end training tests are marked `@pytest.mark.slow`; they are skipped in the default `pytest`
  run and executed with `pytest -m slow`.

## Analysis — implementation details (2026-10-13)

Single entry point `scripts/analyze.py` (statistical core `src/kdq/analysis.py`); outputs in
`results/analysis/`. The output is deterministic (byte-identical across two runs).

- **H1/H2:** a = S2/S3, b = S1/S2 with the same seed; exact binomial McNemar (two-sided), family of 4 tests,
  Holm, α = 0.05. "In the hypothesised direction" = difference > 0.
- **H3:** a = S2/S3, b = S1 with the same seed; pre-registered metrics MMLU acc, HellaSwag acc_norm, ARC-C
  acc_norm. "Drop relative to S0 smaller than that of S1" ⇔ acc_Sx − acc_S1 > 0 (relative to the same S0).
  Family of 12 tests, Holm. Decision per task × comparison (S2−S1, S3−S1), over the two seeds, with the H1 rule.
- **WikiText-2 (H3):** no Holm correction (prereg: bootstrap 95% CI). Significant = the CI excludes 0;
  hypothesised direction = PPL_Sx − PPL_S1 < 0. In each resample PPL = exp(ΣNLL / Σtokens), windows paired.
- **Edge case of the decision rule:** if one seed is significant in the hypothesised direction and the other
  significant in the opposite direction, "partially supported" is written (the pre-registration does not
  define this case separately). The case did not occur in this analysis. (Superseded on 2026-10-13, below.)
- **Bootstrap:** a separate `np.random.default_rng(42)` per comparison, 10,000 resamples, percentile CI.
  Paired at the question level for accuracy and at the window level for WikiText-2.
- **Secondary metrics** (0-shot strict, 4-shot flexible, 4-shot strict): in the H1/H2 structure, Holm over
  the 4 tests within each metric; decisions are descriptive only.
- **Exploratory** (not pre-registered, explicitly labelled): each run − S0 McNemar (no Holm); 0-shot "####"
  usage (= 1 − format error) and flexible accuracy within the subsets.
- **Figures:** Times New Roman (serif fallback), Okabe–Ito palette, no titles; PNG 300 dpi + PDF + caption
  .txt. 25-step moving average in the training-curve figure.
- **Local environment:** matplotlib 3.10.6 added to `.venv` (numpy 2.5.4).

## Update of the decision-rule edge case (2026-10-13)

- One seed Holm-significant in the hypothesised direction and the other in the opposite direction (for
  WikiText-2: 95% CI excluding 0) → **"inconsistent (not supported)"**. This replaces "partially supported"
  in the previous implementation note.
- Note: this rule was defined after the analysis results had been seen, for a case that does **not** occur
  in the current analysis (prereg-v1.0 did not define this case); it changes no hypothesis decision and does
  not affect the results. It is reported as a deviation from the pre-registration.

## Repository preparation (2026-10-18)

- Code, configuration, documentation and reports were translated into English; figures and tables are
  produced in English and Turkish (`figures-{en,tr}/`, `tables-{en,tr}/`) from the same numbers. All
  regenerated reports are numerically identical to the previous versions.
- Absolute pod paths derive from `WORKSPACE` (default `/workspace`) in the shell scripts and in
  `configs/base.yaml` (`${WORKSPACE:-/workspace}`); the default values are unchanged.
- `eval_gsm8k.py`: the default output name (used when `--out` is not given) now takes the split into account
  and uses `adapter_tag`, as described under "Output names"; names for the test split without an adapter are
  unchanged (`{model}_{shots}shot_{engine}`). The main runs and the pilot always pass `--out`, so no reported
  result is affected.
