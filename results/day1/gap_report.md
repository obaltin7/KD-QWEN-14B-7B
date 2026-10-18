# Baseline — teacher–student gap report (GSM8K test)

Engine: `vllm` · greedy · max_new_tokens=512 · plain Question/Answer prompt (no chat template) · answer extraction v2

- **flexible:** the generation is cut at the first hallucinated-question marker / after the first `####` line; `#### number` → `answer is number` → last number.
- **strict:** only the first `#### number`; otherwise unanswered (= format error).

## Accuracy (%, Wilson 95% CI)

| Model | shots | n | flexible | 95% CI | strict | 95% CI | format error | unanswered (flex) | cut at marker | truncated at 512 |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|
| T (14B-Instruct) | 0 | 1319 | 87.4 | [85.5, 89.1] | 0.0 | [0.0, 0.3] | 100.0 | 0.0 | 0.0 | 50.9 |
| T (14B-Instruct) | 4 | 1319 | 92.8 | [91.3, 94.1] | 91.8 | [90.2, 93.2] | 1.7 | 0.0 | 60.7 | 41.8 |
| S0-base (7B) | 0 | 1319 | 80.7 | [78.5, 82.8] | 0.8 | [0.4, 1.4] | 98.8 | 0.0 | 90.1 | 5.5 |
| S0-base (7B) | 4 | 1319 | 82.2 | [80.0, 84.2] | 81.9 | [79.7, 83.9] | 2.4 | 1.0 | 0.7 | 2.0 |
| S0-Instruct (7B-Instruct) | 0 | 1319 | 76.9 | [74.5, 79.1] | 0.0 | [0.0, 0.3] | 100.0 | 0.0 | 0.8 | 6.0 |
| S0-Instruct (7B-Instruct) | 4 | 1319 | 81.7 | [79.5, 83.6] | 81.2 | [79.0, 83.2] | 6.5 | 0.4 | 1.7 | 6.9 |

## T − S0 gap (points) and paired McNemar test (informational)

| Student | shots | gap flexible | p flexible | gap strict | p strict |
|---|---:|---:|---:|---:|---:|
| S0-base (7B) | 0 | +6.7 | 3.1e-08 | -0.8 | 0.00195 |
| S0-base (7B) | 4 | +10.6 | 7.65e-24 | +9.9 | 7.03e-20 |
| S0-Instruct (7B-Instruct) | 0 | +10.5 | 2.57e-16 | +0.0 | 1 |
| S0-Instruct (7B-Instruct) | 4 | +11.1 | 1.07e-25 | +10.6 | 2.12e-22 |

## Recommendation (flexible scoring)

Rule: smaller of the 0- and 4-shot T − S0 gaps ≥ 8 points → *sufficient*; 5–8 → *borderline*; < 5 → *insufficient*. Preference within the same category: student_base, student_instruct.

- S0-base (7B): **borderline** (flexible: 0-shot +6.7, 4-shot +10.6; strict: 0-shot -0.8, 4-shot +9.9)
- S0-Instruct (7B-Instruct): **sufficient** (flexible: 0-shot +10.5, 4-shot +11.1; strict: 0-shot +0.0, 4-shot +10.6)

**RECOMMENDATION:** student = `student_instruct` (S0-Instruct (7B-Instruct)).

_The recommendation follows solely from the pre-specified rule. The primary evaluation setting (0/4-shot) is fixed in the pre-registration._
