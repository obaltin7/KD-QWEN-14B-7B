# General benchmarks (forgetting axis)

Log-likelihood based, zero-shot, bf16, no chat template. Accuracies in % (Wilson 95% CI).

| Model | MMLU acc | HellaSwag acc_norm | ARC-C acc_norm | WikiText-2 PPL ↓ |
|---|---|---|---|---:|
| teacher | 78.7 [76.1, 81.1] | 84.5 [82.1, 86.6] | 62.4 [59.6, 65.1] | 4.975 |
| student_instruct | 70.9 [68.0, 73.6] | 80.6 [78.0, 82.9] | 55.0 [52.2, 57.9] | 6.508 |
| student_instruct_S1_seed1234 | 70.0 [67.1, 72.8] | 79.7 [77.1, 82.1] | 53.4 [50.6, 56.3] | 6.383 |
| student_instruct_S1_seed42 | 70.0 [67.1, 72.8] | 80.3 [77.7, 82.6] | 52.3 [49.4, 55.2] | 6.380 |
| student_instruct_S2_seed1234 | 71.4 [68.5, 74.1] | 79.4 [76.8, 81.8] | 56.7 [53.9, 59.6] | 6.674 |
| student_instruct_S2_seed42 | 70.8 [67.9, 73.5] | 79.5 [76.9, 81.9] | 56.3 [53.5, 59.1] | 6.667 |
| student_instruct_S3_seed1234 | 70.5 [67.6, 73.2] | 79.4 [76.8, 81.8] | 56.7 [53.8, 59.5] | 6.674 |
| student_instruct_S3_seed42 | 70.3 [67.4, 73.1] | 79.7 [77.1, 82.1] | 55.6 [52.8, 58.5] | 6.674 |

## Details

| Model | Task | n | errors | acc | 95% CI | acc_norm | 95% CI | PPL | time (s) |
|---|---|---:|---:|---:|---|---:|---|---:|---:|
| teacher | MMLU | 1000 | 0 | 78.7 | [76.1, 81.1] | — | — | — | 19.6 |
| teacher | HellaSwag | 1000 | 0 | 65.0 | [62.0, 67.9] | 84.5 | [82.1, 86.6] | — | 52.2 |
| teacher | ARC-Challenge | 1172 | 0 | 60.5 | [57.7, 63.3] | 62.4 | [59.6, 65.1] | — | 31.5 |
| teacher | WikiText-2 | 299077 tokens / 582 windows | 0 | — | — | — | — | 4.975 | 167.3 |
| student_instruct | MMLU | 1000 | 0 | 70.9 | [68.0, 73.6] | — | — | — | 11.2 |
| student_instruct | HellaSwag | 1000 | 0 | 61.2 | [58.1, 64.2] | 80.6 | [78.0, 82.9] | — | 27.3 |
| student_instruct | ARC-Challenge | 1172 | 0 | 53.1 | [50.2, 55.9] | 55.0 | [52.2, 57.9] | — | 17.1 |
| student_instruct | WikiText-2 | 299077 tokens / 582 windows | 0 | — | — | — | — | 6.508 | 84.5 |
| student_instruct_S1_seed1234 | MMLU | 1000 | 0 | 70.0 | [67.1, 72.8] | — | — | — | 16.1 |
| student_instruct_S1_seed1234 | HellaSwag | 1000 | 0 | 61.3 | [58.2, 64.3] | 79.7 | [77.1, 82.1] | — | 42.1 |
| student_instruct_S1_seed1234 | ARC-Challenge | 1172 | 0 | 51.9 | [49.0, 54.7] | 53.4 | [50.6, 56.3] | — | 27.4 |
| student_instruct_S1_seed1234 | WikiText-2 | 299077 tokens / 582 windows | 0 | — | — | — | — | 6.383 | 126.0 |
| student_instruct_S1_seed42 | MMLU | 1000 | 0 | 70.0 | [67.1, 72.8] | — | — | — | 16.2 |
| student_instruct_S1_seed42 | HellaSwag | 1000 | 0 | 61.4 | [58.3, 64.4] | 80.3 | [77.7, 82.6] | — | 41.9 |
| student_instruct_S1_seed42 | ARC-Challenge | 1172 | 0 | 51.4 | [48.5, 54.2] | 52.3 | [49.4, 55.2] | — | 27.4 |
| student_instruct_S1_seed42 | WikiText-2 | 299077 tokens / 582 windows | 0 | — | — | — | — | 6.380 | 126.1 |
| student_instruct_S2_seed1234 | MMLU | 1000 | 0 | 71.4 | [68.5, 74.1] | — | — | — | 16.6 |
| student_instruct_S2_seed1234 | HellaSwag | 1000 | 0 | 61.0 | [57.9, 64.0] | 79.4 | [76.8, 81.8] | — | 42.3 |
| student_instruct_S2_seed1234 | ARC-Challenge | 1172 | 0 | 54.6 | [51.7, 57.4] | 56.7 | [53.9, 59.6] | — | 27.5 |
| student_instruct_S2_seed1234 | WikiText-2 | 299077 tokens / 582 windows | 0 | — | — | — | — | 6.674 | 126.2 |
| student_instruct_S2_seed42 | MMLU | 1000 | 0 | 70.8 | [67.9, 73.5] | — | — | — | 16.2 |
| student_instruct_S2_seed42 | HellaSwag | 1000 | 0 | 61.4 | [58.3, 64.4] | 79.5 | [76.9, 81.9] | — | 41.9 |
| student_instruct_S2_seed42 | ARC-Challenge | 1172 | 0 | 53.8 | [51.0, 56.7] | 56.3 | [53.5, 59.1] | — | 27.4 |
| student_instruct_S2_seed42 | WikiText-2 | 299077 tokens / 582 windows | 0 | — | — | — | — | 6.667 | 126.0 |
| student_instruct_S3_seed1234 | MMLU | 1000 | 0 | 70.5 | [67.6, 73.2] | — | — | — | 16.3 |
| student_instruct_S3_seed1234 | HellaSwag | 1000 | 0 | 60.7 | [57.6, 63.7] | 79.4 | [76.8, 81.8] | — | 42.1 |
| student_instruct_S3_seed1234 | ARC-Challenge | 1172 | 0 | 54.4 | [51.5, 57.2] | 56.7 | [53.8, 59.5] | — | 28.0 |
| student_instruct_S3_seed1234 | WikiText-2 | 299077 tokens / 582 windows | 0 | — | — | — | — | 6.674 | 126.3 |
| student_instruct_S3_seed42 | MMLU | 1000 | 0 | 70.3 | [67.4, 73.1] | — | — | — | 16.3 |
| student_instruct_S3_seed42 | HellaSwag | 1000 | 0 | 61.2 | [58.1, 64.2] | 79.7 | [77.1, 82.1] | — | 42.0 |
| student_instruct_S3_seed42 | ARC-Challenge | 1172 | 0 | 52.6 | [49.8, 55.5] | 55.6 | [52.8, 58.5] | — | 27.4 |
| student_instruct_S3_seed42 | WikiText-2 | 299077 tokens / 582 windows | 0 | — | — | — | — | 6.674 | 126.2 |

_In MMLU the options are single letters, hence acc_norm = acc (not shown). Primary: MMLU acc, HellaSwag/ARC-C acc_norm, WikiText-2 PPL._
