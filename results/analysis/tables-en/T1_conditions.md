| Item | Value |
|---|---|
| Teacher (T) | Qwen/Qwen2.5-14B-Instruct (frozen, bf16) |
| Student (S) | Qwen/Qwen2.5-7B-Instruct + LoRA |
| Training data | GSM8K train_rest, n = 7173 |
| Longest example / truncated | 446 tokens / 0 |
| Test | GSM8K test, n = 1319 |
| Prompt | Question: {question}\nAnswer: — no chat template |
| LoRA | r = 16, alpha = 32, dropout = 0.05, targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable LoRA parameters | 40,370,176 |
| Optimisation | AdamW (wd = 0), lr = 0.0001, 3% linear warm-up + cosine, gradient clipping 1.0 |
| Batch | effective 16 (micro 4) |
| Epochs / steps | 2 / 898 (warm-up 27) |
| Sequence length | at most 512 |
| KD (S2, S3) | τ = 2, α = 0.5, forward KL, first 151,665 vocabulary indices |
| Hidden state (S3) | β = 1, pairs 7↔12, 14↔24, 21↔36, 28↔48, W: 3584→5120 (shared) |
| Seeds | 42, 1234 |
| Time / peak memory — S1 (seed 42) | 22.0 min / 17.5 GB |
| Time / peak memory — S2 (seed 42) | 30.1 min / 49.2 GB |
| Time / peak memory — S3 (seed 42) | 31.9 min / 50.2 GB |
| Time / peak memory — S1 (seed 1234) | 21.7 min / 17.4 GB |
| Time / peak memory — S2 (seed 1234) | 30.5 min / 49.1 GB |
| Time / peak memory — S3 (seed 1234) | 31.5 min / 50.1 GB |
