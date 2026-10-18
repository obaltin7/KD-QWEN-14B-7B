# Hyperparameter pilot (S1, val 300, 0-shot flexible)

Full run: 898 steps (n=7173, effective batch 16, 2 epochs).

## Val accuracy (per lr)

| Run | lr | flexible % | 95% CI | strict % | format error % | truncated at 512 % | final loss | NaN | diverged |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| S1_seed42_lr1e-4 | 0.0001 | 85.7 | [81.2, 89.2] | 85.7 | 0.0 | 0.0 | 0.0807 | False | False |
| S1_seed42_lr2e-4 | 0.0002 | 84.0 | [79.4, 87.7] | 84.0 | 0.0 | 0.0 | 0.0631 | False | False |

## Smoke tests (20 steps, lr 1e-4)

| Cfg | memory GB | tok/s | steps/s | first 5 CE | first 5 KL | first 5 feat | β·feat/CE | full-run estimate |
|---|---:|---:|---:|---|---|---|---|---:|
| S1 | 17.3 | 1692 | 0.664 | 1.103, 1.146, 0.950, 1.159, 1.104 | —, —, —, —, — | —, —, —, —, — | —, —, —, —, — | 23 min |
| S2 | 48.6 | 1222 | 0.480 | 1.103, 1.146, 0.949, 1.156, 1.110 | 2.308, 1.923, 2.293, 2.457, 2.564 | —, —, —, —, — | —, —, —, —, — | 31 min |
| S3 | 49.7 | 1164 | 0.458 | 1.103, 1.146, 0.947, 1.157, 1.104 | 2.308, 1.923, 2.299, 2.455, 2.560 | 1.998, 1.999, 1.997, 1.993, 1.989 | 1.81, 1.74, 2.11, 1.72, 1.80 | 33 min |

## Stability rule (prereg §4)

- NaN/inf: **none**
- Divergence (last 10% > first 10% loss): **none**
- β·L_feat / CE (S3, first 5 steps, max): **2.11** -> below 10x; τ, α, β defaults (2, 0.5, 1) unchanged

## Duration estimate (main runs)

- Training (estimated from smoke tests): S1 ~23 min, S2 ~31 min, S3 ~33 min
- Evaluation per run: ~8 min (GSM8K test 0/4-shot + general benchmarks)
- **Total ~3.6 h** (6 runs)

## Recommendation

**RECOMMENDATION:** lr = `0.0001` (S1_seed42_lr1e-4); highest val flexible accuracy (ties go to the smaller lr). Number of epochs: 2 (prereg). The frozen value is recorded in `configs/base.yaml` -> `train.lr` and in `docs/decisions.md`.
