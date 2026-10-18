# KD-QWEN-14B-7B

Knowledge distillation from **Qwen2.5-14B-Instruct** to **Qwen2.5-7B-Instruct** on GSM8K, comparing
cross-entropy fine-tuning with logit-level and logit + intermediate-layer distillation under a
pre-registered protocol. A summary of the outcomes is given in [`RESULTS.md`](RESULTS.md).

- **Teacher (T):** `Qwen/Qwen2.5-14B-Instruct` (frozen, bf16, forward passes only)
- **Student (S):** `Qwen/Qwen2.5-7B-Instruct` with LoRA (r = 16, α = 32, dropout 0.05; `q, k, v, o, gate, up, down`)
- **Task:** GSM8K (`openai/gsm8k`, `main`); general benchmarks MMLU, HellaSwag, ARC-Challenge, WikiText-2

| Configuration | Loss |
|---|---|
| S1 | CE |
| S2 | (1 − α)·CE + α·τ²·KL(p_T^τ ‖ p_S^τ), τ = 2, α = 0.5, forward KL over the first V′ = 151 665 vocabulary entries (fp32) |
| S3 | S2 + β·(1/4)·Σ MSE(LN(W·h_S^(l)), LN(h_T^(m))), β = 1; layer pairs (7, 12), (14, 24), (21, 36), (28, 48); shared linear projection 3584 → 5120; affine-free LayerNorm; answer positions only |

All configurations share data, order, LoRA setup, learning rate (1e-4, 3% linear warm-up + cosine),
2 epochs (898 steps), effective batch 16 (4 × 4), maximum sequence length 512 and seeds 42 / 1234; the
loss function is the only difference. Evaluation: greedy decoding, `max_new_tokens = 512`, 0-shot (primary)
and 4-shot. All settings live in a single file: [`configs/base.yaml`](configs/base.yaml).

## Repository layout

```
configs/base.yaml              single configuration file
src/kdq/                       package: config, data, prompts, answer extraction, evaluation, general
                               benchmarks, distillation (losses, training loop), statistics, analysis
scripts/setup_pod.sh           environment setup on the GPU pod (idempotent)
scripts/check_models.py        config + tokenizer compatibility check (no weights loaded)
scripts/make_splits.py         GSM8K val / train_rest splits and 4-shot exemplar selection
scripts/eval_gsm8k.py          GSM8K evaluation (vllm | hf), question-level JSONL + summary
scripts/engine_crosscheck.py   vllm ↔ hf agreement check
scripts/rescore.py             re-scores existing JSONL outputs with answer extraction v2 (no GPU)
scripts/gap_report.py          teacher–student gap report and student-selection rule
scripts/make_general_subsets.py  MMLU / HellaSwag subsets and dataset cache
scripts/eval_general.py        general benchmarks (log-likelihood), optional --adapter
scripts/general_report.py      model × task table of the general benchmarks
scripts/train.py               S1 / S2 / S3 LoRA training
scripts/pilot_report.py        hyperparameter pilot report
scripts/analyze.py             pre-registered analyses, figures and tables (CPU, local)
scripts/run_*.sh               end-to-end pipelines (baseline, general benchmarks, pilot, main runs)
tests/                         unit tests (pytest; CPU, tiny synthetic Qwen2 models)
data/splits/                   split indices with sha256 digests
results/                       small result files (json / jsonl / md / csv / figures)
docs/preregistration.md        pre-registration (prereg-v1.0)
docs/decisions.md              decision log and documented deviations
env/pip_freeze_pod.txt         package versions of the GPU environment
```

## Reproducing the experiments (1× A100-80GB, torch 2.8.0 / CUDA 12.8)

All absolute paths derive from `WORKSPACE` (default `/workspace`); `VENV`, `HF_HOME` and `RUNS` can be set
individually.

```bash
export WORKSPACE=/workspace
cd "$WORKSPACE" && git clone <repository-url> KD-QWEN-14B-7B && cd KD-QWEN-14B-7B

bash scripts/setup_pod.sh                       # venv + packages, torch check, GSM8K cache
source "$WORKSPACE/venv/bin/activate"
pytest -q

bash scripts/run_day1_baseline.sh               # model check -> splits -> cross-check -> baseline evals -> gap report
bash scripts/run_general_baseline.sh            # general benchmarks for T and S0
bash scripts/run_pilot.sh                       # smoke tests + S1 learning-rate pilot on val
bash scripts/run_main.sh                        # 6 main runs (3 configurations × 2 seeds) + evaluations
```

The scripts resume where they stopped and can be started with `nohup ... &`. Adapters and checkpoints are
written to `$WORKSPACE/runs/` (not tracked); logs and summaries to `results/train/`.

> The RunPod image sets `HF_HUB_ENABLE_HF_TRANSFER=1` without installing `hf_transfer`; all shell scripts
> therefore set `HF_HUB_ENABLE_HF_TRANSFER=0`.

`check_models.py` exit codes: 0 = critical checks passed, 1 = mismatch, 2 = missing from cache. eos/pad
differences are warnings by default and errors with `--strict`.

## Analysis (local, CPU)

```bash
python scripts/analyze.py
```

Reads only `results/`, is deterministic, and writes `results/analysis/{findings.md, analysis.json}` plus
figures and tables in English and Turkish (`figures-en/`, `figures-tr/`, `tables-en/`, `tables-tr/`).

## Local development

No GPU or model weights are needed; the training tests run on CPU with tiny synthetic Qwen2 models:

```bash
python -m venv .venv && .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install transformers==4.56.2 peft==0.17.1 accelerate==1.10.1 safetensors pyyaml numpy matplotlib pytest
.venv/Scripts/python -m pytest -q          # slow end-to-end training tests: pytest -m slow
```

## Data and models

The datasets are downloaded from the Hugging Face Hub; `data/splits/` stores only indices and digests. The
files under `results/` contain question texts, reference answers and model generations derived from these
datasets, which remain subject to their original licenses:

| Resource | Hub ID | License | Reference |
|---|---|---|---|
| GSM8K | `openai/gsm8k` | MIT | Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems* |
| MMLU | `cais/mmlu` | MIT | Hendrycks et al. (2021), *Measuring Massive Multitask Language Understanding* |
| HellaSwag | `Rowan/hellaswag` | MIT | Zellers et al. (2019), *HellaSwag: Can a Machine Really Finish Your Sentence?* |
| ARC-Challenge | `allenai/ai2_arc` | CC BY-SA 4.0 | Clark et al. (2018), *Think you have Solved Question Answering? Try ARC* |
| WikiText-2 | `Salesforce/wikitext` | CC BY-SA 3.0 | Merity et al. (2017), *Pointer Sentinel Mixture Models* |
| Qwen2.5-14B-Instruct | `Qwen/Qwen2.5-14B-Instruct` | Apache 2.0 | Qwen Team (2024), *Qwen2.5 Technical Report* |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | Apache 2.0 | Qwen Team (2024), *Qwen2.5 Technical Report* |

Content derived from ARC-Challenge and WikiText-2 is redistributed under the respective CC BY-SA terms. No
model weights or trained adapters are included in this repository.

## License

The code in this repository is released under the Apache License 2.0; see [`LICENSE`](LICENSE). Third-party
datasets and models are subject to the licenses listed above.
