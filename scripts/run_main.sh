#!/usr/bin/env bash
# MAIN RUNS (prereg §3): S1-42, S2-42, S3-42, S1-1234, S2-1234, S3-1234.
# lr is read from configs/base.yaml train.lr (FROZEN after the pilot); aborts if null.
# After each training run: GSM8K test 0-shot and 4-shot (vLLM, with adapter) + general benchmarks.
# Steps whose output exists are skipped. Log: results/train/run_main.log
#   nohup bash $WORKSPACE/KD-QWEN-14B-7B/scripts/run_main.sh > $WORKSPACE/run_main.out 2>&1 &
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0
WORKSPACE="${WORKSPACE:-/workspace}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
VENV="${VENV:-$WORKSPACE/venv}"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "[ERROR] venv not found: $VENV (run scripts/setup_pod.sh first)" >&2; exit 1
fi
PY="${PY:-python}"
RUNS="${RUNS:-$WORKSPACE/runs}"
OUT=results/train
LOG=$OUT/run_main.log
mkdir -p "$OUT" "$RUNS"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
step() {
  local name=$1 marker=$2; shift 2
  if [ -n "$marker" ] && [ -e "$marker" ]; then log "SKIP   $name ($marker exists)"; return 0; fi
  log "START  $name"; local t0=$SECONDS
  if "$@" 2>&1 | tee -a "$LOG"; then log "DONE   $name ($((SECONDS - t0)) s)"
  else local rc=$?; log "ERROR  $name (exit $rc, $((SECONDS - t0)) s)"; exit "$rc"; fi
}

# Frozen-lr check (explicit error if null)
LR="$("$PY" - <<'EOF'
import sys, yaml
lr = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))["train"].get("lr")
if lr is None:
    sys.exit("[ERROR] configs/base.yaml train.lr is not frozen (null). Set it after the pilot (and record it in docs/decisions.md).")
print(lr)
EOF
)" || { echo "$LR" >&2; exit 1; }

log "=== run_main started (git $(git rev-parse --short HEAD 2>/dev/null || echo '?'), lr=$LR) ==="

for seed in 42 1234; do
  for c in S1 S2 S3; do
    name="${c}_seed${seed}"
    step "train $name" "$RUNS/$name/summary.json" \
      "$PY" scripts/train.py --cfg "$c" --seed "$seed"
    for shots in 0 4; do
      step "gsm8k test ${shots}-shot $name" "$OUT/$name/gsm8k_test_${shots}shot.summary.json" \
        "$PY" scripts/eval_gsm8k.py --model student_instruct --split test --shots "$shots" --engine vllm \
          --adapter "$RUNS/$name/adapter" --out "$OUT/$name/gsm8k_test_${shots}shot"
    done
    step "general benchmarks $name" "results/general/student_instruct_${name}/wikitext2.summary.json" \
      "$PY" scripts/eval_general.py --model student_instruct --tasks all --adapter "$RUNS/$name/adapter"
  done
done

step "general_report" "" "$PY" scripts/general_report.py
log "=== run_main finished ==="
