#!/usr/bin/env bash
# Hyperparameter pilot (prereg §4): smoke tests -> S1 lr=1e-4 (val) -> S1 lr=2e-4 (val) -> report.
# Steps whose output exists are skipped. Log: results/train/pilot/run_pilot.log
#   nohup bash $WORKSPACE/KD-QWEN-14B-7B/scripts/run_pilot.sh > $WORKSPACE/run_pilot.out 2>&1 &
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
PILOT=results/train/pilot
LOG=$PILOT/run_pilot.log
mkdir -p "$PILOT" "$RUNS"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
step() {
  local name=$1 marker=$2; shift 2
  if [ -n "$marker" ] && [ -e "$marker" ]; then log "SKIP   $name ($marker exists)"; return 0; fi
  log "START  $name"; local t0=$SECONDS
  if "$@" 2>&1 | tee -a "$LOG"; then log "DONE   $name ($((SECONDS - t0)) s)"
  else local rc=$?; log "ERROR  $name (exit $rc, $((SECONDS - t0)) s)"; exit "$rc"; fi
}

log "=== run_pilot started (git $(git rev-parse --short HEAD 2>/dev/null || echo '?')) ==="

# (a) smoke tests: 20 steps, seed 42, lr 1e-4
for c in S1 S2 S3; do
  step "smoke $c" "$RUNS/smoke_${c}_seed42/summary.json" \
    "$PY" scripts/train.py --cfg "$c" --seed 42 --lr 1e-4 --max-steps 20 \
      --out "$RUNS/smoke_${c}_seed42" --results-dir "$PILOT/smoke_${c}_seed42"
done
step "smoke_report" "" "$PY" scripts/pilot_report.py --smoke-only

# (b), (c) S1 full training + val (0-shot flexible)
for lr in 1e-4 2e-4; do
  name="S1_seed42_lr${lr}"
  step "train $name" "$RUNS/$name/summary.json" \
    "$PY" scripts/train.py --cfg S1 --seed 42 --lr "$lr" --results-dir "$PILOT/$name"
  step "val $name" "$PILOT/${name}_val0shot.summary.json" \
    "$PY" scripts/eval_gsm8k.py --model student_instruct --split val --shots 0 --engine vllm \
      --adapter "$RUNS/$name/adapter" --out "$PILOT/${name}_val0shot"
done

# (d) report
step "pilot_report" "" "$PY" scripts/pilot_report.py
log "=== run_pilot finished ==="
