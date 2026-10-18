#!/usr/bin/env bash
# Baseline (Day 1) measurement: model check -> splits -> engine cross-check ->
# {T, S0-base, S0-Instruct} x {0,4}-shot -> gap report.
# Steps whose output exists are skipped (resumable). Log: results/day1/run_day1.log
# Self-contained (sets up the venv and environment variables itself), e.g.:
#   nohup bash $WORKSPACE/KD-QWEN-14B-7B/scripts/run_day1_baseline.sh > $WORKSPACE/run_day1.out 2>&1 &
set -euo pipefail
# The image ships with HF_HUB_ENABLE_HF_TRANSFER=1; hf_transfer is absent from the venv -> downloads raise ValueError.
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

OUT=results/day1
EVAL_DIR=$OUT/eval
LOG=$OUT/run_day1.log
mkdir -p "$EVAL_DIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# step <name> <skip-marker-file|""> <command...>
step() {
  local name=$1 marker=$2; shift 2
  if [ -n "$marker" ] && [ -e "$marker" ]; then
    log "SKIP   $name ($marker exists)"; return 0
  fi
  log "START  $name"
  local t0=$SECONDS
  if "$@" 2>&1 | tee -a "$LOG"; then
    log "DONE   $name ($((SECONDS - t0)) s)"
  else
    local rc=$?
    log "ERROR  $name (exit $rc, $((SECONDS - t0)) s)"; exit "$rc"
  fi
}

log "=== run_day1_baseline started (git $(git rev-parse --short HEAD 2>/dev/null || echo '?')) ==="
step "check_models"      ""                          "$PY" scripts/check_models.py
step "make_splits"       data/splits/val.json        "$PY" scripts/make_splits.py
step "engine_crosscheck" $OUT/engine_crosscheck.json "$PY" scripts/engine_crosscheck.py

for model in student_base student_instruct teacher; do
  for shots in 0 4; do
    step "eval $model ${shots}-shot" "$EVAL_DIR/${model}_${shots}shot_vllm.summary.json" \
      "$PY" scripts/eval_gsm8k.py --model "$model" --shots "$shots" --engine vllm
  done
done

step "rescore"           ""                          "$PY" scripts/rescore.py
step "gap_report"        ""                          "$PY" scripts/gap_report.py
log "=== run_day1_baseline finished ==="
