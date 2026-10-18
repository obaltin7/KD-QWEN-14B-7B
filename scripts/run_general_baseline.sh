#!/usr/bin/env bash
# General-benchmark baseline: subsets -> smoke test (student_instruct, --limit 20)
# -> student_instruct full -> teacher full -> report.
# Steps whose output exists are skipped (resumable). Log: results/general/run_general.log
# Self-contained (sets up the venv and environment variables itself), e.g.:
#   nohup bash $WORKSPACE/KD-QWEN-14B-7B/scripts/run_general_baseline.sh > $WORKSPACE/run_general.out 2>&1 &
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

OUT=results/general
LOG=$OUT/run_general.log
mkdir -p "$OUT"

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

log "=== run_general_baseline started (git $(git rev-parse --short HEAD 2>/dev/null || echo '?')) ==="
step "make_general_subsets"     data/splits/general_manifest.json \
  "$PY" scripts/make_general_subsets.py
# eval_general.py itself skips tasks that already have a summary (does not load the model if all do).
step "smoke student_instruct"   $OUT/student_instruct_limit20/wikitext2.summary.json \
  "$PY" scripts/eval_general.py --model student_instruct --tasks all --limit 20
step "student_instruct full"    $OUT/student_instruct/wikitext2.summary.json \
  "$PY" scripts/eval_general.py --model student_instruct --tasks all
step "teacher full"             $OUT/teacher/wikitext2.summary.json \
  "$PY" scripts/eval_general.py --model teacher --tasks all
step "general_report"           "" \
  "$PY" scripts/general_report.py
log "=== run_general_baseline finished ==="
