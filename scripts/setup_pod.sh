#!/usr/bin/env bash
# RunPod environment setup (idempotent).
# Usage: bash scripts/setup_pod.sh
set -euo pipefail
# The image ships with HF_HUB_ENABLE_HF_TRANSFER=1; hf_transfer is absent from the venv -> downloads raise ValueError.
export HF_HUB_ENABLE_HF_TRANSFER=0

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venv}"
SYS_PY="${SYS_PY:-python3}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
EXPECTED_TORCH="2.8.0"

cd "$REPO_DIR"
mkdir -p env results/day1

# 1) Verify the system (image) torch -------------------------------------------
SYS_TORCH="$("$SYS_PY" -c 'import torch; print(torch.__version__)')"
echo "[setup] image torch: $SYS_TORCH"
case "$SYS_TORCH" in
  "$EXPECTED_TORCH"|"$EXPECTED_TORCH"+*) ;;
  *) echo "[ERROR] Image torch is $SYS_TORCH, expected $EXPECTED_TORCH." >&2; exit 1 ;;
esac

# 2) venv (create if missing) --------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] creating venv: $VENV"
  "$SYS_PY" -m venv --system-site-packages "$VENV"
else
  echo "[setup] using existing venv: $VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

# 3) Constrain the torch family: prevents pip from replacing torch -------------
"$PY" - > env/constraints_torch.txt <<'EOF'
from importlib import metadata
for p in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        print(f"{p}=={metadata.version(p)}")
    except metadata.PackageNotFoundError:
        pass
EOF
echo "[setup] constraints:"; sed 's/^/    /' env/constraints_torch.txt

"$PY" -m pip install -r requirements.txt -c env/constraints_torch.txt
"$PY" -m pip install -q -e . --no-deps

# 4) Post-install verification ------------------------------------------------
"$PY" - "$EXPECTED_TORCH" <<'EOF'
import sys, torch
exp = sys.argv[1]
v = torch.__version__
if v.split("+")[0] != exp:
    sys.exit(f"[ERROR] torch changed: {v} (expected {exp}). Delete the venv and "
             "check the pinned versions in requirements.txt.")
if not torch.cuda.is_available():
    sys.exit("[ERROR] CUDA unavailable (torch.cuda.is_available() == False).")
print(f"[setup] torch {v} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
EOF

"$PY" -m pip check || echo "[WARNING] pip check reported dependency issues (see above)."
"$PY" -m pip freeze > env/pip_freeze_pod.txt
echo "[setup] wrote env/pip_freeze_pod.txt."

# 5) Cache GSM8K (small; lets check_models.py run offline) --------------------
"$PY" -c "from datasets import load_dataset; d = load_dataset('openai/gsm8k', 'main'); print('[setup] gsm8k:', {k: len(v) for k, v in d.items()})"

# 6) Version summary ----------------------------------------------------------
"$PY" - <<'EOF'
from importlib import metadata
import sys
print("[setup] python", sys.version.split()[0])
for p in ("torch", "vllm", "transformers", "peft", "accelerate", "datasets",
          "tokenizers", "huggingface-hub", "numpy", "xformers", "flash-attn"):
    try:
        print(f"    {p:16s} {metadata.version(p)}")
    except metadata.PackageNotFoundError:
        print(f"    {p:16s} -")
EOF
echo "[setup] done. To activate: source $VENV/bin/activate"
