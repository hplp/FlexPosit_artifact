#!/usr/bin/env bash
# FlexPosit MICRO 2026 AE — Step 01: environment setup
#
# Creates a fresh conda env `posit-ae` isolated from any existing envs.
# Idempotent: safe to re-run.
#
# Prerequisites:
#   - miniforge / miniconda installed and `conda` on PATH
#   - CUDA 11.8-compatible NVIDIA driver (`nvidia-smi` reports driver >= 520)
#   - Internet access (first run only)

set -euo pipefail

ENV_NAME="${ENV_NAME:-posit-ae}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==========================================================="
echo "FlexPosit MICRO 2026 AE — Install"
echo "==========================================================="
echo "Conda env:    $ENV_NAME"
echo

# --- Preflight ---
command -v conda >/dev/null || { echo "ERROR: conda not on PATH"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }
nvidia-smi -L

# --- Conda env ---
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[skip] Conda env '$ENV_NAME' already exists"
else
  echo "[create] Conda env '$ENV_NAME' (Python 3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

# --- Pip deps ---
echo "[pip] Installing torch stack (CUDA 11.8)"
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

echo "[pip] Installing remaining deps"
pip install -r "$REPO_ROOT/requirements.txt"

# --- Sanity ---
echo
echo "==========================================================="
echo "Sanity checks"
echo "==========================================================="
python - <<'PY'
import sys, torch, transformers, qtorch_plus, lm_eval, datasets
print(f"python           : {sys.version.split()[0]}")
print(f"torch            : {torch.__version__}")
print(f"cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"cuda device      : {name}")
    print(f"cuda mem (GB)    : {mem_gb:.1f}")
    if mem_gb < 40:
        print()
        print("  WARNING: <40 GB VRAM detected.")
        print("  Qwen2.5-14B (fp16, ~34 GB active) will not fit on this card.")
        print("  The 8 smaller models will still run. To skip Qwen2.5-14B in the")
        print("  headline sweep, invoke step 02 with:  SKIP_QWEN14B=1 bash 02_headline_ppl.sh")
print(f"transformers     : {transformers.__version__}")
print(f"qtorch_plus      : {qtorch_plus.__version__ if hasattr(qtorch_plus, '__version__') else 'installed'}")
print(f"lm_eval          : {lm_eval.__version__}")
print(f"datasets         : {datasets.__version__}")
from qtorch_plus.quant import posit_quantize, float_quantize
x = torch.randn(4, 4, device='cuda' if torch.cuda.is_available() else 'cpu')
y = posit_quantize(x, nsize=4, es=1, scale=1.0)
print(f"posit_quantize OK: input {tuple(x.shape)} -> output {tuple(y.shape)}")
PY

echo
echo "==========================================================="
echo "Install complete."
echo "Activate with:  conda activate $ENV_NAME"
echo "Next step:      bash 02_headline_ppl.sh"
echo "==========================================================="
