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

# Isolate this install from any packages the user has under ~/.local
# (a common trap: ~/.local/bin/pip shadows the env's pip, and ~/.local
# site-packages get preferred over the env's site-packages).
export PYTHONNOUSERSITE=1
export PIP_USER=0

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

# --- nvcc / GPU-arch check (warn only) ---
# PyTorch's cpp_extension JIT-compiles qtorch_plus CUDA kernels for the local
# GPU on first import. The nvcc that gets invoked must support the local GPU's
# compute capability (CUDA >= 11.8 for Ada compute_89, >= 12.0 for Hopper
# compute_90, >= 12.8 for Blackwell compute_100). If the nvcc on PATH is too
# old, warn the reviewer and point at CUDA_HOME as the fix; do not override.
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -n1 | tr -d '. ')"
if [[ -n "$GPU_CC" ]]; then
  NVCC_ON_PATH="$(command -v nvcc || true)"
  nvcc_ok=1
  if [[ -z "$NVCC_ON_PATH" ]]; then
    nvcc_ok=0
  elif ! "$NVCC_ON_PATH" --list-gpu-arch 2>/dev/null | grep -qx "compute_$GPU_CC"; then
    # Fallback for old nvcc without --list-gpu-arch: try a trivial compile.
    if ! echo "" | "$NVCC_ON_PATH" -x cu -arch=sm_$GPU_CC -ptx -o /dev/null - 2>/dev/null; then
      nvcc_ok=0
    fi
  fi
  if [[ "$nvcc_ok" != "1" ]]; then
    echo "[warn] nvcc on PATH (${NVCC_ON_PATH:-none}) does not appear to support"
    echo "[warn] your GPU's compute capability ${GPU_CC:0:1}.${GPU_CC:1:1}."
    echo "[warn] qtorch_plus's JIT extension build will fail (nvcc fatal:"
    echo "[warn] Unsupported gpu architecture 'compute_$GPU_CC')."
    echo "[warn] Fix: export CUDA_HOME=/path/to/newer/cuda and PATH=\$CUDA_HOME/bin:\$PATH"
    echo "[warn] before re-running 01_install.sh. Required minimums:"
    echo "[warn]   Ada (8.9)      -> CUDA >= 11.8"
    echo "[warn]   Hopper (9.0)   -> CUDA >= 12.0"
    echo "[warn]   Blackwell (10) -> CUDA >= 12.8"
  fi
fi

# --- Conda env ---
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[skip] Conda env '$ENV_NAME' already exists"
else
  echo "[create] Conda env '$ENV_NAME' (Python 3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

# --- Verify the env is actually the active Python ---
# Guard against the common failure where `pip` on PATH is shadowed by
# ~/.local/bin/pip (which installs into ~/.local, not the env).
PY_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
if [[ "$PY_PREFIX" != "$CONDA_PREFIX" ]]; then
  echo "ERROR: python's sys.prefix ($PY_PREFIX) != CONDA_PREFIX ($CONDA_PREFIX)"
  echo "       Check your PATH; conda activation is not reaching python."
  exit 1
fi
PIP_PATH="$(command -v pip || true)"
if [[ "$PIP_PATH" != "$CONDA_PREFIX/bin/pip" ]]; then
  echo "[warn] 'pip' resolves to $PIP_PATH, not $CONDA_PREFIX/bin/pip"
  echo "[warn] Using 'python -m pip' below to force the env's pip."
fi

# --- Pip deps ---
# Always invoke via `python -m pip` so we use the active interpreter's pip
# regardless of what `pip` resolves to on PATH.
echo "[pip] Installing torch stack (CUDA 11.8)"
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

echo "[pip] Installing remaining deps"
python -m pip install -r "$REPO_ROOT/requirements.txt"

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

# --- Build CACTI from source ---
# The bundled CACTI binary in hardware/sim/mem/cacti/ is built against
# glibc >= 2.29 and may not run on older distros (RHEL/Rocky 8, Ubuntu 18.04).
# We rebuild it here against the host's local glibc so 05_hardware.sh's
# Table 10 regeneration works everywhere.
echo
echo "==========================================================="
echo "Build CACTI from source"
echo "==========================================================="
CACTI_DIR="$REPO_ROOT/hardware/sim/mem/cacti"
if [[ -d "$CACTI_DIR/src" ]]; then
  if ( cd "$CACTI_DIR/src" && make -j4 >/dev/null 2>&1 ); then
    cp -f "$CACTI_DIR/src/cacti" "$CACTI_DIR/cacti"
    echo "[cacti] Built from source (host glibc) and installed to $CACTI_DIR/cacti"
  else
    echo "[cacti] WARN: build failed; keeping the bundled binary (may fail on older glibc)"
  fi
else
  echo "[cacti] No src/ directory found; keeping the bundled binary"
fi

# --- Build Ramulator2 from source ---
# The bundled ramulator2 / libramulator.so were built against glibc >= 2.32
# and libstdc++ >= GLIBCXX_3.4.29, which are absent on RHEL/Rocky 8.
# We rebuild here with -static-libstdc++ -static-libgcc so the resulting
# binaries only depend on glibc >= 2.17 (portable to any Linux from 2013+).
# Ramulator uses C++20 <ranges>, so we need g++ >= 10.
echo
echo "==========================================================="
echo "Build Ramulator2 from source"
echo "==========================================================="
RAMU_DIR="$REPO_ROOT/hardware/ramulator"
RAMU_SRC="$RAMU_DIR/src_build"
if [[ -f "$RAMU_SRC/CMakeLists.txt" ]]; then
  # Pick a g++ >= 10 (C++20 <ranges> requirement). Prefer $CXX if user set it.
  RAMU_CXX="${CXX:-g++}"
  RAMU_CC="${CC:-gcc}"
  cxx_major=0
  if command -v "$RAMU_CXX" >/dev/null 2>&1; then
    cxx_major=$("$RAMU_CXX" -dumpversion 2>/dev/null | cut -d. -f1)
    cxx_major=${cxx_major:-0}
  fi
  if (( cxx_major < 10 )); then
    echo "[ramulator] WARN: $RAMU_CXX is version ${cxx_major:-unknown}; need >= 10 for C++20 <ranges>."
    echo "[ramulator]       Set CC/CXX to a newer compiler (e.g. gcc-11) and re-run 01_install.sh"
    echo "[ramulator]       to enable Table 10 regeneration. Keeping the bundled binary for now."
  else
    RAMU_BUILD="$RAMU_SRC/build_ae"
    rm -rf "$RAMU_BUILD"
    if ( mkdir -p "$RAMU_BUILD" && cd "$RAMU_BUILD" && \
         CC="$RAMU_CC" CXX="$RAMU_CXX" \
         cmake -DCMAKE_CXX_FLAGS='-static-libstdc++ -static-libgcc' \
               -DCMAKE_EXE_LINKER_FLAGS='-static-libstdc++ -static-libgcc' \
               -DCMAKE_SHARED_LINKER_FLAGS='-static-libstdc++ -static-libgcc' \
               -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
               "$RAMU_SRC" >/dev/null 2>&1 && \
         make -j4 >/dev/null 2>&1 ); then
      # ramulator2 lands in $RAMU_BUILD; libramulator.so lands in $RAMU_SRC
      if [[ -f "$RAMU_BUILD/ramulator2" && -f "$RAMU_SRC/libramulator.so" ]]; then
        cp -f "$RAMU_BUILD/ramulator2" "$RAMU_DIR/ramulator2"
        cp -f "$RAMU_SRC/libramulator.so" "$RAMU_DIR/libramulator.so"
        echo "[ramulator] Built from source (static libstdc++) and installed to $RAMU_DIR/"
      else
        echo "[ramulator] WARN: build finished but expected artifacts missing; keeping bundled binary"
      fi
    else
      echo "[ramulator] WARN: build failed; keeping the bundled binary (may fail on older glibc/libstdc++)"
    fi
  fi
else
  echo "[ramulator] No src_build/ directory found; keeping the bundled binary"
fi

echo
echo "==========================================================="
echo "Install complete."
echo "Activate with:  conda activate $ENV_NAME"
echo "Next step:      bash 02_headline_ppl.sh"
echo "==========================================================="
