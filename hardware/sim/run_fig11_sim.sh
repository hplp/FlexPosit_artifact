#!/usr/bin/env bash
# Regenerate the 5 accelerator logs that Figure 11 is built from.
# Must run with cwd = sim/ (the test_*.py use bare `from accelerator import ...`).
set -euo pipefail
cd "$(dirname "$0")"                      # -> sim/
OUT="../fig11_hardware_metrics/data"
mkdir -p "$OUT"

echo "[1/5] Baseline (FP16)";        python test_baseline.py    --is_generation > "$OUT/test_baseline.log"
echo "[2/5] FlexPosit (PEB)";        python test_flexposit.py   --is_generation > "$OUT/test_flexposit.log"
echo "[3/5] BitMoD (4b)";            python test_bitmod.py      --is_generation > "$OUT/test_bitmod.log"
echo "[4/5] OliVe (a8w8)";           python test_olive.py       --is_generation > "$OUT/test_olive.log"
echo "[5/5] FP16-MXFP8 (8b)";        python test_fp16_mxfp8.py  --is_generation > "$OUT/test_fp16_mxfp8.log"

echo "Done. Logs in $OUT"
echo "Now run: python ../fig11_hardware_metrics/plot_fig11.py"
