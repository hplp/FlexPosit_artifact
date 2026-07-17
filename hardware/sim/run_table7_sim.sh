#!/usr/bin/env bash
# Regenerate the 20 accelerator logs backing Table 7 (workload sensitivity).
# For each of the 4 (batch_size, context_length) workloads listed in the paper,
# invokes the 5 accelerator baselines: baseline (FP16), FlexPosit, BitMoD, OliVe,
# and FP16-MXFP8. Must run with cwd = sim/ (test_*.py use bare `from accelerator import ...`).
set -euo pipefail
cd "$(dirname "$0")"                                     # -> sim/
OUT="../table7_workload_sens/data"
mkdir -p "$OUT"

workloads=(
  "1 256"
  "1 8192"
  "4 4096"
  "8 8192"
)

for w in "${workloads[@]}"; do
  read -r B L <<< "$w"
  tag="B${B}_L${L}"
  echo "==== Workload B=$B L=$L ===="
  echo "[1/5] Baseline (FP16)";   python test_baseline.py    --is_generation --batch_size "$B" --context_length "$L" > "$OUT/${tag}_baseline.log"
  echo "[2/5] FlexPosit (PEB)";   python test_flexposit.py   --is_generation --batch_size "$B" --context_length "$L" > "$OUT/${tag}_flexposit.log"
  echo "[3/5] BitMoD (4b)";       python test_bitmod.py      --is_generation --batch_size "$B" --context_length "$L" > "$OUT/${tag}_bitmod.log"
  echo "[4/5] OliVe (a8w8)";      python test_olive.py       --is_generation --batch_size "$B" --context_length "$L" > "$OUT/${tag}_olive.log"
  echo "[5/5] FP16-MXFP8 (8b)";   python test_fp16_mxfp8.py  --is_generation --batch_size "$B" --context_length "$L" > "$OUT/${tag}_fp16_mxfp8.log"
done

echo "Done. Logs in $OUT"
echo "Now run: python ../table7_workload_sens/aggregate_table7.py"
