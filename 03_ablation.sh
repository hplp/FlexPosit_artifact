#!/usr/bin/env bash
# FlexPosit MICRO 2026 AE — Step 03: Table III (Qwen2.5-7B progressive ablation)
#
# Reproduces the ablation table showing progressive contribution of FlexPosit
# components on Qwen2.5-7B (WikiText-2):
#   - INT4 (per-channel range)                       ~11.87
#   - Posit(4,1) (per-channel)                       ~8.39
#   - MPQ (random) @ 4.1b                            ~8.25 (paper 8.31)
#   - MPQ (location) @ 4.1b                          ~8.36 (paper 8.37)
#   - MPQ (Fisher-ranked) @ 4.1b                     ~7.96
#   - MPQ (FlexPosit) @ 4.1b                         ~7.76
#
# Reference: expected/table3_ablation_qwen7b.csv
# Wallclock: ~1 h on 1x A40.

set -euo pipefail

ENV_NAME="${ENV_NAME:-posit-ae}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/ae_work_ablation}"
RESULTS_DIR="$REPO_ROOT/results"
OUT_CSV="$RESULTS_DIR/table3_ablation_qwen7b.csv"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONUNBUFFERED=1

mkdir -p "$WORK_DIR" "$RESULTS_DIR"
cd "$REPO_ROOT"

if [[ ! -f "$OUT_CSV" ]]; then
  echo "method,granularity,target_avg_bits,wikitext2_ppl,notes" > "$OUT_CSV"
fi

MODEL="qwen2.5-7b"
HF_ID="Qwen/Qwen2.5-7B"
SENS_CSV="$REPO_ROOT/sensitivity/qwen2.5-7b.csv"
BASE4_DIR="$WORK_DIR/quant_out_${MODEL}_posit4_es1"
INT4_DIR="$WORK_DIR/quant_out_${MODEL}_int4_pc"

echo "==========================================================="
echo "Table III — Qwen2.5-7B progressive ablation"
echo "==========================================================="

# --- (1) INT4 per-channel (range scale) ---
if [[ ! -f "$INT4_DIR/metrics.json" ]]; then
  echo "[build] INT4 per-channel (range)"
  python quantization/quantize_int4.py \
    --model "$HF_ID" --dtype fp16 --seqlen 2048 \
    --save_dir "$INT4_DIR"
fi
python - <<PY
import json, csv
p = "$INT4_DIR/metrics.json"
d = json.load(open(p))
ppl = d.get("ppl") or d.get("wikitext2_ppl")
with open("$OUT_CSV", "a", newline="") as f:
    csv.writer(f).writerow(["int4","per_channel_range","4.0",f"{ppl:.4f}","per-channel INT4 range scale search"])
print(f"[INT4]        ppl={ppl:.4f}")
PY

# --- (2) Posit(4,1) per-channel — base checkpoint ---
if [[ ! -f "$BASE4_DIR/quant_log.json" ]]; then
  echo "[build] Posit(4,1) base"
  python quantization/posit_quantize.py \
    --model "$MODEL" --dtype fp16 --device cuda \
    --weight_format posit4 --nsize 4 --es_candidates 1 \
    --log2_min -8 --log2_max 9 --ch_batch 64 \
    --ppl_seqlen 2048 --save_dir "$BASE4_DIR"
fi
python - <<PY
import json, csv
p = "$BASE4_DIR/metrics.json"
d = json.load(open(p))
ppl = d.get("ppl") or d.get("wikitext2_ppl")
with open("$OUT_CSV", "a", newline="") as f:
    csv.writer(f).writerow(["posit41","per_channel","4.0",f"{ppl:.4f}","Posit(4,1) per-channel PoT scale"])
print(f"[Posit(4,1)]  ppl={ppl:.4f}")
PY

# --- (3, 4, 5) MPQ strategies via sweep mode @ 4.1b ---
for strat in random location sensitivity; do
  OUT="$WORK_DIR/mpq_${strat}_4p1"
  if [[ ! -f "$OUT/ppl_vs_avg_bits.csv" ]]; then
    echo "[build] MPQ ($strat) @ 4.1b"
    python quantization/flexposit_mpq.py \
      --base_dir "$BASE4_DIR" \
      --fp32_reference_dir "$HF_ID" \
      --sensitivity_csv "$SENS_CSV" \
      --out_dir "$OUT" \
      --sweep_bits_start 4.0 --sweep_bits_end 4.1 --sweep_bits_step 0.1 \
      --sweep_strategy "$strat" \
      --random_seed 20250925 \
      --override_nsize 5 --es_candidates 1 \
      --base_bits 4 --upgrade_bits 5 \
      --log2_min -8 --log2_max 9 \
      --seqlen 2048
  fi
  python - <<PY
import csv
with open("$OUT/ppl_vs_avg_bits.csv") as f:
    rows = list(csv.DictReader(f))
row = rows[-1]  # target 4.1
ppl = float(row["ppl"])
name = {"random":"mpq_random","location":"mpq_location","sensitivity":"mpq_flexposit"}["$strat"]
label = {"random":"MPQ random selection","location":"MPQ by layer position","sensitivity":"FlexPosit (Δ-PPL-ranked)"}["$strat"]
with open("$OUT_CSV", "a", newline="") as f:
    csv.writer(f).writerow([name,"channel_window","4.1",f"{ppl:.4f}",label])
print(f"[MPQ $strat]  ppl={ppl:.4f}")
PY
done

# NOTE: MPQ (Fisher-ranked) at 4.1b — requires Fisher CSV computation first
# (see reviewerE_fisher_window_sensitivity.py). Ship as reference-only in
# expected/table3_ablation_qwen7b.csv.

echo
echo "==========================================================="
echo "Done. See $OUT_CSV"
echo "==========================================================="

python "$REPO_ROOT/quantization/verify.py" \
  --produced "$OUT_CSV" \
  --expected "$REPO_ROOT/expected/table3_ablation_qwen7b.csv" \
  --key method \
  --produced-value wikitext2_ppl \
  --expected-value wikitext2_ppl \
  --tol 0.1 \
  --label "Table 3 (Qwen-7B ablation)"
