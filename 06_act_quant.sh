#!/usr/bin/env bash
# FlexPosit MICRO 2026 AE — Step 06: Table VI (WikiText-2 PPL with FP8 activation quant)
#
# Reproduces the 3-model × 5-weight-config × 2-act-config = 30-cell table showing
# FlexPosit's accuracy resilience under per-Linear FP8 (E4M3) activation quantization.
#
# Weight configs use checkpoints built by step 03:
#   FP16              -> HuggingFace model, no weight quant
#   Posit(4,1)        -> quant_out_{model}_posit4_es1/  (from step 03 baseline)
#   FlexPosit 4.1b    -> quant_out_{model}_flexposit_4.1_es1/
#   FlexPosit 4.4b    -> quant_out_{model}_flexposit_4.4_es1/
#   FlexPosit 5.0b    -> quant_out_{model}_posit5_es1/   (5-bit endpoint)
#
# Activation configs:
#   none      -> FP16 activations (headline PPL, same as Table II column)
#   fp8_e4m3  -> per-Linear FP8-E4M3 activation quantization via forward-pre hook
#
# Reference: expected/table6_act_quant.csv
# Wallclock: ~1-1.5 h on 1x A40 (eval-only; reuses step 03 checkpoints).
#
# Prerequisite: run 02_headline_ppl.sh first (produces the checkpoints).

set -euo pipefail

ENV_NAME="${ENV_NAME:-posit-ae}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/ae_work}"
RESULTS_DIR="$REPO_ROOT/results"
OUT_CSV="$RESULTS_DIR/table6_act_quant.csv"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONUNBUFFERED=1

# --- Preflight: HF access needed unless SKIP_LLAMA2=1 (llama-2-7b is the only gated model) ---
if [[ "${SKIP_LLAMA2:-0}" != "1" ]] && [[ "${HF_HUB_OFFLINE:-0}" != "1" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[preflight] ERROR: HuggingFace model access not configured." >&2
  echo "            Set HF_TOKEN, HF_HUB_OFFLINE=1, or SKIP_LLAMA2=1.  See README's 'HuggingFace model access' section." >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR"
cd "$REPO_ROOT"

if [[ ! -f "$OUT_CSV" ]]; then
  echo "model,weight_config,act_quant,wikitext2_ppl" > "$OUT_CSV"
fi

MODELS="${MODELS:-phi-2 llama-2-7b qwen2.5-7b}"
if [[ "${SKIP_LLAMA2:-0}" == "1" ]]; then
  MODELS="$(echo "$MODELS" | tr ' ' '\n' | grep -v '^llama-2-7b$' | tr '\n' ' ')"
  echo "[SKIP_LLAMA2=1] Removed llama-2-7b from Table 6 (leaves phi-2, qwen2.5-7b)"
fi
declare -A HFID SEQLEN DTYPE
HFID[phi-2]="microsoft/phi-2";               SEQLEN[phi-2]=2048;      DTYPE[phi-2]=fp16
HFID[llama-2-7b]="meta-llama/Llama-2-7b-hf"; SEQLEN[llama-2-7b]=2048; DTYPE[llama-2-7b]=fp16
HFID[qwen2.5-7b]="Qwen/Qwen2.5-7B";          SEQLEN[qwen2.5-7b]=2048; DTYPE[qwen2.5-7b]=fp16

echo "==========================================================="
echo "Table VI — WikiText-2 PPL with FP8 activation quant"
echo "==========================================================="

run_cell () {
  local model=$1 cfg=$2 ckpt=$3 act=$4 seqlen=$5 dtype=$6
  if grep -q "^${model},${cfg},${act}," "$OUT_CSV" 2>/dev/null; then
    echo "[skip] $model / $cfg / $act (already in CSV)"
    return
  fi
  echo "----------------------------------------------------------"
  echo "$model  weight=$cfg  act=$act"
  echo "----------------------------------------------------------"
  local ppl
  ppl=$(python quantization/eval_ppl.py --model "$ckpt" --act_quant "$act" --seqlen "$seqlen" --dtype "$dtype" 2>&1 \
        | tee /dev/stderr | grep "\[Result\]" | tail -1 | awk '{print $NF}')
  if [[ -z "$ppl" ]]; then
    echo "  FAILED  (no [Result] line)"
    return
  fi
  echo "  -> ppl=$ppl"
  echo "${model},${cfg},${act},${ppl}" >> "$OUT_CSV"
}

for model in $MODELS; do
  hf="${HFID[$model]}"
  sl="${SEQLEN[$model]}"
  dt="${DTYPE[$model]}"

  declare -A CKPT
  CKPT[FP16]="$hf"
  CKPT[Posit41]="$WORK_DIR/quant_out_${model}_posit4_es1"
  CKPT[FlexPosit_4.1b]="$WORK_DIR/quant_out_${model}_flexposit_4.1_es1"
  CKPT[FlexPosit_4.4b]="$WORK_DIR/quant_out_${model}_flexposit_4.4_es1"
  CKPT[FlexPosit_5.0b]="$WORK_DIR/quant_out_${model}_posit5_es1"

  for cfg in FP16 Posit41 FlexPosit_4.1b FlexPosit_4.4b FlexPosit_5.0b; do
    for act in none fp8_e4m3; do
      run_cell "$model" "$cfg" "${CKPT[$cfg]}" "$act" "$sl" "$dt"
    done
  done
done

echo
echo "==========================================================="
echo "Done. Rows: $(($(wc -l < $OUT_CSV) - 1)) / 30"
echo "CSV: $OUT_CSV"
echo "==========================================================="

python "$REPO_ROOT/quantization/verify.py" \
  --produced "$OUT_CSV" \
  --expected "$REPO_ROOT/expected/table6_act_quant.csv" \
  --key model,weight_config,act_quant \
  --produced-value wikitext2_ppl \
  --expected-value wikitext2_ppl_ours \
  --tol 0.1 \
  --label "Table 6 (FP8 activation quant)"
