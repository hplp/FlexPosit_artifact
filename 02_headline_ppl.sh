#!/usr/bin/env bash
# FlexPosit MICRO 2026 AE — Step 02: Headline WikiText-2 PPL sweep
#
# For each model, builds:
#   - Posit(4,1) base checkpoint          (target 4.0, upgrade path source)
#   - Posit(5,1) base checkpoint          (target 5.0, downgrade path source)
#   - FlexPosit checkpoints at all requested bit targets
#     * target < 4.5:  upgrade   from Posit(4,1) — top-N most-sensitive → 5b
#     * target >= 4.5: downgrade from Posit(5,1) — bottom-M least-sensitive → 4b
#     Both directions produce equivalent PPL (validated). Downgrade path
#     touches (5-target)/1 fraction of windows — up to 9x faster at target 4.9.
#
# NOTE: Sensitivity profiling is NOT re-run. Pre-computed CSVs are shipped
# at sensitivity/. To re-profile from scratch, see reports/DOWNSTREAM_README.md.
#
# Override via env vars:
#   MODELS="phi-2 llama-2-7b"          # subset (default: all 9)
#   TARGETS="4.0 4.5 5.0"              # subset (default: 4.0 4.1 4.4 4.7 5.0)
#   WORK_DIR=/path/to/scratch          # checkpoint output root
#   SKIP_QWEN14B=1                     # drop 14B (for <40 GB VRAM cards)

set -euo pipefail

ENV_NAME="${ENV_NAME:-posit-ae}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/ae_work}"
RESULTS_DIR="$REPO_ROOT/results"
OUT_CSV="$RESULTS_DIR/table2_headline_ppl.csv"

DEFAULT_MODELS="gpt2-large gpt2-xl phi-2 opt-2.7b llama-2-7b mistral-7b deepseek-llm-7b qwen2.5-7b qwen2.5-14b"
DEFAULT_TARGETS="4.0 4.1 4.4 4.7 5.0"
MODELS="${MODELS:-$DEFAULT_MODELS}"
TARGETS="${TARGETS:-$DEFAULT_TARGETS}"

if [[ "${SKIP_QWEN14B:-0}" == "1" ]]; then
  MODELS="$(echo "$MODELS" | tr ' ' '\n' | grep -v '^qwen2.5-14b$' | tr '\n' ' ')"
  echo "[SKIP_QWEN14B=1] Removed qwen2.5-14b from sweep"
fi

if [[ "${SKIP_LLAMA2:-0}" == "1" ]]; then
  MODELS="$(echo "$MODELS" | tr ' ' '\n' | grep -v '^llama-2-7b$' | tr '\n' ' ')"
  echo "[SKIP_LLAMA2=1] Removed llama-2-7b from sweep (Meta approval not required)"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PYTHONUNBUFFERED=1

# --- Preflight: gated Llama-2 needs either HF_TOKEN (online), a warm offline cache, or SKIP_LLAMA2=1 ---
# Bail before starting the 6-13 h sweep rather than 30 s into it.
if [[ "${SKIP_LLAMA2:-0}" != "1" ]] && [[ "${HF_HUB_OFFLINE:-0}" != "1" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  cat >&2 <<EOF
================================================================================
[preflight] ERROR: HuggingFace model access is not configured.

  meta-llama/Llama-2-7b-hf is gated and cannot be downloaded without a token.
  Please set one of:

    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx    # online path
    export HF_HUB_OFFLINE=1                                  # use pre-warmed cache
    export SKIP_LLAMA2=1                                     # skip llama-2-7b entirely

  See the "HuggingFace model access" section in README.md.
  Aborting before the 6-13 h sweep.
================================================================================
EOF
  exit 2
fi

mkdir -p "$WORK_DIR" "$RESULTS_DIR"
cd "$REPO_ROOT"

if [[ ! -f "$OUT_CSV" ]]; then
  echo "model,target_avg_bits,achieved_avg_bits,wikitext2_ppl,direction,sensitivity_csv,checkpoint_dir,timestamp_utc" > "$OUT_CSV"
fi

declare -A HFID SEQLEN SENSCSV
HFID[gpt2-large]="gpt2-large";                            SEQLEN[gpt2-large]=1024;      SENSCSV[gpt2-large]="gpt2-large.csv"
HFID[gpt2-xl]="gpt2-xl";                                  SEQLEN[gpt2-xl]=1024;         SENSCSV[gpt2-xl]="gpt2-xl.csv"
HFID[phi-2]="microsoft/phi-2";                            SEQLEN[phi-2]=2048;           SENSCSV[phi-2]="phi-2.csv"
HFID[opt-2.7b]="facebook/opt-2.7b";                       SEQLEN[opt-2.7b]=2048;        SENSCSV[opt-2.7b]="opt-2.7b.csv"
HFID[llama-2-7b]="meta-llama/Llama-2-7b-hf";              SEQLEN[llama-2-7b]=2048;      SENSCSV[llama-2-7b]="llama-2-7b.csv"
HFID[mistral-7b]="mistralai/Mistral-7B-v0.1";             SEQLEN[mistral-7b]=2048;      SENSCSV[mistral-7b]="mistral-7b.csv"
HFID[deepseek-llm-7b]="deepseek-ai/deepseek-llm-7b-base"; SEQLEN[deepseek-llm-7b]=2048; SENSCSV[deepseek-llm-7b]="deepseek-llm-7b.csv"
HFID[qwen2.5-7b]="Qwen/Qwen2.5-7B";                       SEQLEN[qwen2.5-7b]=2048;      SENSCSV[qwen2.5-7b]="qwen2.5-7b.csv"
HFID[qwen2.5-14b]="Qwen/Qwen2.5-14B";                     SEQLEN[qwen2.5-14b]=2048;     SENSCSV[qwen2.5-14b]="qwen2.5-14b.csv"

# fp16 (=bf16 for llama-2-7b to avoid overflow in some layers)
declare -A DTYPE
for m in gpt2-large gpt2-xl phi-2 opt-2.7b mistral-7b deepseek-llm-7b qwen2.5-7b qwen2.5-14b; do DTYPE[$m]=fp16; done
DTYPE[llama-2-7b]=bf16

# Decide direction: >=4.5 -> downgrade (fewer iterations), else upgrade
needs_posit4_base=0
needs_posit5_base=0
for t in $TARGETS; do
  # awk handles float comparison portably
  if awk "BEGIN{exit !($t >= 4.5)}"; then
    needs_posit5_base=1
  else
    needs_posit4_base=1
  fi
done

echo "==========================================================="
echo "Headline PPL sweep (auto-selects upgrade/downgrade per target)"
echo "==========================================================="
echo "Models:   $MODELS"
echo "Targets:  $TARGETS"
echo "Bases:    posit4=$needs_posit4_base  posit5=$needs_posit5_base"
echo "Work dir: $WORK_DIR"
echo "Output:   $OUT_CSV"
echo

# --------------------------------------------------------------------
# Baselines: FP16, INT4 per-channel, MXFP8 per-group (g=32)
# --------------------------------------------------------------------
BASE_CSV="$RESULTS_DIR/table2_baselines.csv"
if [[ ! -f "$BASE_CSV" ]]; then
  echo "model,method,wikitext2_ppl" > "$BASE_CSV"
fi

extract_ppl () {
  # Extract [Result] wikitext2_ppl = X.XXXX from a log file
  grep -oE 'wikitext2_ppl = [0-9]+\.[0-9]+' "$1" | tail -1 | awk '{print $NF}'
}

echo "==========================================================="
echo "Baselines: FP16 + INT4 per-channel + MXFP8 PG"
echo "==========================================================="
for model in $MODELS; do
  hf_id="${HFID[$model]}"
  seqlen="${SEQLEN[$model]}"
  dtype="${DTYPE[$model]}"

  # FP16 (weights untouched, no act quant)
  if ! grep -q "^${model},fp16," "$BASE_CSV" 2>/dev/null; then
    echo "[baseline] $model  fp16"
    log="$WORK_DIR/baseline_fp16_${model}.log"
    python quantization/eval_ppl.py \
      --model "$hf_id" --act_quant none --dtype "$dtype" --seqlen "$seqlen" > "$log" 2>&1 || true
    ppl=$(extract_ppl "$log")
    if [[ -n "$ppl" ]]; then
      echo "${model},fp16,${ppl}" >> "$BASE_CSV"
      echo "  -> ppl=$ppl"
    else
      echo "[WARN] $model fp16: no PPL extracted, see $log"
    fi
  fi

  # INT4 per-channel range
  if ! grep -q "^${model},int4_per_channel," "$BASE_CSV" 2>/dev/null; then
    echo "[baseline] $model  int4_per_channel"
    int4_dir="$WORK_DIR/quant_out_${model}_int4_pc"
    if [[ ! -f "$int4_dir/metrics.json" ]]; then
      python quantization/quantize_int4.py \
        --model "$hf_id" --seqlen "$seqlen" --dtype "$dtype" \
        --save_dir "$int4_dir"
    fi
    ppl=$(python -c "import json; print(f\"{json.load(open('$int4_dir/metrics.json'))['wikitext2_ppl']:.4f}\")")
    echo "${model},int4_per_channel,${ppl}" >> "$BASE_CSV"
    echo "  -> ppl=$ppl"
  fi
done

# MXFP8 PG: runs all requested models in one call (has its own model map)
if [[ -n "$MODELS" ]]; then
  only_csv=$(echo "$MODELS" | tr ' ' ',')
  # Skip models already recorded
  need_mxfp8=$(echo "$MODELS" | tr ' ' '\n' | while read m; do
    grep -q "^${m},mxfp8_pg_g32," "$BASE_CSV" 2>/dev/null || echo "$m"
  done | tr '\n' ',' | sed 's/,$//')
  if [[ -n "$need_mxfp8" ]]; then
    echo "[baseline] MXFP8 PG (g=32): $need_mxfp8"
    python quantization/mxfp8_group_wise.py --only "$need_mxfp8" --csv_out "$BASE_CSV"
  fi
fi

echo
echo "[verify baselines]"
python "$REPO_ROOT/quantization/verify.py" \
  --produced "$BASE_CSV" \
  --expected "$REPO_ROOT/expected/table2_baselines.csv" \
  --key model,method \
  --produced-value wikitext2_ppl \
  --expected-value wikitext2_ppl \
  --tol 0.1 \
  --label "Table 2 baselines (FP16, INT4, MXFP8 PG)" || echo "[baselines] tolerance breaches noted (continuing to FlexPosit sweep)"
echo

for model in $MODELS; do
  hf_id="${HFID[$model]}"
  seqlen="${SEQLEN[$model]}"
  senscsv="$REPO_ROOT/sensitivity/${SENSCSV[$model]}"

  echo
  echo "############################################################"
  echo "# $model  (hf=$hf_id  seqlen=$seqlen)"
  echo "############################################################"

  base4_dir="$WORK_DIR/quant_out_${model}_posit4_es1"
  base5_dir="$WORK_DIR/quant_out_${model}_posit5_es1"

  # Build Posit(4,1) base if any target < 4.5 needs it
  if [[ "$needs_posit4_base" == "1" && ! -f "$base4_dir/quant_log.json" ]]; then
    echo "[build] $base4_dir  (posit4)"
    python quantization/posit_quantize.py \
      --model "$model" --dtype fp16 --device cuda \
      --weight_format posit4 --nsize 4 --es_candidates 1 \
      --log2_min -8 --log2_max 9 --ch_batch 64 \
      --ppl_seqlen "$seqlen" --save_dir "$base4_dir"
  fi

  # Build Posit(5,1) base if any target >= 4.5 needs it
  if [[ "$needs_posit5_base" == "1" && ! -f "$base5_dir/quant_log.json" ]]; then
    echo "[build] $base5_dir  (posit5)"
    python quantization/posit_quantize.py \
      --model "$model" --dtype fp16 --device cuda \
      --weight_format posit5 --nsize 5 --es_candidates 1 \
      --log2_min -8 --log2_max 9 --ch_batch 64 \
      --ppl_seqlen "$seqlen" --save_dir "$base5_dir"
  fi

  for t in $TARGETS; do
    out_dir="$WORK_DIR/quant_out_${model}_flexposit_${t}_es1"

    if grep -q "^${model},${t}," "$OUT_CSV" 2>/dev/null; then
      echo "[skip] $model @ target=$t already in $OUT_CSV"
      continue
    fi

    # Endpoints — target 4.0 = base posit4, target 5.0 = base posit5
    if awk "BEGIN{exit !($t <= 4.0)}"; then
      # Report base posit4 as target 4.0
      python - <<PY
import json, csv, os, datetime
p = os.path.join("$base4_dir", "metrics.json")
if not os.path.exists(p):
    raise SystemExit(f"missing {p}")
d = json.load(open(p))
ppl = d.get("wikitext2_ppl") or d.get("ppl")
with open("$OUT_CSV", "a", newline="") as f:
    csv.writer(f).writerow([
        "$model","$t","4.0000",f"{ppl:.4f}","base",
        os.path.basename("$senscsv"),"$base4_dir",
        datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z"])
print(f"  -> posit4 base ppl={ppl:.4f}")
PY
      continue
    fi
    if awk "BEGIN{exit !($t >= 5.0)}"; then
      # Report base posit5 as target 5.0
      python - <<PY
import json, csv, os, datetime
p = os.path.join("$base5_dir", "metrics.json")
if not os.path.exists(p):
    raise SystemExit(f"missing {p}")
d = json.load(open(p))
ppl = d.get("wikitext2_ppl") or d.get("ppl")
with open("$OUT_CSV", "a", newline="") as f:
    csv.writer(f).writerow([
        "$model","$t","5.0000",f"{ppl:.4f}","base",
        os.path.basename("$senscsv"),"$base5_dir",
        datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z"])
print(f"  -> posit5 base ppl={ppl:.4f}")
PY
      continue
    fi

    # Interior target: pick direction and base
    if awk "BEGIN{exit !($t >= 4.5)}"; then
      direction="downgrade"
      chosen_base="$base5_dir"
      extra_flag="--downgrade"
    else
      direction="upgrade"
      chosen_base="$base4_dir"
      extra_flag=""
    fi

    if [[ ! -f "$out_dir/mixposit_budget_log.json" ]]; then
      echo "[build] $model @ target=$t via $direction  (base=$(basename $chosen_base))"
      python quantization/flexposit_mpq.py \
        --base_dir "$chosen_base" \
        --fp32_reference_dir "$hf_id" \
        --sensitivity_csv "$senscsv" \
        --out_dir "$out_dir" \
        --target_avg_bits "$t" \
        $extra_flag \
        --override_nsize 5 --es_candidates 1 \
        --base_bits 4 --upgrade_bits 5 \
        --log2_min -8 --log2_max 9 \
        --seqlen "$seqlen"
    fi

    python - <<PY
import json, csv, os, datetime
p = "$out_dir/mixposit_budget_log.json"
d = json.load(open(p))
with open("$OUT_CSV", "a", newline="") as f:
    csv.writer(f).writerow([
        "$model", "$t",
        f"{d['achieved_avg_bits']:.4f}",
        f"{d['final_ppl']:.4f}",
        "$direction",
        os.path.basename("$senscsv"),
        "$out_dir",
        datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z",
    ])
print(f"  -> $direction achieved_bits={d['achieved_avg_bits']:.4f} ppl={d['final_ppl']:.4f}")
PY
  done
done

echo
echo "==========================================================="
echo "Done. Aggregate CSV: $OUT_CSV"
echo "Row count: $(($(wc -l < $OUT_CSV) - 1))"
echo "==========================================================="

python "$REPO_ROOT/quantization/verify.py" \
  --produced "$OUT_CSV" \
  --expected "$REPO_ROOT/expected/table2_headline_ppl.csv" \
  --key model,target_avg_bits \
  --produced-value wikitext2_ppl \
  --expected-value wikitext2_ppl \
  --tol 0.1 \
  --label "Table 2 (headline PPL)"
