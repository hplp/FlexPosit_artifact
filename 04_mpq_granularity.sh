#!/usr/bin/env bash
# FlexPosit MICRO 2026 AE — Step 04: granularity MPQ
#
# Reproduces the 27-row table comparing FlexPosit's channel-window MPQ against
# two layer-wise MPQ baselines (PPL-profiled and Fisher-scored) across three
# models (opt-2.7b, phi-2, llama-2-7b) at three bit targets (4.0, 4.4, 4.7).
#
# Two reproduction levels:
#   DEFAULT ("fast aggregate"): reads shipped raw sweep CSVs from
#                               data/table4_raw_sweeps/ AND the FlexPosit rows
#                               from results/table2_headline_ppl.csv
#                               (produced by 02_headline_ppl.sh). Runs in seconds.
#   FULL=1 mode: re-runs quantization/mpq_software_baseline.py for each
#                (model, sensitivity) pair from scratch, then aggregates the
#                freshly-produced sweep CSVs. Regeneration takes ~6-10 h on 1x A40.
#
# Reference: expected/table4_granularity_mpq.csv
# Prerequisite: 02_headline_ppl.sh must have run first (need its Table 2 CSV
#               for the FlexPosit channel-window MPQ rows).
#
# Env flags:
#   FULL=1     Regenerate the 6 baseline sweep CSVs from scratch (~6-10 h on 1x A40)
#   WORK_DIR   Scratch dir for FULL=1 output (default: ./ae_work)

set -euo pipefail

ENV_NAME="${ENV_NAME:-posit-ae}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/ae_work}"
RESULTS_DIR="$REPO_ROOT/results"
OUT_CSV="$RESULTS_DIR/table4_granularity_mpq.csv"

# Table 2 CSV — 03 writes both filenames depending on version; accept either.
TABLE2_CSV="$RESULTS_DIR/table2_headline_ppl.csv"
if [[ ! -f "$TABLE2_CSV" && -f "$RESULTS_DIR/headline_ppl.csv" ]]; then
  TABLE2_CSV="$RESULTS_DIR/headline_ppl.csv"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONUNBUFFERED=1

mkdir -p "$RESULTS_DIR"
cd "$REPO_ROOT"

# --- Preflight: HF token needed only in FULL=1 (baselines pull FP reference) ---
if [[ "${FULL:-0}" == "1" ]]; then
  if [[ "${SKIP_LLAMA2:-0}" != "1" ]] && [[ "${HF_HUB_OFFLINE:-0}" != "1" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[preflight] ERROR: HuggingFace model access not configured for FULL=1." >&2
    echo "            Set HF_TOKEN, HF_HUB_OFFLINE=1, or SKIP_LLAMA2=1.  See README's 'HuggingFace model access' section." >&2
    exit 2
  fi
fi

# --- Fast-mode requires Table 2 CSV (for FlexPosit rows) ---
if [[ ! -f "$TABLE2_CSV" ]]; then
  cat >&2 <<EOF
================================================================================
[04] ERROR: Table 2 CSV not found.

  Looked for:
    $RESULTS_DIR/table2_headline_ppl.csv
    $RESULTS_DIR/headline_ppl.csv

  Table 4 aggregates the FlexPosit channel-window MPQ rows from Table 2. Please
  run 02_headline_ppl.sh first (or produce table2_headline_ppl.csv manually)
  before running this script.
================================================================================
EOF
  exit 2
fi

MODELS_T4="opt-2.7b phi-2 llama-2-7b"
TARGETS_T4="4.0 4.4 4.7"

if [[ "${SKIP_LLAMA2:-0}" == "1" ]]; then
  MODELS_T4="$(echo "$MODELS_T4" | tr ' ' '\n' | grep -v '^llama-2-7b$' | tr '\n' ' ')"
  echo "[SKIP_LLAMA2=1] Removed llama-2-7b from Table 4 (leaves opt-2.7b, phi-2)"
fi

# Path where sweep CSVs will be read from. Fast mode = shipped; FULL = fresh outputs.
if [[ "${FULL:-0}" == "1" ]]; then
  SWEEP_ROOT="$WORK_DIR/mpq_layerwise"
else
  SWEEP_ROOT="$REPO_ROOT/data/table4_raw_sweeps"
fi

echo "==========================================================="
echo "Table IV — granularity MPQ (3 models x 3 methods x 3 targets)"
echo "==========================================================="
echo "Models:    $MODELS_T4"
echo "Targets:   $TARGETS_T4"
echo "Sweep root: $SWEEP_ROOT"
echo "Table 2 CSV: $TABLE2_CSV"
echo "Output:    $OUT_CSV"
echo

# --- FULL=1: regenerate the 6 baseline sweep CSVs ---
if [[ "${FULL:-0}" == "1" ]]; then
  echo "[FULL=1] Regenerating 6 baseline sweep CSVs (~6-10 h on 1x A40)..."

  declare -A HFID SEQLEN DTYPE
  HFID[opt-2.7b]="facebook/opt-2.7b";           SEQLEN[opt-2.7b]=2048;   DTYPE[opt-2.7b]=fp16
  HFID[phi-2]="microsoft/phi-2";                SEQLEN[phi-2]=2048;      DTYPE[phi-2]=fp16
  HFID[llama-2-7b]="meta-llama/Llama-2-7b-hf";  SEQLEN[llama-2-7b]=2048; DTYPE[llama-2-7b]=bf16

  for m in $MODELS_T4; do
    hf="${HFID[$m]}"
    sl="${SEQLEN[$m]}"
    dt="${DTYPE[$m]}"
    base_dir="$WORK_DIR/quant_out_${m}_posit4_es1"

    if [[ ! -d "$base_dir" ]]; then
      echo "[04 FULL] ERROR: base posit4 checkpoint missing at $base_dir" >&2
      echo "           Run 02_headline_ppl.sh first (with $m in MODELS)." >&2
      exit 3
    fi

    for method_pair in "ppl_probe:ppl_probe_layer_4to5" "fisher:fisher_layer_4to5"; do
      sens="${method_pair%%:*}"
      subdir="${method_pair##*:}"
      out_dir="$SWEEP_ROOT/$m/$subdir"

      if [[ -f "$out_dir/ppl_vs_avg_bits.csv" ]]; then
        echo "[skip] $m/$subdir already has ppl_vs_avg_bits.csv"
        continue
      fi
      echo "[build] $m  sensitivity=$sens  granularity=layer  4b->5b"
      python quantization/mpq_software_baseline.py \
        --base_dir "$base_dir" \
        --ref_id "$hf" \
        --out_dir "$out_dir" \
        --granularity layer \
        --sensitivity "$sens" \
        --b_low 4 --b_high 5 \
        --es 1 \
        --seqlen "$sl" \
        --dtype "$dt"
    done
  done
fi

# --- Sanity: expected sweep CSVs must exist before aggregation ---
missing=""
for m in $MODELS_T4; do
  for sub in ppl_probe_layer_4to5 fisher_layer_4to5; do
    f="$SWEEP_ROOT/$m/$sub/ppl_vs_avg_bits.csv"
    if [[ ! -f "$f" ]]; then
      missing="$missing $f"
    fi
  done
done
if [[ -n "$missing" ]]; then
  echo "[04] ERROR: missing sweep CSV(s):" >&2
  for f in $missing; do echo "        $f" >&2; done
  exit 4
fi

# --- Aggregate 27 rows: 18 baseline + 9 FlexPosit ---
python - <<PY
import csv, os, sys

REPO_ROOT   = "$REPO_ROOT"
SWEEP_ROOT  = "$SWEEP_ROOT"
TABLE2_CSV  = "$TABLE2_CSV"
OUT_CSV     = "$OUT_CSV"
FULL        = "${FULL:-0}" == "1"

models  = "$MODELS_T4".split()
targets = ["4.0", "4.4", "4.7"]

# Method map: pretty name -> (subdir under SWEEP_ROOT/<model>/)
baseline_methods = [
    ("ppl_profiled_layer", "ppl_probe_layer_4to5"),
    ("fisher_layer",       "fisher_layer_4to5"),
]

def rel(p):
    """Relative to repo root, POSIX-style, for CSV cosmetic consistency."""
    try:
        return os.path.relpath(p, REPO_ROOT).replace(os.sep, "/")
    except ValueError:
        return p

def pick_row(rows, target):
    """Sweep CSVs have rows every 0.1 bit; pick exact target if present, else
    the row with the closest achieved_avg_bits."""
    # Skip step-0 duplicate (both step 0 and step 1 land at 4.0)
    cand = [r for r in rows if r.get("target_avg_bits", "") not in ("",)]
    # Prefer exact target match (last occurrence — the non-step-0 one)
    exact = [r for r in cand if float(r["target_avg_bits"]) == float(target)]
    if exact:
        return exact[-1]
    # Fall back: closest achieved
    return min(cand, key=lambda r: abs(float(r["achieved_avg_bits"]) - float(target)))

def fmt_bits(x):
    """Format bit-average matching the expected CSV (e.g. 4.0, 4.4062, 4.7188)."""
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    # Preserve trailing .0 for exact integers (e.g. "4" -> "4.0")
    return s if "." in s else s + ".0"

rows_out = []

# --- Baseline rows (18) ---
for m in models:
    for pretty, sub in baseline_methods:
        src = os.path.join(SWEEP_ROOT, m, sub, "ppl_vs_avg_bits.csv")
        with open(src, newline="") as f:
            sweep = list(csv.DictReader(f))
        src_path = rel(src)
        for t in targets:
            r = pick_row(sweep, t)
            achieved = float(r["achieved_avg_bits"])
            ppl      = float(r["ppl"])
            rows_out.append([
                m, pretty, "layer", t,
                fmt_bits(achieved),
                f"{ppl:.4f}",
                src_path,
            ])

# --- FlexPosit rows (9) from Table 2 CSV ---
with open(TABLE2_CSV, newline="") as f:
    t2 = list(csv.DictReader(f))

table2_rel = rel(TABLE2_CSV)
targ_set = set(targets)
for m in models:
    hits = [r for r in t2 if r["model"] == m and r["target_avg_bits"] in targ_set]
    for t in targets:
        matches = [r for r in hits if r["target_avg_bits"] == t]
        if not matches:
            print(f"[warn] Table 2 has no row for model={m} target={t}; "
                  f"flexposit_cw_mpq row will be MISSING.", file=sys.stderr)
            continue
        r = matches[-1]
        achieved = float(r["achieved_avg_bits"])
        ppl      = float(r["wikitext2_ppl"])
        rows_out.append([
            m, "flexposit_cw_mpq", "channel_window", t,
            f"{achieved:g}",
            f"{ppl:.4f}",
            table2_rel,
        ])

# --- Write output CSV ---
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "model", "method", "granularity", "target_avg_bits",
        "achieved_avg_bits", "wikitext2_ppl", "source_csv",
    ])
    w.writerows(rows_out)

print(f"[05 aggregate] Wrote {len(rows_out)}/27 rows to {OUT_CSV}")
if len(rows_out) < 27:
    print(f"[05 aggregate] WARNING: only {len(rows_out)} rows produced "
          f"(expected 27). Verify step will flag missing keys.")
PY

echo
echo "==========================================================="
echo "Done. Aggregate CSV: $OUT_CSV"
echo "==========================================================="

python "$REPO_ROOT/quantization/verify.py" \
  --produced "$OUT_CSV" \
  --expected "$REPO_ROOT/expected/table4_granularity_mpq.csv" \
  --key model,method,target_avg_bits \
  --produced-value wikitext2_ppl \
  --expected-value wikitext2_ppl \
  --tol 0.1 \
  --label "Table 4 (granularity MPQ)"
