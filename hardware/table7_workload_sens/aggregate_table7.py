#!/usr/bin/env python3
# aggregate_table7.py — reduce 20 test_*.log files into Table 7 (workload sensitivity).
# For each of the 4 (batch, context) workloads, computes FlexPosit's geomean speedup and
# energy reduction versus FP16 baseline, BitMoD, and OliVe. Writes table7_workload_sens.csv.

import math, os, re, csv, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

WORKLOADS = [("1", "256"), ("1", "8192"), ("4", "4096"), ("8", "8192")]
BASELINES = ["baseline", "bitmod", "olive"]        # column baselines
FLEX      = "flexposit"

CYCLE_RE  = re.compile(r"Total Cycle:\s+([\d,\.]+)")
ENERGY_RE = re.compile(r"Total Energy:\s+([\d\.]+)\s*uJ")


def parse_log(path):
    """Return [(cycles, energy_uJ)] per model in file order."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    cycles, energies = [], []
    with open(path) as f:
        for line in f:
            m = CYCLE_RE.search(line)
            if m:
                cycles.append(float(m.group(1).replace(",", "")))
            m = ENERGY_RE.search(line)
            if m:
                energies.append(float(m.group(1)))
    if len(cycles) != len(energies) or not cycles:
        raise ValueError(f"parse mismatch in {path}: cycles={len(cycles)} energies={len(energies)}")
    return list(zip(cycles, energies))


def geomean(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    rows = []
    for B, L in WORKLOADS:
        tag = f"B{B}_L{L}"
        flex = parse_log(os.path.join(DATA, f"{tag}_{FLEX}.log"))
        base = {b: parse_log(os.path.join(DATA, f"{tag}_{b}.log")) for b in BASELINES}

        speedup = {}
        energy_red = {}
        for b, pairs in base.items():
            # per-model speedup = baseline_cycles / flexposit_cycles
            per_model_speedup = [pb[0] / pf[0] for pf, pb in zip(flex, pairs) if pf[0] > 0 and pb[0] > 0]
            per_model_energy  = [pb[1] / pf[1] for pf, pb in zip(flex, pairs) if pf[1] > 0 and pb[1] > 0]
            speedup[b]    = geomean(per_model_speedup)
            energy_red[b] = geomean(per_model_energy)

        rows.append({
            "batch_size":                B,
            "context_length":            L,
            "speedup_vs_FP16":           round(speedup["baseline"], 2),
            "speedup_vs_BitMoD":         round(speedup["bitmod"],   2),
            "speedup_vs_OliVe":          round(speedup["olive"],    2),
            "energy_reduction_vs_FP16":  round(energy_red["baseline"], 2),
            "energy_reduction_vs_BitMoD":round(energy_red["bitmod"],   2),
            "energy_reduction_vs_OliVe": round(energy_red["olive"],    2),
        })

    out = os.path.join(HERE, "table7_workload_sens.csv")
    cols = ["batch_size", "context_length",
            "speedup_vs_FP16", "speedup_vs_BitMoD", "speedup_vs_OliVe",
            "energy_reduction_vs_FP16", "energy_reduction_vs_BitMoD", "energy_reduction_vs_OliVe"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)

    print(f"[written] {out}")
    print()
    print(f"{'(B,L)':<12} {'x FP16':>6} {'x BitMoD':>8} {'x OliVe':>7} | {'/ FP16':>6} {'/ BitMoD':>8} {'/ OliVe':>7}")
    for r in rows:
        print(f"({r['batch_size']},{r['context_length']:<4}) {r['speedup_vs_FP16']:>6.1f} {r['speedup_vs_BitMoD']:>8.1f} {r['speedup_vs_OliVe']:>7.1f} | "
              f"{r['energy_reduction_vs_FP16']:>6.1f} {r['energy_reduction_vs_BitMoD']:>8.1f} {r['energy_reduction_vs_OliVe']:>7.1f}")


if __name__ == "__main__":
    main()
