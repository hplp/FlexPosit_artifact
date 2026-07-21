"""Diff a produced CSV against a shipped expected CSV cell-by-cell.

Used at the tail of 03/04/06 to give reviewers a pass/fail summary
instead of two CSVs to eyeball.
"""
import argparse
import csv
import sys


def _load(path, key_cols, value_col):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if any(k not in r for k in key_cols):
            sys.exit(f"[verify] {path}: missing key column(s) {key_cols}; have {list(r.keys())}")
        if value_col not in r:
            sys.exit(f"[verify] {path}: missing value column '{value_col}'; have {list(r.keys())}")
    return {tuple(r[k] for k in key_cols): r[value_col] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--produced", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--key", required=True, help="Comma-separated key columns (same names in both files)")
    ap.add_argument("--produced-value", required=True)
    ap.add_argument("--expected-value", required=True)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--label", default="table")
    args = ap.parse_args()

    key_cols = [c.strip() for c in args.key.split(",")]
    prod = _load(args.produced, key_cols, args.produced_value)
    exp = _load(args.expected, key_cols, args.expected_value)

    shared = sorted(set(prod).intersection(exp))
    only_prod = sorted(set(prod) - set(exp))
    only_exp = sorted(set(exp) - set(prod))

    fails = []
    rows = []
    for k in shared:
        try:
            p, e = float(prod[k]), float(exp[k])
        except ValueError:
            fails.append((k, prod[k], exp[k], "non-numeric"))
            rows.append((k, prod[k], exp[k], None))
            continue
        d = p - e
        rows.append((k, p, e, d))
        if abs(d) > args.tol:
            fails.append((k, p, e, d))

    n_ok = len(shared) - len(fails)
    print()
    print("=" * 79)
    print(f"[verify] {args.label}")
    print("=" * 79)

    # Full side-by-side table
    key_widths = [max(len(c), max((len(str(k[i])) for k, _, _, _ in rows), default=1)) for i, c in enumerate(key_cols)]
    header = "  ".join(c.ljust(w) for c, w in zip(key_cols, key_widths))
    print(f"  {header}    produced    expected    delta   status")
    print("  " + "-" * (sum(key_widths) + 2 * (len(key_widths) - 1) + 40))
    for k, p, e, d in rows:
        key_str = "  ".join(str(v).ljust(w) for v, w in zip(k, key_widths))
        if d is None:
            print(f"  {key_str}  {str(p):>10}  {str(e):>10}  {'':>7}   NUM?")
        else:
            status = "OK" if abs(d) <= args.tol else "FAIL"
            print(f"  {key_str}  {p:>10.4f}  {e:>10.4f}  {d:>+7.4f}   {status}")

    print("  " + "-" * (sum(key_widths) + 2 * (len(key_widths) - 1) + 40))
    max_abs_d = max((abs(d) for _, _, _, d in rows if d is not None), default=0.0)
    if not fails:
        print(f"  Result: {n_ok}/{len(shared)} cells within +/-{args.tol}  (max |Δ| = {max_abs_d:.4f})    PASS")
    else:
        print(f"  Result: {n_ok}/{len(shared)} pass, {len(fails)} diffs > {args.tol}  (max |Δ| = {max_abs_d:.4f})    FAIL")

    if only_exp:
        print()
        print(f"  Note: {len(only_exp)} expected row(s) not produced "
              f"(denser reference grid or skipped by MODELS/SKIP_QWEN14B):")
        for k in only_exp[:5]:
            print(f"    {', '.join(f'{c}={v}' for c, v in zip(key_cols, k))}")
        if len(only_exp) > 5:
            print(f"    ... and {len(only_exp) - 5} more")
    if only_prod:
        print(f"  Note: {len(only_prod)} produced row(s) have no expected match (ignored)")
    print("=" * 79)

    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
