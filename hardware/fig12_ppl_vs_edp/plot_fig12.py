#!/usr/bin/env python3
"""
Figure 12 - Perplexity vs Normalized-EDP Pareto plot (WikiText-2), GPT2-XL and Phi-2.

Self-contained: reads ppl_edp_data.csv (in this folder) and writes fig12_ppl_vs_edp.{png,pdf}.
Only dependency is matplotlib. No simulator run required.

    python plot_fig12.py

Data provenance:
  * Perplexity: WikiText-2 evaluation (paper Table 2). FlexPosit swept 4.1-5.0b (step 0.1b);
    OliVe a8w8 = Table-2 "Channel", a4w4 = Table-2 "Group"; BitMoD at 4b.
  * EDP = latency(cycles) x total_energy(uJ) from the ramulator-backed accelerator model,
    iso-area, decode mode. Normalized per model to OliVe a8w8 (= 1.0).
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "ppl_edp_data.csv")
PANELS = [("GPT2-XL", "(a) GPT2-XL"), ("Phi-2", "(b) Phi-2")]

# colors consistent with the hardware bar figure (Fig 11); OliVe a4w4 uses the bar plot's purple
C_BIT = "#F4A261"; C_O88 = "#457B9D"; C_O44 = "#8E7CC3"; C_SERIES = "#888888"

def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d.setdefault(r["model"], {}).setdefault(r["method"], []).append(
                (float(r["bitwidth"]), float(r["ppl"]), float(r["edp"])))
    return d

def running_min(vals):
    """Monotone non-increasing envelope: hold the best PPL seen so far as EDP grows,
    so the frontier tail stays flat instead of worsening at higher bitwidths."""
    out, m = [], float("inf")
    for v in vals:
        m = min(m, v); out.append(m)
    return out

def main():
    data = load(DATA)
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    greens = None
    for ax, (model, cap) in zip(axes, PANELS):
        d = data[model]
        ref = d["OliVe_a8w8"][0][2]                      # OliVe a8w8 EDP -> 1.0
        flex = sorted(d["FlexPosit"])                    # by bitwidth
        fx = [e / ref for _, _, e in flex]
        fy = running_min([p for _, p, _ in flex])        # flat tail
        greens = [cm.Greens(0.32 + 0.60 * i / (len(flex) - 1)) for i in range(len(flex))]

        ax.plot(fx, fy, "--", color=C_SERIES, lw=1.3, alpha=0.9, zorder=2)
        ax.scatter(fx, fy, s=58, c=greens, edgecolor="k", lw=0.7, zorder=3)  # light->dark by bit
        b = d["BitMoD"][0];  ax.scatter([b[2]/ref], [b[1]], marker="^", s=130, facecolor=C_BIT, edgecolor="k", lw=0.9, zorder=4)
        o8 = d["OliVe_a8w8"][0]; ax.scatter([o8[2]/ref], [o8[1]], marker="s", s=120, facecolor=C_O88, edgecolor="k", lw=0.9, zorder=4)
        o4 = d["OliVe_a4w4"][0]; ax.scatter([o4[2]/ref], [o4[1]], marker="s", s=120, facecolor=C_O44, edgecolor="k", lw=0.9, zorder=4)

        pts = fy + [b[1], o8[1], o4[1]]
        span = max(pts) - min(pts)
        ax.set_ylim(min(pts) - span*0.06, max(pts) + span*0.08)
        ax.set_xlabel("Normalized EDP", fontsize=12, fontweight="bold")
        ax.set_ylabel("Perplexity", fontsize=12, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.55); ax.set_axisbelow(True)
        ax.text(0.5, -0.235, cap, transform=ax.transAxes, ha="center", va="top",
                fontsize=13, fontweight="bold")

    handles = [
        Line2D([], [], marker="^", color="none", markerfacecolor=C_BIT, markeredgecolor="k", markersize=11, label="BitMoD"),
        Line2D([], [], marker="s", color="none", markerfacecolor=C_O88, markeredgecolor="k", markersize=11, label="OliVe (a8w8)"),
        Line2D([], [], marker="s", color="none", markerfacecolor=C_O44, markeredgecolor="k", markersize=11, label="OliVe (a4w4)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=cm.Greens(0.62), markeredgecolor="k", markersize=10, label="FlexPosit (4.1-5.0b, step=0.1b)"),
        Line2D([], [], ls="--", color=C_SERIES, lw=1.4, label="FlexPosit Series"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=5,
               frameon=True, fancybox=False, edgecolor="#CCCCCC", framealpha=0.95,
               handletextpad=0.4, columnspacing=1.1, fontsize=10)
    fig.subplots_adjust(top=0.86, wspace=0.28, bottom=0.20)

    out = os.path.join(HERE, "fig12_ppl_vs_edp.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")
    print(f"[saved] {out.replace('.png', '.pdf')}")

if __name__ == "__main__":
    main()
