#!/usr/bin/env python3
"""Runtime and speedup at fixed bandwidth, grouped by cryptographic parameters.

Both tiers fix the link and let the message size set the round trip. A level's
ganged correction word is c^2 * 16t^2 bytes in each direction, so

    alpha = 2 * c^2 * 16 t^2 / beta

with beta the effective link bandwidth: 40 Gbps and 400 Mbps, a 100x drop. No
latency floor is added, so alpha is pure transfer time and both settings span
the same 64x range across the suite -- whatever changes between them is
attributable to bandwidth alone. Configurations are ordered by message size, so
alpha rises left to right.

Each bar is the geometric mean over N = 2^20 .. 2^24. Cells averaged over
fewer than five N are hatched (L40S is missing (2,64)@2^24 and (2,128)@2^23,2^24).

Sized for a two-column paper: 7.16 x 2.35 in, drop in at \\textwidth.

    python3 plot_bw_bars.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (c, t) ordered by ganged message size
CFG = [(8, 4), (4, 16), (8, 8), (2, 64), (2, 128)]
LOGN = [20, 21, 22, 23, 24]
NCELL = {"L40S": [5, 5, 5, 4, 3], "B200": [5, 5, 5, 5, 5]}

# tier -> alpha per configuration (us)
ALPHA = {
    "fast": [6.6, 26.2, 26.2, 104.9, 419.4],
    "slow": [655.4, 2621.4, 2621.4, 10485.8, 41943.0],
}
TIER = [("fast", "40 Gbps"), ("slow", "400 Mbps")]

#  tier -> machine -> (GPU ms, PIM ms, speedup) per configuration
D = {
    "fast": {
        "L40S": [(998.6, 410.3, 2.43), (572.4, 218.4, 2.62), (1415.0, 443.8, 3.19),
                 (341.0, 154.4, 2.21), (458.8, 218.3, 2.10)],
        "B200": [(410.5, 81.9, 5.01), (224.9, 23.7, 9.50), (562.2, 81.9, 6.86),
                 (187.6, 23.7, 7.92), (368.5, 47.4, 7.78)],
    },
    "slow": {
        "L40S": [(1020.8, 410.3, 2.49), (650.0, 218.4, 2.98), (1498.9, 443.8, 3.38),
                 (566.9, 238.0, 2.38), (1202.8, 712.2, 1.69)],
        "B200": [(430.5, 81.9, 5.26), (294.7, 60.1, 4.90), (637.6, 107.5, 5.93),
                 (418.7, 198.7, 2.11), (1203.4, 752.6, 1.60)],
    },
}
GEO = {"fast": {"L40S": 2.53, "B200": 7.26}, "slow": {"L40S": 2.61, "B200": 3.49}}

COL = {"L40S": "#2E6F9E", "B200": "#C1666B"}
SERIES = [("L40S", 0, "L40S GPU baseline"), ("L40S", 1, "L40S PCG$^2$"),
          ("B200", 0, "B200 GPU baseline"), ("B200", 1, "B200 PCG$^2$")]


def main(outpath="bw_bars.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), sharey=True,
                             gridspec_kw=dict(wspace=0.05, left=0.075, right=0.925,
                                              top=0.82, bottom=0.24))
    n = len(CFG)
    span, nser = 0.86, 4
    w = span / nser * 0.90

    for ax, (tier, title) in zip(axes, TIER):
        rax = ax.twinx()
        for si, (mach, kind, lab) in enumerate(SERIES):
            x = np.arange(n) + (si - (nser - 1) / 2) * (span / nser)
            for i, xi in enumerate(x):   # per-cell, so the hatch marks the right bars
                thin = kind and NCELL[mach][i] < len(LOGN)
                ax.bar(xi, D[tier][mach][i][kind], w, color=COL[mach],
                       alpha=1.0 if kind else 0.34,
                       edgecolor="black" if kind else COL[mach],
                       linewidth=0.35 if kind else 0.7, zorder=3,
                       hatch="///" if thin else None,
                       label=lab if ax is axes[0] and i == 0 else None)

        for mach in ("L40S", "B200"):
            s = [D[tier][mach][i][2] for i in range(n)]
            rax.plot(np.arange(n), s, "-o", color=COL[mach], ms=4.2, lw=1.6,
                     mec="white", mew=0.9, zorder=7)
            # L40S labels below its markers, B200 above, so they never collide;
            # a B200 peak near the top of the axis flips below to stay inside
            for i, v in enumerate(s):
                dy = -10 if mach == "L40S" or v > 9.0 else 6
                rax.annotate(f"{v:.1f}", (i, v), fontsize=6.0, color=COL[mach],
                             xytext=(0, dy), textcoords="offset points",
                             ha="center", zorder=8)

        # float the speedup markers above the bars: the axis extends below zero,
        # but only the meaningful ticks are drawn
        rax.set_ylim(-12.0, 10.6)
        rax.set_yticks([2, 4, 6, 8, 10])
        rax.tick_params(labelsize=6.4, pad=1.5)
        if ax is axes[1]:
            rax.set_ylabel("speedup", fontsize=7.6, labelpad=2)
        else:
            rax.set_yticklabels([])

        ax.set_yscale("log")
        ax.set_ylim(14, 22000)
        ax.set_xticks(range(n))
        ax.set_xticklabels(
            [f"{c},{t}\n" + (f"{a:.0f}µs" if a < 1000 else f"{a/1000:.1f}ms")
             for (c, t), a in zip(CFG, ALPHA[tier])], fontsize=6.4)
        ax.set_xlabel(r"$(c,t)$  /  resulting $\alpha$", fontsize=7.6, labelpad=1)
        ax.set_title(f"{title}      geomean "
                     f"{GEO[tier]['L40S']:.2f}× / {GEO[tier]['B200']:.2f}×",
                     fontsize=7.6, pad=2.5)
        ax.grid(axis="y", lw=0.4, color="0.9", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.4, pad=1.5)
        ax.set_xlim(-0.58, n - 0.42)

    axes[0].set_ylabel("runtime (ms, log)", fontsize=7.6, labelpad=2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.4, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), frameon=False,
               columnspacing=1.4, handlelength=1.2, handletextpad=0.5)

    fig.savefig(outpath, bbox_inches="tight", dpi=300, pad_inches=0.02)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bw_bars.pdf")
