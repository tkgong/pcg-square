#!/usr/bin/env python3
"""Fig. 8 with the energy-efficiency improvement added as a second set of lines on the right axis:
solid lines = speedup (as before), dashed lines with hollow markers = energy efficiency (Table IV accounting,
per-(c,t) geomean from energy_l40s.txt / energy_b200.txt).    plot_fig8_energy.py E2E_JSON OUT"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_fig8_paper_style import data, CFG, ALPHA, TIER, COL, LINE, SERIES
EE = {"fast": {"L40S": [2.76, 4.89, 3.85, 7.33, 7.17], "B200": [4.20, 10.67, 6.44, 15.18, 14.93]},
      "slow": {"L40S": [2.76, 4.89, 3.85, 5.09, 2.83], "B200": [4.20, 5.77, 5.44, 2.93, 1.92]}}

def main(D, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), sharey=True,
                             gridspec_kw=dict(wspace=0.06, left=0.075, right=0.925, top=0.80, bottom=0.24))
    n = len(CFG); span, nser = 0.86, 4; w = span / nser * 0.92
    for ax, (tier, title) in zip(axes, TIER):
        rax = ax.twinx()
        for si, (mach, kind, lab) in enumerate(SERIES):
            x = np.arange(n) + (si - (nser - 1) / 2) * (span / nser)
            ax.bar(x, [D[tier][mach][i][kind] for i in range(n)], w, color=COL[(mach, kind)], edgecolor="black", linewidth=0.35, zorder=3, label=lab if ax is axes[0] else None)
        for mach in ("L40S", "B200"):
            s = [D[tier][mach][i][2] for i in range(n)]
            rax.plot(np.arange(n), s, "-o", color=LINE[mach], ms=4.2, lw=1.6, mec="white", mew=0.9, zorder=7)
            rax.plot(np.arange(n), EE[tier][mach], "--", marker="o", color=LINE[mach], ms=4.2, lw=1.3, mfc="white", mec=LINE[mach], mew=1.2, zorder=6)
        rax.set_ylim(0, 16.8); rax.set_yticks([0, 4, 8, 12, 16]); rax.tick_params(labelsize=6.4, pad=1.5)
        if ax is axes[1]: rax.set_ylabel("speedup / energy efficiency", fontsize=7.6, labelpad=2)
        else: rax.set_yticklabels([])
        ax.set_yscale("log"); ax.set_ylim(15, 25000)
        ax.set_yticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{c},{t}\n" + (f"{a:.0f}µs" if a < 1000 else f"{a/1000:.1f}ms") for (c, t), a in zip(CFG, ALPHA[tier])], fontsize=6.4)
        ax.set_xlabel(r"$(c,t)$  /  resulting $\alpha$", fontsize=7.6, labelpad=1)
        ax.set_title(f"Comm. BW {title}", fontsize=7.6, pad=2.5)
        ax.grid(axis="y", lw=0.4, color="0.9", zorder=0); ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.4, pad=1.5); ax.set_xlim(-0.58, n - 0.42)
    axes[0].set_ylabel("runtime (ms, log)", fontsize=7.6, labelpad=2)
    axes[0].text(0.0, 1.015, r"$\times10^3$", transform=axes[0].transAxes, fontsize=6.4, ha="left", va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], color="0.25", marker="o", ms=4.2, lw=1.6, mec="white", mew=0.9), Line2D([], [], color="0.25", marker="o", ms=4.2, lw=1.3, ls="--", mfc="white", mec="0.25", mew=1.2)]
    l += ["speedup", "energy efficiency"]
    fig.legend(h, l, fontsize=6.4, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False, columnspacing=1.2, handlelength=1.4, handletextpad=0.5)
    fig.savefig(outpath, bbox_inches="tight", dpi=300, pad_inches=0.02); fig.savefig(outpath.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote", outpath)

if __name__ == "__main__":
    main(data(sys.argv[1]), sys.argv[2])
