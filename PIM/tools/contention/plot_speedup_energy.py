#!/usr/bin/env python3
"""Speedup and energy-efficiency improvement of PCG^2 over the GPU baseline, per (c,t) configuration, both as lines:
solid = speedup (Fig. 8), dashed = energy efficiency (Table IV accounting), one panel per network tier."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
CFG = ["8,4", "4,16", "8,8", "2,64", "2,128"]
SP = {"40 Gbps": {"L40S": [3.03, 2.33, 2.63, 2.21, 2.13], "B200": [5.20, 9.71, 7.33, 8.34, 8.16]},
      "400 Mbps": {"L40S": [3.03, 2.33, 2.63, 1.57, 1.06], "B200": [5.20, 3.60, 5.28, 1.39, 1.10]}}
EE = {"40 Gbps": {"L40S": [2.76, 4.89, 3.85, 7.33, 7.17], "B200": [4.20, 10.67, 6.44, 15.18, 14.93]},
      "400 Mbps": {"L40S": [2.76, 4.89, 3.85, 5.09, 2.83], "B200": [4.20, 5.77, 5.44, 2.93, 1.92]}}
COL = {"L40S": "#C8322B", "B200": "#C9950F"}; MK = {"L40S": "o", "B200": "s"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True, gridspec_kw=dict(left=0.07, right=0.99, top=0.80, bottom=0.2, wspace=0.08))
x = range(len(CFG))
for ax, tier in zip(axes, ("40 Gbps", "400 Mbps")):
    for mach in ("L40S", "B200"):
        ax.plot(x, SP[tier][mach], "-", marker=MK[mach], ms=4, lw=1.4, color=COL[mach], mec="black", mew=0.4)
        ax.plot(x, EE[tier][mach], "--", marker=MK[mach], ms=4, lw=1.2, color=COL[mach], mfc="white", mec=COL[mach], mew=1.1)
    ax.set_title(f"Comm. BW {tier}", fontsize=8, pad=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(CFG, fontsize=7); ax.set_xlabel("(c, t)", fontsize=7.5, labelpad=1)
    ax.set_yscale("log"); ax.set_ylim(0.8, 22); ax.set_yticks([1, 2, 4, 8, 16]); ax.set_yticklabels(["1", "2", "4", "8", "16"], fontsize=7)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.grid(axis="y", lw=0.4, color="0.9"); ax.set_axisbelow(True)
    ax.axhline(1, color="0.4", lw=0.7, ls=":")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("improvement over\nGPU baseline (x)", fontsize=7.5)
H = [Line2D([], [], color=COL["L40S"], marker="o", ms=4, lw=1.4, mec="black", mew=0.4), Line2D([], [], color=COL["L40S"], marker="o", ms=4, lw=1.2, ls="--", mfc="white", mec=COL["L40S"], mew=1.1),
     Line2D([], [], color=COL["B200"], marker="s", ms=4, lw=1.4, mec="black", mew=0.4), Line2D([], [], color=COL["B200"], marker="s", ms=4, lw=1.2, ls="--", mfc="white", mec=COL["B200"], mew=1.1)]
fig.legend(H, ["L40S speedup", "L40S energy efficiency", "B200 speedup", "B200 energy efficiency"], fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=4, handlelength=2.4, columnspacing=1.6)
fig.savefig("fig_speedup_energy.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig("fig_speedup_energy.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
