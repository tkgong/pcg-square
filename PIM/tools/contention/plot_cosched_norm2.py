#!/usr/bin/env python3
# Two flat panels (L40S top, B200 bottom), same total height as the single-panel figure: GPU co-scheduled and
# PCG^2 co-scheduled normalized to the serial GPU baseline (= 1) vs network bandwidth, 40 Mbps .. 40 Gbps.
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
OUT = sys.argv[1] if len(sys.argv) > 1 else "fig_cosched_norm2.png"
BW = [40e9, 20e9, 10e9, 4e9, 2e9, 1e9, 400e6, 200e6, 100e6, 40e6]
D = {
"L40S": {"gpu_cos": [1.004, 1.009, 1.018, 1.043, 1.082, 1.142, 1.211, 1.245, 1.286, 1.316],
         "pcg_cos": [2.50, 2.51, 2.53, 2.59, 2.66, 2.67, 2.58, 2.46, 2.27, 1.95], "ylim": (0.8, 3.0), "yt": [1, 2, 3]},
"B200": {"gpu_cos": [1.010, 1.020, 1.039, 1.092, 1.155, 1.214, 1.283, 1.318, 1.321, 1.300],
         "pcg_cos": [7.67, 7.68, 7.44, 6.71, 5.83, 4.78, 3.51, 2.75, 2.17, 1.64], "ylim": (0.8, 8.2), "yt": [1, 2, 4, 6, 8]},
}
COL = {"L40S": "#C8322B", "B200": "#C9950F"}; MK = {"L40S": "o", "B200": "s"}
fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.7), sharex=True, gridspec_kw=dict(left=0.13, right=0.98, top=0.86, bottom=0.14, hspace=0.12))
for ax, mach in zip(axes, ("L40S", "B200")):
    d = D[mach]
    ax.axhline(1.0, color="0.35", lw=0.9, ls=":", label="GPU serial (reference)" if mach == "L40S" else None)
    ax.plot(BW, d["gpu_cos"], "--", marker=MK[mach], ms=3.2, lw=1.1, color=COL[mach], mfc="white", mec=COL[mach], mew=1.0, label=f"{mach} GPU co-scheduled")
    ax.plot(BW, d["pcg_cos"], "-", marker=MK[mach], ms=3.2, lw=1.4, color=COL[mach], mec="black", mew=0.3, label=f"{mach} PCG$^2$ co-scheduled")
    ax.set_xscale("log"); ax.set_xlim(3e7, 5e10); ax.set_ylim(*d["ylim"]); ax.set_yticks(d["yt"]); ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", lw=0.4, color="0.9"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[1].set_xticks(BW); axes[1].set_xticklabels(["40G", "20G", "10G", "4G", "2G", "1G", "400M", "200M", "100M", "40M"], fontsize=6); axes[1].xaxis.set_minor_locator(mt.NullLocator())
axes[1].set_xlabel("network bandwidth (bit/s)", fontsize=7.5, labelpad=1)
fig.text(0.02, 0.5, "speedup over serial GPU (geomean)", rotation=90, va="center", fontsize=7.5)
h = axes[0].get_legend_handles_labels()[0] + axes[1].get_legend_handles_labels()[0]; l = axes[0].get_legend_handles_labels()[1] + axes[1].get_legend_handles_labels()[1]
fig.legend(h, l, fontsize=5.4, frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, handlelength=2.0, columnspacing=1.0, labelspacing=0.3)
fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
