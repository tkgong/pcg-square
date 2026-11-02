#!/usr/bin/env python3
# Speedup over the co-scheduled GPU baseline vs network bandwidth (40 Mbps .. 40 Gbps), PCG^2 serial and co-scheduled,
# both cards. The gap between the two curves of a card is PCG^2's own co-scheduling gain. Geomean over the suite.
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
OUT = sys.argv[1] if len(sys.argv) > 1 else "fig_cosched_speedup.png"
BW = [40e9, 20e9, 10e9, 4e9, 2e9, 1e9, 400e6, 200e6, 100e6, 40e6]
D = {
"L40S": {"serial": [2.46, 2.44, 2.39, 2.28, 2.13, 1.93, 1.67, 1.49, 1.32, 1.13], "cosched": [2.48, 2.48, 2.48, 2.48, 2.46, 2.34, 2.13, 1.98, 1.77, 1.48]},
"B200": {"serial": [7.03, 6.6, 5.97, 4.81, 3.85, 3.01, 2.15, 1.69, 1.38, 1.12], "cosched": [7.59, 7.53, 7.16, 6.14, 5.04, 3.93, 2.73, 2.09, 1.64, 1.26]},
}
COL = {"L40S": "#C8322B", "B200": "#C9950F"}; MK = {"L40S": "o", "B200": "s"}
fig, ax = plt.subplots(figsize=(3.5, 2.7), gridspec_kw=dict(left=0.13, right=0.98, top=0.84, bottom=0.16))
for mach in ("L40S", "B200"):
    ax.plot(BW, D[mach]["serial"], "--", marker=MK[mach], ms=3.5, lw=1.2, color=COL[mach], mfc="white", mec=COL[mach], mew=1.0, label=f"{mach} PCG$^2$ serial")
    ax.plot(BW, D[mach]["cosched"], "-", marker=MK[mach], ms=3.5, lw=1.5, color=COL[mach], mec="black", mew=0.3, label=f"{mach} PCG$^2$ co-scheduled")
ax.set_xscale("log"); ax.set_xlim(3e7, 5e10); ax.set_ylim(1, 8)
ax.set_xticks(BW); ax.set_xticklabels(["40G", "20G", "10G", "4G", "2G", "1G", "400M", "200M", "100M", "40M"], fontsize=6)
ax.xaxis.set_minor_locator(mt.NullLocator()); ax.tick_params(axis="y", labelsize=6.5)
ax.set_xlabel("network bandwidth (bit/s)", fontsize=7.5, labelpad=1); ax.set_ylabel("speedup over co-scheduled GPU (geomean)", fontsize=7.5)
ax.grid(axis="y", lw=0.4, color="0.9"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=5.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.2, columnspacing=1.2, labelspacing=0.3)
fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
