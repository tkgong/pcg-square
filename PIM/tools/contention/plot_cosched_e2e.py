#!/usr/bin/env python3
# Usage: plot_cosched_e2e.py COSCHED_SWEEP_JSON OUT_PNG
# End-to-end runtime vs network bandwidth, 10 points from 40 Gbps to 40 Mbps, both co-scheduled:
# GPU baseline (DPF + Cheddar NTT, network overlapped) and PCG^2. Geomean over the suite, ms.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import json, sys
OUT = sys.argv[2] if len(sys.argv) > 2 else "fig_cosched_e2e.png"
BW = [40e9, 20e9, 10e9, 4e9, 2e9, 1e9, 400e6, 200e6, 100e6, 40e6]
T = {(r["mach"], r["beta"]): r for r in json.load(open(sys.argv[1]))["table"]}   # cosched_sweep.py JSON (geomean ms per bandwidth)
row = lambda m, b: T[(m, b)]
D = {m: {"gpu": [row(m, b)["t_gpu_cos"] for b in BW], "pcg": [row(m, b)["t_pcg_cos"] for b in BW]} for m in ("L40S", "B200")}
COL = {"L40S": "#C8322B", "B200": "#C9950F"}; MK = {"L40S": "o", "B200": "s"}
fig, ax = plt.subplots(figsize=(3.5, 2.7), gridspec_kw=dict(left=0.14, right=0.98, top=0.84, bottom=0.16))
for mach in ("L40S", "B200"):
    ax.plot(BW, D[mach]["gpu"], "--", marker=MK[mach], ms=3.5, lw=1.2, color=COL[mach], mfc="white", mec=COL[mach], mew=1.0, label=f"{mach} GPU baseline (co-scheduled)")
    ax.plot(BW, D[mach]["pcg"], "-", marker=MK[mach], ms=3.5, lw=1.5, color=COL[mach], mec="black", mew=0.3, label=f"{mach} PCG$^2$ (co-scheduled)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3e7, 5e10); ax.set_ylim(30, 2000)
ax.set_xticks(BW); ax.set_xticklabels(["40G", "20G", "10G", "4G", "2G", "1G", "400M", "200M", "100M", "40M"], fontsize=6)
ax.xaxis.set_minor_locator(mt.NullLocator())
ax.set_yticks([30, 100, 300, 1000]); ax.set_yticklabels(["30", "100", "300", "1000"], fontsize=6.5); ax.yaxis.set_minor_locator(mt.NullLocator())
ax.set_xlabel("network bandwidth (bit/s)", fontsize=7.5, labelpad=1); ax.set_ylabel("end-to-end runtime (ms, geomean)", fontsize=7.5)
ax.grid(axis="y", lw=0.4, color="0.9"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=5.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.2, columnspacing=1.2, labelspacing=0.3)
fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
