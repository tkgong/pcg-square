#!/usr/bin/env python3
"""Reviewer B: the butterfly DRU's gain as a function of the channel count, one curve per card, one knee each. Card variants
<org>_ch<C> (the card's DRAM timing, per-channel bandwidth, GPU kernels fixed; 8 SPUs + 1 DRU per channel; device bytes
spread over the channels). Gain = Table IV's DRU increment: PCG^2 with Cheddar's NTT on the SMs -> PCG^2 with the butterfly
DRU (f4b), end-to-end, co-scheduled, geomean over the suite at 40 Gbps. The runtime enables the DRU only where its NTT lane
beats Cheddar's; below that channel count the curve is dashed (design gain, not realized) and the realized gain is 1.
Inputs: RUN_DIR/e2e_nom.json + RUN_DIR/channel_sweep.json.    Usage: plot_channel_dse.py RUN_DIR OUT_PNG"""
import json, os, sys
from math import exp, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import final_design as FD
W, OUT = sys.argv[1], sys.argv[2]
def load(p): return json.load(open(p)) if os.path.exists(p) else []
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]
ALL = load(f"{W}/e2e_nom.json") + load(f"{W}/channel_sweep.json")
CARD = {"B200": dict(org="b200", native=256, chs=[8, 16, 32, 48, 96, 128, 192, 256, 512], bw=8.2 / 256, col="#C9950F", mk="s"),
        "L40S": dict(org="l40s", native=24, chs=[8, 16, 24, 48, 96, 128, 192, 256, 512], bw=0.864 / 24, col="#C8322B", mk="o")}
fig, ax = plt.subplots(figsize=(3.5, 2.5), gridspec_kw=dict(left=0.16, right=0.98, top=0.86, bottom=0.2))
for mach, c in CARD.items():
    CH, gain, ratio = [], [], []
    for ch in c["chs"]:
        org = c["org"] if ch == c["native"] else f"{c['org']}_ch{ch}"
        s, d = FD.cells(ALL, org, FD.SM_NTT), FD.cells(ALL, org, "f4b"); ks = sorted(k for k in d if k in s)
        if not ks: continue
        CH.append(ch); gain.append(FD.gm(s[k]["pcg_ms"] / d[k]["pcg_ms"] for k in ks)); ratio.append(FD.gm(lane(s[k]) / lane(d[k]) for k in ks))
    if not CH: continue
    knee = None
    for i in range(len(CH) - 1):
        if ratio[i] < 1 <= ratio[i + 1]: knee = exp(log(CH[i]) + (1 - ratio[i]) / (ratio[i + 1] - ratio[i]) * (log(CH[i + 1]) - log(CH[i]))); break
    print(mach, "channels", CH, "gain cheddar->f4b", [round(g, 3) for g in gain], "Cheddar/f4b NTT lane", [round(r, 3) for r in ratio], "knee", knee)
    off = [i for i, r in enumerate(ratio) if r < 1]; on = [i for i, r in enumerate(ratio) if r >= 1]
    if off: ax.plot([CH[i] for i in off], [gain[i] for i in off], "--", marker=c["mk"], ms=4.5, mfc="white", mec=c["col"], mew=1.2, color=c["col"], lw=1.3, zorder=3, label=f"{mach}: design gain, DRU not enabled")
    if on: ax.plot([CH[i] for i in on], [gain[i] for i in on], "-", marker=c["mk"], ms=4.5, color=c["col"], mec="black", mew=0.4, lw=1.6, zorder=4, label=f"{mach}: DRU enabled by the runtime")
    if knee:
        i0 = off[-1]; gk = gain[i0] + (gain[i0 + 1] - gain[i0]) * (log(knee) - log(CH[i0])) / (log(CH[i0 + 1]) - log(CH[i0]))
        ax.plot([knee], [gk], "*", ms=11, color=c["col"], mec="black", mew=0.5, zorder=6); ax.annotate(f"{knee:.0f} ch", (knee, gk), fontsize=5.6, color=c["col"], xytext=(4, -9), textcoords="offset points")
    for ch, g in zip(CH, gain): ax.annotate(f"{g:.2f}", (ch, g), fontsize=5.2, color=c["col"], xytext=(0, 6), textcoords="offset points", ha="center")
    ax.axvline(c["native"], color="0.85", lw=0.8, zorder=0); ax.text(c["native"] * 1.03, ax.get_ylim()[0] + 0.005, mach, fontsize=5.6, color="0.5", va="bottom")
CHS = sorted({ch for c in CARD.values() for ch in c["chs"]})
ax.set_xscale("log"); ax.set_xticks(CHS); ax.set_xticklabels([str(c) for c in CHS], fontsize=6); ax.minorticks_off(); ax.set_xlim(6.5, 620)
ax.tick_params(axis="y", labelsize=6.5); ax.axhline(1.0, color="0.4", lw=0.7, ls=":")
ax.set_xlabel("channels (8 SPUs + 1 DRU each)", fontsize=7.5, labelpad=1); ax.set_ylabel("butterfly-DRU gain (end-to-end, geomean)", fontsize=7.5, labelpad=2)
ax.grid(axis="y", lw=0.4, color="0.9", zorder=0); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(fontsize=5.0, frameon=False, loc="upper left", handlelength=1.6)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
