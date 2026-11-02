#!/usr/bin/env python3
"""Reviewer B: the DRU's gain as a function of the channel count, one curve, one knee. B200 variants with 48..512
channels (HBM3e timing, per-channel bandwidth, GPU kernels fixed; 8 SPUs + 1 DRU per channel; device bytes spread over
the channels). Gain = Table IV's DRU increment: PCG^2 with the submission's four-step on the SMs -> PCG^2 with the
four-step + DRU, end-to-end, co-scheduled, geomean over the suite at 40 Gbps. The runtime enables the DRU only where
the DRU design beats the merge NTT; below that channel count the curve is dashed (design gain, not realized) and the
realized gain is 1 (merge kept, as on L40S).    Usage: plot_channel_dse.py FINAL_WINDOW_DIR OUT_PNG"""
import json, os, sys
from math import exp, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
W, OUT = sys.argv[1], sys.argv[2]
def gm(v): v = list(v); return exp(sum(map(log, v)) / len(v))
def load(p): return json.load(open(p)) if os.path.exists(p) else []
def cells(rows, org, des): return {(r["c"], r["t"], r["logN"]): r for r in rows if r["org"] == org and r["design"] == des and r["tier"] == "fast"}
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]
MF = load(f"{W}/channel_sweep/e2e.json") + load(f"{W}/win22/e2e_nom.json")
SQ = load(f"{W}/fourstep_dru/e2e_chsweep_conservative_dru.json") + load(f"{W}/fourstep_dru/e2e_nom_conservative_dru.json")
CH = [48, 96, 128, 192, 256, 512]; BW = [8.2 * c / 256 for c in CH]
gain, ratio = [], []
for ch in CH:
    org = "b200" if ch == 256 else f"b200_ch{ch}"
    m, d, s = cells(MF, org, "merge"), cells(MF, org, "f4dru"), cells(SQ, org, "sq"); ks = sorted(d)
    gain.append(gm(s[k]["pcg_ms"] / d[k]["pcg_ms"] for k in ks)); ratio.append(gm(lane(m[k]) / lane(d[k]) for k in ks))
knee = None
for i in range(len(CH) - 1):
    if ratio[i] < 1 <= ratio[i + 1]: knee = exp(log(CH[i]) + (1 - ratio[i]) / (ratio[i + 1] - ratio[i]) * (log(CH[i + 1]) - log(CH[i]))); break
CSTAR = 240 * (1 << 24) / 2.2766e-3 / (8.2e12 / 256 * (1 - 0.21))
print("channels", CH); print("gain sq->f4dru", [round(g, 3) for g in gain]); print("merge/f4dru NTT lane", [round(r, 3) for r in ratio]); print("knee", knee, "C*", CSTAR)
fig, ax = plt.subplots(figsize=(3.5, 2.5), gridspec_kw=dict(left=0.16, right=0.98, top=0.8, bottom=0.2))
GOLD = "#C9950F"
off = [i for i, c in enumerate(CH) if c < knee]; on = [i for i, c in enumerate(CH) if c >= knee]
i0 = off[-1]; gk = gain[i0] + (gain[i0 + 1] - gain[i0]) * (log(knee) - log(CH[i0])) / (log(CH[i0 + 1]) - log(CH[i0]))   # gain at the knee
ax.plot([CH[i] for i in off] + [knee], [gain[i] for i in off] + [gk], "--", color=GOLD, lw=1.3, zorder=3)
ax.plot([CH[i] for i in off], [gain[i] for i in off], "s", ms=4.5, mfc="white", mec=GOLD, mew=1.2, zorder=4, label="DRU design gain, not enabled (merge NTT faster)")
ax.plot([knee] + [CH[i] for i in on], [gk] + [gain[i] for i in on], "-", color=GOLD, lw=1.6, zorder=3)
ax.plot([CH[i] for i in on], [gain[i] for i in on], "s", ms=4.5, color=GOLD, mec="black", mew=0.4, zorder=4, label="DRU gain, enabled by the runtime")
ax.plot([knee], [gk], "*", ms=11, color="#C8322B", mec="black", mew=0.5, zorder=6)
ax.annotate(f"knee: {knee:.0f} channels ({8.2*knee/256:.1f} TB/s):\nthe DRU design overtakes\nthe merge NTT, runtime enables it", (knee, gk), fontsize=5.6, color="#C8322B", xytext=(250, 1.1), textcoords="data", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#C8322B", lw=0.7, shrinkB=4))
for c, g in zip(CH, gain): ax.annotate(f"{g:.2f}", (c, g), fontsize=5.6, color="#7a5a08", xytext=(0, 6), textcoords="offset points", ha="center")
ax.axvline(CSTAR, color="0.6", lw=0.8, ls=":", zorder=1); ax.text(CSTAR * 1.04, 1.015, f"Eq. (dru_enable):\nDRU stream fits, {CSTAR:.0f} ch", fontsize=5.2, color="0.4", va="bottom")
ax.axvline(256, color="0.85", lw=0.8, zorder=0); ax.text(256 * 1.03, 1.015, "B200", fontsize=5.6, color="0.5", va="bottom")
ax.set_xscale("log"); ax.set_xticks(CH); ax.set_xticklabels([str(c) for c in CH], fontsize=6.5); ax.minorticks_off(); ax.set_xlim(40, 620)
ax.set_ylim(1.0, 1.42); ax.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4]); ax.tick_params(axis="y", labelsize=6.5)
ax.set_xlabel("channels (8 SPUs + 1 DRU each)", fontsize=7.5, labelpad=1); ax.set_ylabel("DRU gain (end-to-end, geomean)", fontsize=7.5, labelpad=2)
top = ax.secondary_xaxis("top"); top.set_xscale("log"); top.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(CH)); top.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{b:.1f}" for b in BW]))
top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); top.tick_params(labelsize=6); top.set_xlabel("aggregate bandwidth (TB/s)", fontsize=7, labelpad=2)
ax.grid(axis="y", lw=0.4, color="0.9", zorder=0); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(fontsize=5.4, frameon=False, loc="upper left", handlelength=1.6)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
