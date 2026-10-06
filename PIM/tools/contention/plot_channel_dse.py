#!/usr/bin/env python3
"""Channel-count DSE figure (Reviewer B): B200 variants with 48..512 channels (HBM3e timing, per-channel bandwidth,
GPU kernels fixed; 8 SPUs + 1 DRU per channel). (a) PCG^2 end-to-end runtime (geomean over the suite, 40 Gbps) with the
NTT on the SMs (four-step, transposes on the SMs), with the DRU (four-step + DRU, the headline design), with the
conservative DRU on the submission's four-step, and with the merge NTT (what the runtime falls back to).
(b) The DRU's gain against the channel count and the two knees: Eq. dru_enable (the DRU's stream fits beside the GPU's)
and the point where the DRU design overtakes the merge NTT (the runtime enables the DRU).
Usage: plot_channel_dse.py FINAL_WINDOW_DIR OUT_PNG"""
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
SRC = {d: load(f"{W}/channel_sweep/e2e.json") + load(f"{W}/win22/e2e_nom.json") for d in ("merge", "f4dru")}
SRC["f4sm"] = load(f"{W}/channel_sweep/e2e_f4sm.json") + load(f"{W}/win22/e2e_f4sm.json")
for d in ("sq", "sqdruh"): SRC[d] = load(f"{W}/fourstep_dru/e2e_chsweep_conservative_dru.json") + load(f"{W}/fourstep_dru/e2e_nom_conservative_dru.json")
CH = [48, 96, 128, 192, 256, 512]; BW = [8.2 * c / 256 for c in CH]
T, G = {d: [] for d in SRC}, {"cons": [], "indes": [], "indes_lane": [], "merge": []}
for ch in CH:
    org = "b200" if ch == 256 else f"b200_ch{ch}"; C = {d: cells(SRC[d], org, d) for d in SRC}; ks = sorted(C["f4dru"])
    for d in SRC: T[d].append(gm(C[d][k]["pcg_ms"] for k in ks))
    G["cons"].append(gm(C["sq"][k]["pcg_ms"] / C["sqdruh"][k]["pcg_ms"] for k in ks)); G["indes"].append(gm(C["f4sm"][k]["pcg_ms"] / C["f4dru"][k]["pcg_ms"] for k in ks))
    G["indes_lane"].append(gm(lane(C["f4sm"][k]) / lane(C["f4dru"][k]) for k in ks)); G["merge"].append(gm(lane(C["merge"][k]) / lane(C["f4dru"][k]) for k in ks))
U_SPU, NEED = 0.21, 240 * (1 << 24) / 2.2766e-3; CSTAR = NEED / (8.2e12 / 256 * (1 - U_SPU))
# merge crossover: interpolate the lane ratio = 1 on a log axis
xo = None
for i in range(len(CH) - 1):
    a, b = G["merge"][i], G["merge"][i + 1]
    if a < 1 <= b: xo = exp(log(CH[i]) + (1 - a) / (b - a) * (log(CH[i + 1]) - log(CH[i]))); break
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw=dict(wspace=0.28, left=0.08, right=0.99, top=0.82, bottom=0.2))
S = [("f4sm", "four-step, transposes on the SMs (no DRU)", "0.35", "--", "s"), ("f4dru", "four-step + DRU (PCG$^2$)", "#C9950F", "-", "s"),
     ("sqdruh", "submission four-step + conservative DRU", "#C9950F", ":", "^"), ("merge", "merge NTT (runtime fallback)", "#C8322B", "-", "o")]
for d, lab, col, ls, mk in S:
    ax.plot(CH, T[d], ls, marker=mk, ms=3.8, lw=1.4, color=col, mec="black", mew=0.3, label=lab, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(CH); ax.set_xticklabels([str(c) for c in CH], fontsize=6.5); ax.minorticks_off()
ax.set_yticks([20, 50, 100, 200, 500]); ax.set_yticklabels(["20", "50", "100", "200", "500"], fontsize=6.5)
ax.set_xlabel("channels (8 SPUs + 1 DRU each)", fontsize=7.5, labelpad=1); ax.set_ylabel("PCG$^2$ runtime (ms, geomean)", fontsize=7.5, labelpad=2)
ax.set_title("(a) end-to-end runtime, 40 Gbps", fontsize=7.5, pad=3); ax.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=2.0)
for a_ in (ax, bx):
    top = a_.secondary_xaxis("top"); top.set_xscale("log")
    top.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(CH)); top.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{b:.1f}" for b in BW]))
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); top.tick_params(labelsize=6); top.set_xlabel("aggregate bandwidth (TB/s)", fontsize=7, labelpad=2)
    a_.axvline(256, color="0.75", lw=0.8, zorder=0); a_.axvline(CSTAR, color="#C8322B", lw=0.8, ls="--", zorder=0)
    a_.grid(axis="y", lw=0.4, color="0.9", zorder=0); a_.set_axisbelow(True); a_.tick_params(pad=1.5)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
ax.text(256 * 1.03, 22, "B200", fontsize=6, color="0.4", rotation=90, va="bottom"); ax.text(CSTAR * 0.96, 22, "Eq. (dru_enable)\nknee 70 ch", fontsize=5.5, color="#C8322B", rotation=90, va="bottom", ha="right")
bx.plot(CH, G["cons"], ":", marker="^", ms=4, lw=1.4, color="#C9950F", mec="black", mew=0.3, label="conservative DRU, end-to-end", zorder=3)
bx.plot(CH, G["indes"], "-", marker="s", ms=3.8, lw=1.4, color="#C9950F", mec="black", mew=0.3, label="DRU in the four-step, end-to-end", zorder=3)
bx.plot(CH, G["indes_lane"], "-", marker="s", ms=3.8, lw=1.0, color="#E6C77A", mec="black", mew=0.3, label="DRU in the four-step, NTT lane", zorder=3)
bx.plot(CH, G["merge"], "-", marker="o", ms=3.8, lw=1.4, color="#C8322B", mec="black", mew=0.3, label="DRU design vs merge NTT, NTT lane", zorder=3)
bx.axhline(1.0, color="0.5", lw=0.7, zorder=1)
if xo: bx.plot([xo], [1.0], "*", ms=9, color="#C8322B", mec="black", mew=0.5, zorder=6); bx.annotate(f"runtime enables\nthe DRU: {xo:.0f} ch", (xo, 1.0), fontsize=5.5, color="#C8322B", xytext=(-4, 8), textcoords="offset points", ha="right")
bx.set_xscale("log"); bx.set_xticks(CH); bx.set_xticklabels([str(c) for c in CH], fontsize=6.5); bx.minorticks_off(); bx.set_ylim(0.84, 1.36); bx.tick_params(axis="y", labelsize=6.5)
bx.set_xlabel("channels (8 SPUs + 1 DRU each)", fontsize=7.5, labelpad=1); bx.set_ylabel("DRU gain (x)", fontsize=7.5, labelpad=2)
bx.set_title("(b) DRU gain vs channel count", fontsize=7.5, pad=3); bx.legend(fontsize=5.6, frameon=False, loc="lower right", handlelength=2.0)
for i, c in enumerate(CH): bx.annotate(f"{G['cons'][i]:.2f}", (c, G["cons"][i]), fontsize=5.5, color="#8a6508", xytext=(0, 5), textcoords="offset points", ha="center")
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT, "| merge crossover", xo, "| C*", CSTAR)
for d in T: print(d, " ".join(f"{x:.1f}" for x in T[d]))
