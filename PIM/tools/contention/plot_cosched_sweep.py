#!/usr/bin/env python3
"""Figure for the network co-scheduling ablation: co-scheduling gain (serial / co-scheduled) of the GPU baseline and of
PCG^2 against the network bandwidth, one panel per GPU, and the resulting speedup under the three accountings.
Reads the json written by cosched_sweep.py.      plot_cosched_sweep.py SWEEP_JSON OUT_PNG"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
S = json.load(open(sys.argv[1])); out = sys.argv[2]
COL = {"L40S": "#C8322B", "B200": "#C9950F"}
fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.0), sharex=True, gridspec_kw=dict(hspace=0.12, wspace=0.08, left=0.075, right=0.99, top=0.9, bottom=0.11))
for j, mach in enumerate(("L40S", "B200")):
    g = [r for r in S["grid"] if r["mach"] == mach]; b = np.array([r["beta"] for r in g])
    tab = {r["beta"]: r for r in S["table"] if r["mach"] == mach}
    ax = axes[0][j]
    ax.plot(b, [r["gain_pcg"] for r in g], "-", color=COL[mach], lw=1.8, label="PCG$^2$")
    ax.plot(b, [r["gain_gpu"] for r in g], "--", color="0.35", lw=1.5, label="GPU baseline")
    # peaks: T_c = T_n of the geomean cell, from the table rows (tn scales 1/beta)
    r1 = tab[1e9]
    for tc, lab, col, dy in ((r1["tp"], "PCG$^2$ peak", COL[mach], 0.07), (r1["tg"], "GPU peak", "0.35", -0.1)):
        bp = 1e9 * r1["tn"] / tc
        ax.plot([bp], [2.0 if False else max(x for x in [np.interp(np.log10(bp), np.log10(b), [r["gain_pcg" if col == COL[mach] else "gain_gpu"] for r in g])])], "o", ms=4, color=col, mec="white", mew=0.8, zorder=5)
        ax.annotate(f"{lab}\n{bp/1e6:.0f} Mbps", (bp, np.interp(np.log10(bp), np.log10(b), [r["gain_pcg" if col == COL[mach] else "gain_gpu"] for r in g])), fontsize=6, color=col, ha="center", xytext=(0, 9 if dy > 0 else -16), textcoords="offset points")
    ax.set_ylim(0.98, 1.5); ax.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ax.set_title(mach, fontsize=8, pad=3)
    ax2 = axes[1][j]
    ax2.plot(b, [r["sp_both"] for r in g], "-", color=COL[mach], lw=1.8, label="both co-scheduled (this paper)")
    ax2.plot(b, [r["sp_pcg_only"] for r in g], ":", color=COL[mach], lw=1.5, label="PCG$^2$ only (submission)")
    ax2.plot(b, [r["sp_none"] for r in g], "--", color="0.35", lw=1.5, label="neither")
    ax2.set_ylim(0.9, 8.5 if mach == "B200" else 3.0)
    for ax_ in (ax, ax2):
        ax_.set_xscale("log"); ax_.set_xlim(1e7, 1e11)
        for x, lab in ((40e9, "40 Gbps"), (400e6, "400 Mbps")):
            ax_.axvline(x, color="0.75", lw=0.7, ls="-", zorder=0)
        ax_.grid(axis="y", lw=0.4, color="0.9", zorder=0); ax_.set_axisbelow(True); ax_.tick_params(labelsize=6.5, pad=1.5)
        for s in ("top", "right"): ax_.spines[s].set_visible(False)
    ylo = ax2.get_ylim()[0]
    ax2.text(40e9, ylo + 0.03 * (ax2.get_ylim()[1] - ylo), "40 Gbps", fontsize=6, color="0.4", ha="center"); ax2.text(400e6, ylo + 0.03 * (ax2.get_ylim()[1] - ylo), "400 Mbps", fontsize=6, color="0.4", ha="center")
    ax2.set_xlabel("network bandwidth (bit/s, log)", fontsize=7.5, labelpad=1)
    ax2.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11]); ax2.set_xticklabels(["10M", "100M", "1G", "10G", "100G"])
    if j == 0:
        ax.set_ylabel("co-scheduling gain\n(serial / co-scheduled)", fontsize=7.5, labelpad=2)
        ax2.set_ylabel("speedup over GPU", fontsize=7.5, labelpad=2)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=1.6)
        ax2.legend(fontsize=6.2, frameon=False, loc="upper left", handlelength=1.6)
    else:
        ax.set_yticklabels([]); 
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote", out)
