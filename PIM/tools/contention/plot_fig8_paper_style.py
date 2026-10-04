#!/usr/bin/env python3
"""Fig. 8 in the submitted figure's style (red/pink L40S, gold/yellow B200, runtime axis in x10^3 ms,
"Comm. BW" titles, speedup labels on the lines), from the final-accounting rows (e2e_window.py json):
GPU baseline = max(GPU DPF + merge NTT, network); PCG^2 = pcg_ms. B200 with the DRU, L40S with merge.

    plot_fig8_paper_style.py E2E_JSON OUT
"""
import json, os, sys
from math import exp, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from plot_bw_bars import CFG, LOGN, ALPHA, TIER


def gm(v):
    v = list(v); return exp(sum(map(log, v)) / len(v))


def data(path):
    R = json.load(open(path)); D = {}
    for tier in ("fast", "slow"):
        D[tier] = {}
        for M, org, des in (("L40S", "l40s", "merge"), ("B200", "b200", "f4dru")):
            cells = []
            for (c, t) in CFG:
                rr = [r for r in R if r["org"] == org and r["design"] == des and r["tier"] == tier and r["c"] == c and r["t"] == t]
                base = [max(r["gpu_ms"], r["nic_ms"]) for r in rr]; pcg = [r["pcg_ms"] for r in rr]
                cells.append((gm(base), gm(pcg), gm(b / p for b, p in zip(base, pcg))))
            D[tier][M] = cells
    return D


COL = {("L40S", 0): "#C8322B", ("L40S", 1): "#F2A5A0", ("B200", 0): "#D9A21B", ("B200", 1): "#F6E9AB"}
LINE = {"L40S": "#C8322B", "B200": "#C9950F"}
SERIES = [("L40S", 0, "L40S GPU baseline"), ("L40S", 1, "L40S PCG$^2$"), ("B200", 0, "B200 GPU baseline"), ("B200", 1, "B200 PCG$^2$")]


def main(D, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), sharey=True,
                             gridspec_kw=dict(wspace=0.06, left=0.075, right=0.925, top=0.80, bottom=0.24))
    n = len(CFG); span, nser = 0.86, 4; w = span / nser * 0.92
    for ax, (tier, title) in zip(axes, TIER):
        rax = ax.twinx()
        for si, (mach, kind, lab) in enumerate(SERIES):
            x = np.arange(n) + (si - (nser - 1) / 2) * (span / nser)
            ax.bar(x, [D[tier][mach][i][kind] for i in range(n)], w, color=COL[(mach, kind)], edgecolor="black", linewidth=0.35,
                   zorder=3, label=lab if ax is axes[0] else None)
        for mach in ("L40S", "B200"):
            s = [D[tier][mach][i][2] for i in range(n)]
            rax.plot(np.arange(n), s, "-o", color=LINE[mach], ms=4.2, lw=1.6, mec="white", mew=0.9, zorder=7)
            for i, v in enumerate(s):
                dy = -10 if mach == "L40S" or v > 9.2 else 6
                rax.annotate(f"{v:.1f}", (i, v), fontsize=6.0, color=LINE[mach], xytext=(0, dy), textcoords="offset points", ha="center", zorder=8,
                             bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
        rax.set_ylim(0, 10.6); rax.set_yticks([0, 2, 4, 6, 8, 10]); rax.tick_params(labelsize=6.4, pad=1.5)
        if ax is axes[1]: rax.set_ylabel("speedup", fontsize=7.6, labelpad=2)
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
    fig.legend(h, l, fontsize=6.4, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False, columnspacing=1.4, handlelength=1.2, handletextpad=0.5)
    fig.savefig(outpath, bbox_inches="tight", dpi=300, pad_inches=0.02)
    print("wrote", outpath)


if __name__ == "__main__":
    main(data(sys.argv[1]), sys.argv[2])
