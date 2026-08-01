#!/usr/bin/env python3
"""Lane utilisation with and without the network-SPU co-schedule.

Utilisation = lane busy time / wall clock.

    co-schedule OFF :  wall = SPU + SM + NIC        (three lanes serialised)
    co-schedule ON  :  wall = max(SPU, SM, NIC)     (three lanes overlapped)

Both configurations gang their communication, so the NIC carries the same n+2
messages and the same bytes either way; the only difference is whether the
network wait can be filled with another batch's in-bank expansion.

Numbers are the mean over the measured cells (L40S 15, B200 18; logN 22-24 x the
six BCG+20 security rows). Every input is measured: GPU phases on silicon, PIM
lanes from 52 ramulator2 points.

    python3 plot_utilisation.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- data ------------------------------------------------------------------
ALPHA = [100, 500, 10000, 20000, 30000]          # microseconds
XLAB = ["100 µs", "500 µs", "10 ms", "20 ms", "30 ms"]

#                     co-schedule OFF            co-schedule ON
#                  SPU     SM     NIC          SPU     SM     NIC
UTIL = {
    "L40S": {
        100:   ((60.5, 39.2,  0.3), (89.2, 63.1,   0.5)),
        500:   ((59.7, 38.7,  1.6), (89.2, 63.1,   2.3)),
        10000: ((45.5, 31.5, 22.9), (89.2, 63.1,  46.7)),
        20000: ((37.2, 26.8, 36.0), (79.7, 59.0,  75.8)),
        30000: ((31.7, 23.5, 44.9), (65.4, 50.7,  86.3)),
    },
    "B200": {
        100:   ((56.2, 41.6,  2.2), (85.2, 69.8,   3.5)),
        500:   ((51.5, 38.8,  9.7), (85.2, 69.8,  17.4)),
        10000: ((20.2, 17.6, 62.1), (35.0, 33.1,  96.6)),
        20000: ((12.9, 11.8, 75.4), (18.8, 19.0, 100.0)),
        30000: (( 9.5,  8.9, 81.6), (12.5, 12.7, 100.0)),
    },
}

# wall-clock ratio OFF/ON -- what the overlap actually buys
SPEEDUP = {"L40S": [1.60, 1.62, 1.92, 2.11, 2.08],
           "B200": [1.53, 1.61, 1.70, 1.40, 1.26]}

# a* = max(SPU,SM)/(n+2): median alpha at which the NIC overtakes compute
ASTAR = {"L40S": 21.8, "B200": 4.1}               # ms

LANES = ["SPU (in bank)", "SM (NTT)", "NIC (network)"]
COLOR = ["#2E6F9E", "#7FB069", "#C1666B"]
PANELS = [("L40S", "L40S — 192 SPU"), ("B200", "B200 — 2048 SPU")]


def main(outpath="utilisation.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True,
                             gridspec_kw=dict(wspace=0.08))

    ngrp, nlane = len(ALPHA), len(LANES)
    inner = 0.36                       # width of one (off, on) pair
    span = 0.86                        # total width used per alpha group

    for ax, (key, title) in zip(axes, PANELS):
        for li, (lane, col) in enumerate(zip(LANES, COLOR)):
            base = np.arange(ngrp) + (li - (nlane - 1) / 2) * (span / nlane)
            off = [UTIL[key][a][0][li] for a in ALPHA]
            on = [UTIL[key][a][1][li] for a in ALPHA]
            w = span / nlane * inner
            ax.bar(base - w / 2, off, w, color=col, alpha=0.42,
                   edgecolor=col, linewidth=0.7,
                   label=f"{lane} — no co-sched" if ax is axes[0] else None)
            ax.bar(base + w / 2, on, w, color=col,
                   edgecolor="black", linewidth=0.35,
                   label=f"{lane} — co-scheduled" if ax is axes[0] else None)

        # a* marker: where the NIC lane overtakes compute
        astar = ASTAR[key]
        pos = np.interp(np.log10(astar * 1000),
                        np.log10(ALPHA), np.arange(ngrp))
        ax.axvline(pos, color="0.35", ls=(0, (4, 3)), lw=1.1, zorder=0)
        ax.text(pos, 104, f" $a^*$≈{astar:g} ms", fontsize=6.6,
                color="0.3", va="bottom", ha="left")

        ax.set_xticks(range(ngrp))
        ax.set_xticklabels([f"{x}\n{s:.2f}$\\times$" for x, s in zip(XLAB, SPEEDUP[key])],
                           fontsize=8)
        ax.set_xlabel(r"$\alpha$  /  wall-clock gain", fontsize=8, labelpad=4)
        ax.set_title(title, fontsize=9.5, pad=12)
        ax.set_ylim(0, 108)
        ax.grid(axis="y", lw=0.4, color="0.85", zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].set_ylabel("lane utilisation (%)", fontsize=8.5)
    axes[0].legend(fontsize=6.2, ncol=3, loc="upper center",
                   bbox_to_anchor=(1.04, -0.30), frameon=False,
                   columnspacing=1.1, handlelength=1.3)

    fig.text(0.5, -0.40,
             r"Per-exchange round trip $\alpha$ on the top tick line; the figure below "
             "it is the wall-clock gain from the co-schedule.\n"
             "Pale bar: three lanes serialised. Solid: co-scheduled, "
             r"wall $=\max(\mathrm{SPU},\mathrm{SM},\mathrm{NIC})$. Both gang the "
             "communication — same messages, same bytes.\n"
             r"$a^*=\max(\mathrm{SPU},\mathrm{SM})/(n{+}2)$ is where the NIC lane "
             "overtakes compute. Lanes run concurrently, so the three bars in a "
             "group need not sum to 100%.",
             ha="center", va="top", fontsize=6.4)

    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "utilisation.pdf")
