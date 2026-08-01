#!/usr/bin/env python3
"""DRU and kernel-pipeline ablation: what the DRU buys, and what decides it.

The DRU streams the transpose and bit-reversal tiles of the four-step NTT out of
the SM timeline. Its effect on the SM lane is a constant, measured from the
per-stage NTT breakdown:

    L40S :  those stages are 34.5-48.6% of the NTT (mean 39.1%), but GDDR6
            leaves the SM already bandwidth-bound with no shadow bandwidth to
            move the tiles into, so the lane gain is booked at 1.000x
    B200 :  24.4-28.6% (mean 25.4%) of the NTT, and HBM3e has the headroom:
            the SM lane drops 29.4 -> 21.9 ms, a 1.342x lane gain

What that buys end to end is decided by the kernel pipeline, which is why the
two are ablated together rather than separately:

    pipeline off :  wall = SPU + SM + NIC   every lane contributes to a sum, so
                    the DRU always shows something (1.02-1.11x on B200)
    pipeline on  :  wall = max(SPU, SM, NIC)  only the binding lane counts, so
                    the DRU is worth 1.156x where ten of twenty-five cells are
                    SM-bound and exactly nothing where none are

The pipeline is therefore not a peer knob but a gate: it amplifies the DRU where
the SM binds and nullifies it where it does not. The right axis carries the
count of SM-bound cells, which is what closes that causal loop inside the figure
-- the DRU's end-to-end gain tracks it, not the lane gain.

Bars are the geometric mean over the 25 (c,t,N) cells of the suite (22 on L40S)
at each card's built budget, 192 SPU on L40S and 2048 on B200.

Single column, 4:3 -- 3.4 x 2.55 in.

    python3 plot_dru_pipe.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUILT = {"L40S": 192, "B200": 2048}
NCELL = {"L40S": 22, "B200": 25}
ALPHA = [500, 2000, 10000, 30000]
XLAB = ["500 µs", "2 ms", "10 ms", "30 ms"]

# machine -> per alpha, e2e ms:
#   (pipe-off -DRU, pipe-off +DRU, pipe-on -DRU, pipe-on +DRU)
E2E = {
    "L40S": [(414.0, 414.0, 278.0, 278.0),
             (459.3, 459.3, 278.0, 278.0),
             (671.9, 671.9, 357.5, 357.5),
             (1139.2, 1139.2, 691.5, 691.5)],
    "B200": [(90.1, 80.9, 53.4, 46.2),
             (130.3, 120.2, 73.9, 66.9),
             (311.5, 298.8, 215.5, 210.7),
             (731.8, 716.8, 614.0, 614.0)],
}

# machine -> per alpha, how many cells have the SM as the binding lane
SMBOUND = {"L40S": [8, 8, 6, 4], "B200": [10, 6, 2, 0]}

OFF_C, ON_C = "#8AA6BF", "#2E6F9E"       # pipeline off / on
SM_C = "#C1666B"                          # SM-bound count
# (pipeline arm, dru, colour, alpha, hatch, label)
SERIES = [
    (0, 0, OFF_C, 1.00, None, "no pipeline, no DRU"),
    (0, 1, OFF_C, 1.00, "///", "no pipeline, DRU"),
    (1, 0, ON_C, 1.00, None, "pipeline, no DRU"),
    (1, 1, ON_C, 1.00, "///", "pipeline, DRU"),
]


def main(outpath="dru_pipe.pdf"):
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.55),
                             gridspec_kw=dict(hspace=0.60, left=0.135,
                                              right=0.845, top=0.995,
                                              bottom=0.285))

    ngrp, nser = len(ALPHA), len(SERIES)
    span, gap = 0.80, 0.025
    w = span / nser - gap

    for ax, mach in zip(axes, ("L40S", "B200")):
        for si, (arm, dru, col, al, hat, lab) in enumerate(SERIES):
            x = np.arange(ngrp) + (si - (nser - 1) / 2) * (span / nser)
            y = [E2E[mach][i][2 * arm + dru] for i in range(ngrp)]
            ax.bar(x, y, w, color=col, alpha=al, hatch=hat,
                   edgecolor="black", linewidth=0.3, zorder=3,
                   label=lab if ax is axes[0] else None)

        for g in range(1, ngrp):
            ax.axvline(g - 0.5, color="0.91", lw=0.6, zorder=1)

        # what the DRU is worth end to end, once the pipeline is on
        lo = min(min(E2E[mach][i]) for i in range(ngrp))
        hi = max(max(E2E[mach][i]) for i in range(ngrp))
        for i in range(ngrp):
            a, b = E2E[mach][i][2], E2E[mach][i][3]
            ax.text(i, hi * 1.55, f"{a / b:.2f}×", fontsize=5.6,
                    color=ON_C if a != b else "0.55",
                    ha="center", va="bottom", zorder=6)

        # SM-bound cell count on the right -- the DRU's gain tracks this
        tx = ax.twinx()
        tx.plot(np.arange(ngrp), SMBOUND[mach], "-s", ms=3.0, lw=1.2,
                color=SM_C, mec="white", mew=0.6, zorder=7)
        tx.set_ylim(0, NCELL[mach] * 2.6)
        tx.set_yticks([0, NCELL[mach] // 2, NCELL[mach]])
        tx.tick_params(labelsize=5.6, pad=1.0, colors=SM_C)
        tx.set_ylabel("SM-bound cells", fontsize=5.8, labelpad=1.5, color=SM_C)
        for sp in ("top", "left"):
            tx.spines[sp].set_visible(False)

        ax.text(1.155, 0.5, f"{mach} — {BUILT[mach]} SPU",
                transform=ax.transAxes, fontsize=6.0,
                va="center", ha="left", color="0.2", rotation=270)
        ax.set_xticks(range(ngrp))
        ax.set_xticklabels(XLAB, fontsize=6.4)
        ax.set_yscale("log")
        ax.set_ylim(lo / 1.7, hi * 3.2)
        ax.set_xlim(-0.55, ngrp - 0.45)
        ax.grid(axis="y", lw=0.4, color="0.92", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.0, pad=1.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[1].set_xlabel(r"per-exchange round trip $\alpha$",
                       fontsize=6.8, labelpad=1.5)
    fig.text(0.022, 0.62, "runtime per expansion (ms)", fontsize=7.0,
             rotation=90, va="center", ha="center")

    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="s", ms=3.0, lw=1.2, color=SM_C,
                        mec="white", mew=0.6))
    l.append("SM-bound cells")
    fig.legend(h, l, fontsize=5.4, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.075), frameon=False,
               columnspacing=0.9, handlelength=1.3, handletextpad=0.35)

    fig.savefig(outpath, bbox_inches="tight", dpi=300, pad_inches=0.02)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dru_pipe.pdf")
