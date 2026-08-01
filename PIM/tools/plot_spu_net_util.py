#!/usr/bin/env python3
"""SPU and NIC utilisation with and without the SPU-network co-schedule,
sampled around each card's own lane crossover, with throughput on the right.

Isolating the two lanes the co-schedule touches:

    serial    :  wall = SPU + NIC        the SPUs idle through every exchange
    co-sched  :  wall = max(SPU, NIC)    the wait is filled with another batch

Both arms gang the communication, so the NIC carries the same n+2 messages and
the same bytes either way; the only difference is whether the wait overlaps.

The overlap is worth most where the two lanes balance, and that point moves with
the SPU budget: the NIC lane is (n+2)*alpha while the SPU lane is fixed, so
r = NIC/SPU scales as alpha/alpha*. Solving r = 1 at each card's built budget,

    L40S,  192 SPU :  alpha* = 11.69 ms
    B200, 2048 SPU :  alpha* =  1.75 ms          a 6.7x shift

Each card is therefore sampled on its own alpha range, chosen so that the
crossover falls between the second and third point: 5/10/15/20 ms on L40S and
0.5/1.5/2.5/3.5 ms on B200. On these aligned ranges the two rows track each
other closely, which a shared axis hides -- there B200 looks flat past 10 ms
only because its crossover is already six times behind.

Bars are lane utilisation on the left axis; the open and filled circles are
throughput before and after the co-schedule on the right axis, in millions of
OLE correlations per second. One expansion emits N of them, and N spans
2^20..2^24 across the suite, so expansions per second would average
incomparable quantities -- N/wall does not. Each is the geometric mean over the 25 (c,t,N) cells of the suite (22 on
L40S). The figures above each group are the gain of each lane in percentage
points; the two ratios are necessarily equal (scheduling moves only the wall, so
both utilisations scale by wall_serial/wall_sched) but the absolute gains are
not, because the lanes start from different levels.

Throughput gains are upper bounds: against four native-network ramulator2 runs
(ISR_NET_DELAY / ISR_NET_WAIT, L40S org, n=12) the max() model is exact once the
network dominates (+0.5% at r = 7.8) and optimistic by 24% at r = 1, where a
finite round count cannot fill the pipeline.

Single column, 4:3 -- 3.4 x 2.55 in.

    python3 plot_spu_net_util.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUILT = {"L40S": 192, "B200": 2048}
ASTAR = {"L40S": 11.693, "B200": 1.754}          # ms, where NIC = SPU

# machine -> the sampled round trips, in us. Round numbers rather than multiples
# of alpha*, but placed so that each card's crossover falls between the second
# and third point: r = 1 at 11.69 ms on L40S and 1.75 ms on B200.
ALPHA = {
    "L40S": [5000, 10000, 15000, 20000],
    "B200": [500, 1500, 2500, 3500],
}
XCROSS = {"L40S": 1.338, "B200": 1.254}          # crossover, in bar-index units

# machine -> per point,
#   (serial SPU%, serial NIC%, co-sched SPU%, co-sched NIC%, SPU +pp, NIC +pp)
# The two lanes' utilisation RATIOS are necessarily equal -- co-scheduling does
# not change either lane's busy time, only the wall, so both scale by
# wall_serial/wall_sched. What differs between the lanes is the absolute gain in
# percentage points, since they start from different levels; that is what is
# reported above each group.
UTIL = {
    "L40S": [(64.6, 27.6, 92.7, 39.7, 28.1, 12.0),
             (49.1, 42.0, 74.2, 63.4, 25.1, 21.4),
             (39.9, 51.2, 59.6, 76.5, 19.7, 25.3),
             (33.8, 57.8, 50.1, 85.6, 16.3, 27.9)],
    "B200": [(71.8, 20.5, 96.7, 27.6, 24.9, 7.1),
             (48.3, 41.3, 71.0, 60.7, 22.7, 19.4),
             (37.1, 52.8, 54.4, 77.5, 17.3, 24.7),
             (30.2, 60.3, 43.3, 86.5, 13.1, 26.2)],
}

# machine -> per multiple, (serial, co-scheduled) million OLE correlations/s.
# One expansion emits N of them and N spans 2^20..2^24 across the suite, so
# "expansions per second" would average incomparable quantities; N/wall does not.
TPUT = {
    "L40S": [(9.65, 13.84), (7.33, 11.07), (5.96, 8.90), (5.04, 7.47)],
    "B200": [(83.93, 113.03), (56.49, 82.97), (43.31, 63.52), (35.32, 50.63)],
}
TMAX = {"L40S": 19.0, "B200": 155.0}             # right-axis top, per card

SPU_C, NIC_C = "#2E6F9E", "#C1666B"
TP_C = "#3F3F3F"
# (arm, lane, colour, alpha, label)
SERIES = [
    (0, 0, SPU_C, 0.36, "SPU, serial"),
    (0, 1, NIC_C, 0.36, "NIC, serial"),
    (1, 0, SPU_C, 1.00, "SPU, co-sched"),
    (1, 1, NIC_C, 1.00, "NIC, co-sched"),
]


def fmt(us):
    return f"{us:.0f} µs" if us < 1000 else f"{us / 1000:.1f} ms"


def main(outpath="spu_net_util.pdf"):
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.55),
                             gridspec_kw=dict(hspace=0.60, left=0.125,
                                              right=0.845, top=0.995,
                                              bottom=0.285))

    ngrp, nser = len(ALPHA["L40S"]), len(SERIES)
    span, gap = 0.80, 0.025
    w = span / nser - gap

    for ax, mach in zip(axes, ("L40S", "B200")):
        for si, (arm, lane, col, al, lab) in enumerate(SERIES):
            x = np.arange(ngrp) + (si - (nser - 1) / 2) * (span / nser)
            y = [UTIL[mach][i][2 * arm + lane] for i in range(ngrp)]
            ax.bar(x, y, w, color=col, alpha=al,
                   edgecolor="black" if arm else col,
                   linewidth=0.3 if arm else 0.55, zorder=3,
                   label=lab if ax is axes[0] else None)

        for g in range(1, ngrp):
            ax.axvline(g - 0.5, color="0.91", lw=0.6, zorder=1)
        ax.axvline(XCROSS[mach], color="0.55", ls=(0, (2.5, 2)), lw=0.9, zorder=2)

        # per-lane gain in percentage points -- the two lanes differ here,
        # unlike their ratios
        for i in range(ngrp):
            ax.text(i - 0.20, 101, f"+{UTIL[mach][i][4]:.0f}", fontsize=5.6,
                    color=SPU_C, ha="center", va="bottom", zorder=6)
            ax.text(i + 0.20, 101, f"+{UTIL[mach][i][5]:.0f}", fontsize=5.6,
                    color=NIC_C, ha="center", va="bottom", zorder=6)

        # throughput before and after, right axis
        tx = ax.twinx()
        xs = np.arange(ngrp)
        ser = [TPUT[mach][i][0] for i in range(ngrp)]
        sch = [TPUT[mach][i][1] for i in range(ngrp)]
        tx.vlines(xs, ser, sch, color=TP_C, lw=0.7, zorder=6)
        tx.plot(xs, ser, "o", ms=3.2, mfc="white", mec=TP_C, mew=0.9,
                zorder=7, label="throughput, serial")
        tx.plot(xs, sch, "o", ms=3.2, mfc=TP_C, mec=TP_C, mew=0.9,
                zorder=7, label="throughput, co-sched")
        tx.set_ylim(0, TMAX[mach])
        tx.tick_params(labelsize=5.6, pad=1.0)
        tx.set_ylabel("M OLE / s", fontsize=6.0, labelpad=1.5)
        for sp in ("top", "left"):
            tx.spines[sp].set_visible(False)

        ax.text(1.135, 0.5, f"{mach} — {BUILT[mach]} SPU",
                transform=ax.transAxes, fontsize=6.0,
                va="center", ha="left", color="0.2", rotation=270)
        ax.set_xticks(range(ngrp))
        ax.set_xticklabels([fmt(a) for a in ALPHA[mach]], fontsize=6.4)
        ax.set_ylim(0, 128)
        ax.set_yticks([0, 50, 100])
        ax.set_xlim(-0.55, ngrp - 0.45)
        ax.grid(axis="y", lw=0.4, color="0.92", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.0, pad=1.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[1].set_xlabel(r"per-exchange round trip $\alpha$"
                       "     (dashed: that card's crossover)",
                       fontsize=6.8, labelpad=1.5)
    fig.text(0.016, 0.62, "lane utilisation (%)", fontsize=7.0,
             rotation=90, va="center", ha="center")

    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white",
                     mec=TP_C, mew=0.9),
          plt.Line2D([], [], ls="none", marker="o", ms=3.2, mfc=TP_C,
                     mec=TP_C, mew=0.9)]
    l += ["throughput, serial", "throughput, co-sched"]
    fig.legend(h, l, fontsize=5.4, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.075), frameon=False,
               columnspacing=0.9, handlelength=1.1, handletextpad=0.35)

    fig.savefig(outpath, bbox_inches="tight", dpi=300, pad_inches=0.02)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spu_net_util.pdf")
