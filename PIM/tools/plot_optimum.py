#!/usr/bin/env python3
"""Speedup, area and power at the buildable optimum, per network round trip.

The optimum is min(knee, ceiling): the knee is the smallest SPU budget within 2%
of the flat floor, and the ceiling is banks/2, forced by the double-buffered
expansion (level i reads bank i%2, writes (i+1)%2). L40S has 24 ch x 16 banks so
its ceiling is 192; B200 has 256 pc x 16 so 2048.

L40S sits at its ceiling for every alpha -- its knee is always beyond the card,
so the whole PIM budget is worth building. B200 tracks the knee once alpha
reaches 10 ms and sheds 25-75% of its area with no loss.

Area and power per channel (8 SPU + 1 DRU), from the synthesised breakdown:
    2.0339 mm^2  and  0.1210 W

    python3 plot_optimum.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALPHA = [100, 500, 10000, 20000, 30000]
XLAB = ["100 µs", "500 µs", "10 ms", "20 ms", "30 ms"]

# alpha -> machine -> (SPU, speedup, runtime ms, area mm2, power W, unclamped knee)
OPT = {
    100:   {"L40S": (192, 2.60, 556.2, 48.8, 2.91, 3072),
            "B200": (2048, 7.07, 81.5, 520.7, 30.99, 8192)},
    500:   {"L40S": (192, 2.61, 556.2, 48.8, 2.91, 3072),
            "B200": (2048, 7.20, 81.5, 520.7, 30.99, 8192)},
    10000: {"L40S": (192, 3.05, 556.2, 48.8, 2.91, 384),
            "B200": (1536, 3.60, 227.7, 390.5, 23.24, 1536)},
    20000: {"L40S": (192, 3.08, 632.3, 48.8, 2.91, 384),
            "B200": (768, 2.40, 438.5, 195.3, 11.62, 768)},
    30000: {"L40S": (192, 2.76, 791.4, 48.8, 2.91, 256),
            "B200": (512, 1.94, 657.8, 130.2, 7.75, 512)},
}
FULL_B200 = (520.7, 30.99)                 # always-2048 reference

COL = {"L40S": "#2E6F9E", "B200": "#C1666B"}
PANELS = [
    (1, "(a)  end-to-end speedup", r"speedup over the ganged-GPU baseline", "{:.2f}×"),
    (3, "(b)  PIM logic area", "mm$^2$", "{:.0f}"),
    (4, "(c)  PIM power", "W", "{:.1f}"),
]


def main(outpath="optimum.pdf"):
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.95),
                             gridspec_kw=dict(wspace=0.34))

    n = len(ALPHA)
    w = 0.36
    for ax, (idx, title, ylab, fmt) in zip(axes, PANELS):
        for k, mach in enumerate(("L40S", "B200")):
            x = np.arange(n) + (k - 0.5) * w
            y = [OPT[a][mach][idx] for a in ALPHA]
            ax.bar(x, y, w * 0.92, color=COL[mach], zorder=3,
                   edgecolor="black", linewidth=0.35,
                   label=mach if ax is axes[0] else None)
            for xi, yi, a in zip(x, y, ALPHA):
                ax.text(xi, yi, fmt.format(yi), ha="center", va="bottom",
                        fontsize=5.6, color="0.15", zorder=5)
                if idx == 1:                       # SPU budget, on the speedup panel only
                    ax.text(xi, yi * 0.5, str(OPT[a][mach][0]), ha="center",
                            va="center", fontsize=5.2, color="white",
                            rotation=90, zorder=6)

        # on the area/power panels, mark what an always-2048 B200 would cost
        if idx in (3, 4):
            ref = FULL_B200[0 if idx == 3 else 1]
            ax.axhline(ref, color=COL["B200"], ls=(0, (3, 2)), lw=1.0, zorder=2)
            ax.text(n - 0.5, ref, "  B200 fixed 2048", fontsize=5.8,
                    color=COL["B200"], va="bottom", ha="right")

        ax.set_xticks(range(n))
        ax.set_xticklabels(XLAB, fontsize=6.8, rotation=30, ha="right")
        ax.set_xlabel(r"round trip $\alpha$", fontsize=7.6, labelpad=1)
        ax.set_ylabel(ylab, fontsize=7.6)
        ax.set_title(title, fontsize=8.4, pad=4, loc="left")
        ax.grid(axis="y", lw=0.4, color="0.89", zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6.8)
        ax.margins(y=0.20)

    axes[0].legend(fontsize=6.6, frameon=False, loc="upper center",
                   ncol=2, bbox_to_anchor=(0.5, 1.02), columnspacing=1.0)

    fig.text(0.5, -0.16,
             "White figure inside each bar is the SPU budget chosen at that α, "
             "namely min(knee, banks/2 ceiling). L40S never reaches "
             "its knee — 192 SPU is worth building at every α, so its area and "
             "power are flat and\nonly the speedup moves. B200 reaches its knee "
             "from 10 ms on and sheds 25 / 63 / 75% of its area at 10 / 20 / 30 ms "
             "for under 2% of the runtime. Its\nspeedup is the smaller number "
             "only because its GPU baseline is faster: in absolute runtime B200 "
             "leads at every α (81.5 ms against 556.2 at 100 µs).",
             ha="center", va="top", fontsize=6.2)

    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "optimum.pdf")
