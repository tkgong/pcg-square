#!/usr/bin/env python3
"""Combined figure: what the co-schedule buys, and how much SPU is worth building.

(a) Lane utilisation with and without the SPU-network co-schedule. Eight bars per
    group: {L40S, B200} x {off, on} x {SPU, NIC}.

        off :  wall = SPU + NIC        the SPU idles through every exchange
        on  :  wall = max(SPU, NIC)    the wait is filled with another batch

(b,c) Design-space exploration. The SPU lane shrinks as 1/N_SPU while the NTT and
    NIC lanes do not move, so past the point where the SPU lane drops below
    max(SM, NIC) extra SPUs buy nothing. Area per channel (8 SPU + 1 DRU) is
    8 x 0.251120 + 0.024924 = 2.0339 mm^2 from the synthesised breakdown.

Both configurations gang the communication, so the NIC carries the same n+2
messages and the same bytes throughout. Values are the mean over the measured
cells (L40S 15, B200 18; logN 22-24 across the six BCG+20 security rows).

    python3 plot_util_pareto.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALPHA = [100, 500, 10000, 20000, 30000]
XLAB = ["100 µs", "500 µs", "10 ms", "20 ms", "30 ms"]
LBL = dict(zip(ALPHA, XLAB))

# ---- (a) utilisation: alpha -> machine -> (off(SPU,NIC), on(SPU,NIC), gain) --
UTIL = {
    100:   {"L40S": ((99.5,  0.5), (100.0,  0.5), 1.00),
            "B200": ((96.3,  3.7), (100.0,  3.9), 1.03)},
    500:   {"L40S": ((97.4,  2.6), (100.0,  2.7), 1.02),
            "B200": ((84.5, 15.5), (100.0, 19.5), 1.13)},
    10000: {"L40S": ((67.4, 32.6), ( 99.7, 52.9), 1.38),
            "B200": ((25.4, 74.6), ( 37.6,100.0), 1.38)},
    20000: {"L40S": ((52.0, 48.0), ( 86.5, 81.1), 1.60),
            "B200": ((15.1, 84.9), ( 18.8,100.0), 1.19)},
    30000: {"L40S": ((42.6, 57.4), ( 70.2, 90.9), 1.61),
            "B200": ((10.7, 89.3), ( 12.5,100.0), 1.13)},
}
SPU_L, NIC_L = "#2E6F9E", "#C1666B"
SPU_B, NIC_B = "#4E9F3D", "#E0A458"
SERIES = [
    ("L40S", 0, 0, SPU_L, 0.40, "L40S SPU, no co-sched"),
    ("L40S", 0, 1, NIC_L, 0.40, "L40S NIC, no co-sched"),
    ("L40S", 1, 0, SPU_L, 1.00, "L40S SPU, co-scheduled"),
    ("L40S", 1, 1, NIC_L, 1.00, "L40S NIC, co-scheduled"),
    ("B200", 0, 0, SPU_B, 0.40, "B200 SPU, no co-sched"),
    ("B200", 0, 1, NIC_B, 0.40, "B200 NIC, no co-sched"),
    ("B200", 1, 0, SPU_B, 1.00, "B200 SPU, co-scheduled"),
    ("B200", 1, 1, NIC_B, 1.00, "B200 NIC, co-scheduled"),
]

# ---- (b,c) Pareto: (channels, SPU, area mm2, power W, speedup) --------------
L40S = {
    100: [(1,8,2.034,0.1211,0.1255),(2,16,4.068,0.2421,0.2511),(3,24,6.102,0.3632,0.3766),(4,32,8.136,0.4842,0.5022),(6,48,12.203,0.7263,0.7533),(8,64,16.271,0.9684,1.0043),(12,96,24.407,1.4527,1.4987),(16,128,32.542,1.9369,1.8866),(20,160,40.678,2.4211,2.2553),(24,192,48.813,2.9053,2.5959)],
    500: [(1,8,2.034,0.1211,0.1265),(2,16,4.068,0.2421,0.2529),(3,24,6.102,0.3632,0.3794),(4,32,8.136,0.4842,0.5058),(6,48,12.203,0.7263,0.7587),(8,64,16.271,0.9684,1.0116),(12,96,24.407,1.4527,1.5096),(16,128,32.542,1.9369,1.9003),(20,160,40.678,2.4211,2.2717),(24,192,48.813,2.9053,2.6148)],
    10000: [(1,8,2.034,0.1211,0.1476),(2,16,4.068,0.2421,0.2953),(3,24,6.102,0.3632,0.4429),(4,32,8.136,0.4842,0.5905),(6,48,12.203,0.7263,0.8858),(8,64,16.271,0.9684,1.181),(12,96,24.407,1.4527,1.7624),(16,128,32.542,1.9369,2.2184),(20,160,40.678,2.4211,2.652),(24,192,48.813,2.9053,3.0526)],
    20000: [(1,8,2.034,0.1211,0.1691),(2,16,4.068,0.2421,0.3383),(3,24,6.102,0.3632,0.5074),(4,32,8.136,0.4842,0.6766),(6,48,12.203,0.7263,1.0149),(8,64,16.271,0.9684,1.3532),(12,96,24.407,1.4527,2.0131),(16,128,32.542,1.9369,2.4274),(20,160,40.678,2.4211,2.7752),(24,192,48.813,2.9053,3.0767)],
    30000: [(1,8,2.034,0.1211,0.1902),(2,16,4.068,0.2421,0.3803),(3,24,6.102,0.3632,0.5705),(4,32,8.136,0.4842,0.7606),(6,48,12.203,0.7263,1.141),(8,64,16.271,0.9684,1.5161),(12,96,24.407,1.4527,2.061),(16,128,32.542,1.9369,2.4459),(20,160,40.678,2.4211,2.645),(24,192,48.813,2.9053,2.7635)],
}
B200 = {
    100: [(8,64,16.271,0.9684,0.3017),(16,128,32.542,1.9369,0.6035),(24,192,48.813,2.9053,0.9052),(32,256,65.084,3.8737,1.2069),(48,384,97.627,5.8106,1.8104),(64,512,130.169,7.7475,2.4139),(96,768,195.253,11.6212,3.4434),(128,1024,260.338,15.495,4.3763),(192,1536,390.506,23.2425,5.8348),(256,2048,520.675,30.99,7.0684)],
    500: [(8,64,16.271,0.9684,0.3072),(16,128,32.542,1.9369,0.6143),(24,192,48.813,2.9053,0.9215),(32,256,65.084,3.8737,1.2286),(48,384,97.627,5.8106,1.8429),(64,512,130.169,7.7475,2.4572),(96,768,195.253,11.6212,3.5052),(128,1024,260.338,15.495,4.4548),(192,1536,390.506,23.2425,5.9396),(256,2048,520.675,30.99,7.1953)],
    10000: [(8,64,16.271,0.9684,0.4293),(16,128,32.542,1.9369,0.8586),(24,192,48.813,2.9053,1.2878),(32,256,65.084,3.8737,1.6791),(48,384,97.627,5.8106,2.3016),(64,512,130.169,7.7475,2.7628),(96,768,195.253,11.6212,3.257),(128,1024,260.338,15.495,3.5059),(192,1536,390.506,23.2425,3.6003),(256,2048,520.675,30.99,3.6473)],
    20000: [(8,64,16.271,0.9684,0.5512),(16,128,32.542,1.9369,1.0779),(24,192,48.813,2.9053,1.4775),(32,256,65.084,3.8737,1.7736),(48,384,97.627,5.8106,2.1299),(64,512,130.169,7.7475,2.2939),(96,768,195.253,11.6212,2.3996),(128,1024,260.338,15.495,2.4378),(192,1536,390.506,23.2425,2.4378),(256,2048,520.675,30.99,2.4378)],
    30000: [(8,64,16.271,0.9684,0.6701),(16,128,32.542,1.9369,1.1975),(24,192,48.813,2.9053,1.519),(32,256,65.084,3.8737,1.7262),(48,384,97.627,5.8106,1.8837),(64,512,130.169,7.7475,1.9449),(96,768,195.253,11.6212,1.9758),(128,1024,260.338,15.495,1.9758),(192,1536,390.506,23.2425,1.9758),(256,2048,520.675,30.99,1.9758)],
}
CMAP = {100: "#1F4E79", 500: "#2E86AB", 10000: "#7FB069", 20000: "#E0A458", 30000: "#C1666B"}


def knee(pts):
    full = pts[-1][4]
    return next(p for p in pts if p[4] >= 0.98 * full)


def main(outpath="util_pareto.pdf"):
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0],
                          hspace=0.72, wspace=0.26)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])

    # ---------- (a) utilisation ----------
    ngrp, nser = len(ALPHA), len(SERIES)
    span, gap = 0.88, 0.012
    w = span / nser - gap
    for si, (mach, sched, lane, col, al, lab) in enumerate(SERIES):
        x = np.arange(ngrp) + (si - (nser - 1) / 2) * (span / nser)
        y = [UTIL[a][mach][sched][lane] for a in ALPHA]
        axa.bar(x, y, w, color=col, alpha=al, label=lab,
                edgecolor="black" if sched else col,
                linewidth=0.4 if sched else 0.7, zorder=3)
    for g in range(ngrp):
        axa.axvline(g, color="0.90", lw=0.8, zorder=1)
    axa.set_xticks(range(ngrp))
    axa.set_xticklabels(
        [f"{x}\n{UTIL[a]['L40S'][2]:.2f} / {UTIL[a]['B200'][2]:.2f}"
         for x, a in zip(XLAB, ALPHA)], fontsize=7.6)
    axa.set_xlabel(r"per-exchange round trip $\alpha$"
                   "   (below: wall-clock gain, L40S / B200)",
                   fontsize=8, labelpad=3)
    axa.set_ylabel("lane utilisation (%)", fontsize=8.5)
    axa.set_ylim(0, 108)
    axa.set_title("(a)  what the SPU–network co-schedule recovers",
                  fontsize=9, pad=6, loc="left")
    axa.grid(axis="y", lw=0.4, color="0.88", zorder=0)
    axa.set_axisbelow(True)
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)
    axa.tick_params(labelsize=7.5)
    axa.legend(fontsize=5.9, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.38), frameon=False,
               columnspacing=0.9, handlelength=1.1)

    # ---------- (b,c) Pareto ----------
    for ax, (tag, title, D, cap) in zip(
            (axb, axc),
            [("(b)", "L40S — 24 ch ceiling (192 SPU)", L40S, 192),
             ("(c)", "B200 — 256 ch ceiling (2048 SPU)", B200, 2048)]):
        for a in ALPHA:
            pts = D[a]
            ax.plot([p[2] for p in pts], [p[4] for p in pts], "-o",
                    color=CMAP[a], ms=2.4, lw=1.1,
                    label=LBL[a] if ax is axb else None, zorder=3)
            k = knee(pts)
            if k[1] < cap:
                ax.plot(k[2], k[4], "*", color=CMAP[a], ms=10,
                        mec="black", mew=0.45, zorder=5)
                ax.annotate(f"{k[1]}", (k[2], k[4]), fontsize=6,
                            xytext=(3, -8.5), textcoords="offset points",
                            color=CMAP[a], fontweight="bold")
        ax.set_xlabel("PIM logic area (mm$^2$)", fontsize=8.5)
        ax.set_title(f"{tag}  {title}", fontsize=9, pad=14, loc="left")
        ax.grid(lw=0.4, color="0.89", zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7.2)
        sec = ax.secondary_xaxis(
            "top", functions=(lambda v: v / 2.0339 * 8, lambda s: s / 8 * 2.0339))
        sec.set_xlabel("SPU count", fontsize=7, labelpad=1.5)
        sec.tick_params(labelsize=6.4)
    axb.set_ylabel("end-to-end speedup", fontsize=8.5)
    axb.legend(title=r"$\alpha$", fontsize=6.4, title_fontsize=6.8,
               loc="upper left", frameon=False, labelspacing=0.25)

    fig.text(0.5, 0.015,
             "(a) Pale: SPU and NIC serialised. Solid: co-scheduled, "
             r"wall $=\max(\mathrm{SPU},\mathrm{NIC})$. Lanes run concurrently, "
             "so a pair need not sum to 100%.\n"
             "(b,c) ★ = knee, the smallest design within 2% of the full card, "
             "labelled with its SPU count. L40S has none — 24 channels never reach\n"
             "the network-bound regime. B200 does: at 20 ms, 768 SPU delivers 98% "
             "of what 2048 does for 37% of the area. Both gang the communication.",
             ha="center", va="top", fontsize=6.2)

    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "util_pareto.pdf")
