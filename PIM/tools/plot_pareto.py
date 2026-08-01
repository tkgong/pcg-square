#!/usr/bin/env python3
"""Design-space exploration: how many SPUs are worth building at a given network RTT.

The SPU lane shrinks as 1/N_SPU; the NTT lane and the NIC lane do not move. Past
the point where the SPU lane drops below max(SM, NIC), extra SPUs buy nothing
and the silicon idles. The knee is the smallest design within 2% of the
full-card result.

Area and power per channel come from the synthesised breakdown (8 SPU + 1 DRU):

    area  = 8 x 0.251120 + 0.024924 = 2.0339 mm^2
    power = 8 x 0.014948 + 0.001469 = 0.1210 W

L40S tops out at 24 channels (192 SPU) and B200 at 256 (2048). Points beyond a
card's channel count are not drawn -- they would need a different memory
organisation, not a different PIM budget.

    python3 plot_pareto.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- data: (channels, SPU, area mm2, power W, speedup) ---------------------
# L40S, 192 SPU ceiling
L40S = {
    100: [(1,8,2.034,0.1211,0.1255),(2,16,4.068,0.2421,0.2511),(3,24,6.102,0.3632,0.3766),(4,32,8.136,0.4842,0.5022),(6,48,12.203,0.7263,0.7533),(8,64,16.271,0.9684,1.0043),(12,96,24.407,1.4527,1.4987),(16,128,32.542,1.9369,1.8866),(20,160,40.678,2.4211,2.2553),(24,192,48.813,2.9053,2.5959)],
    500: [(1,8,2.034,0.1211,0.1265),(2,16,4.068,0.2421,0.2529),(3,24,6.102,0.3632,0.3794),(4,32,8.136,0.4842,0.5058),(6,48,12.203,0.7263,0.7587),(8,64,16.271,0.9684,1.0116),(12,96,24.407,1.4527,1.5096),(16,128,32.542,1.9369,1.9003),(20,160,40.678,2.4211,2.2717),(24,192,48.813,2.9053,2.6148)],
    10000: [(1,8,2.034,0.1211,0.1476),(2,16,4.068,0.2421,0.2953),(3,24,6.102,0.3632,0.4429),(4,32,8.136,0.4842,0.5905),(6,48,12.203,0.7263,0.8858),(8,64,16.271,0.9684,1.181),(12,96,24.407,1.4527,1.7624),(16,128,32.542,1.9369,2.2184),(20,160,40.678,2.4211,2.652),(24,192,48.813,2.9053,3.0526)],
    20000: [(1,8,2.034,0.1211,0.1691),(2,16,4.068,0.2421,0.3383),(3,24,6.102,0.3632,0.5074),(4,32,8.136,0.4842,0.6766),(6,48,12.203,0.7263,1.0149),(8,64,16.271,0.9684,1.3532),(12,96,24.407,1.4527,2.0131),(16,128,32.542,1.9369,2.4274),(20,160,40.678,2.4211,2.7752),(24,192,48.813,2.9053,3.0767)],
    30000: [(1,8,2.034,0.1211,0.1902),(2,16,4.068,0.2421,0.3803),(3,24,6.102,0.3632,0.5705),(4,32,8.136,0.4842,0.7606),(6,48,12.203,0.7263,1.141),(8,64,16.271,0.9684,1.5161),(12,96,24.407,1.4527,2.061),(16,128,32.542,1.9369,2.4459),(20,160,40.678,2.4211,2.645),(24,192,48.813,2.9053,2.7635)],
}
# B200, 2048 SPU ceiling
B200 = {
    100: [(8,64,16.271,0.9684,0.3017),(16,128,32.542,1.9369,0.6035),(24,192,48.813,2.9053,0.9052),(32,256,65.084,3.8737,1.2069),(48,384,97.627,5.8106,1.8104),(64,512,130.169,7.7475,2.4139),(96,768,195.253,11.6212,3.4434),(128,1024,260.338,15.495,4.3763),(192,1536,390.506,23.2425,5.8348),(256,2048,520.675,30.99,7.0684)],
    500: [(8,64,16.271,0.9684,0.3072),(16,128,32.542,1.9369,0.6143),(24,192,48.813,2.9053,0.9215),(32,256,65.084,3.8737,1.2286),(48,384,97.627,5.8106,1.8429),(64,512,130.169,7.7475,2.4572),(96,768,195.253,11.6212,3.5052),(128,1024,260.338,15.495,4.4548),(192,1536,390.506,23.2425,5.9396),(256,2048,520.675,30.99,7.1953)],
    10000: [(8,64,16.271,0.9684,0.4293),(16,128,32.542,1.9369,0.8586),(24,192,48.813,2.9053,1.2878),(32,256,65.084,3.8737,1.6791),(48,384,97.627,5.8106,2.3016),(64,512,130.169,7.7475,2.7628),(96,768,195.253,11.6212,3.257),(128,1024,260.338,15.495,3.5059),(192,1536,390.506,23.2425,3.6003),(256,2048,520.675,30.99,3.6473)],
    20000: [(8,64,16.271,0.9684,0.5512),(16,128,32.542,1.9369,1.0779),(24,192,48.813,2.9053,1.4775),(32,256,65.084,3.8737,1.7736),(48,384,97.627,5.8106,2.1299),(64,512,130.169,7.7475,2.2939),(96,768,195.253,11.6212,2.3996),(128,1024,260.338,15.495,2.4378),(192,1536,390.506,23.2425,2.4378),(256,2048,520.675,30.99,2.4378)],
    30000: [(8,64,16.271,0.9684,0.6701),(16,128,32.542,1.9369,1.1975),(24,192,48.813,2.9053,1.519),(32,256,65.084,3.8737,1.7262),(48,384,97.627,5.8106,1.8837),(64,512,130.169,7.7475,1.9449),(96,768,195.253,11.6212,1.9758),(128,1024,260.338,15.495,1.9758),(192,1536,390.506,23.2425,1.9758),(256,2048,520.675,30.99,1.9758)],
}

ALPHA = [100, 500, 10000, 20000, 30000]
LBL = {100: "100 µs", 500: "500 µs", 10000: "10 ms", 20000: "20 ms", 30000: "30 ms"}
CMAP = {100: "#1F4E79", 500: "#2E86AB", 10000: "#7FB069", 20000: "#E0A458", 30000: "#C1666B"}
PANELS = [("L40S — 24 ch ceiling (192 SPU)", L40S, 192),
          ("B200 — 256 ch ceiling (2048 SPU)", B200, 2048)]


def knee(pts):
    """smallest design within 2% of the largest."""
    full = pts[-1][4]
    return next(p for p in pts if p[4] >= 0.98 * full)


def main(outpath="pareto.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2),
                             gridspec_kw=dict(wspace=0.24))

    for ax, (title, D, cap) in zip(axes, PANELS):
        for a in ALPHA:
            pts = D[a]
            x = [p[2] for p in pts]
            y = [p[4] for p in pts]
            ax.plot(x, y, "-o", color=CMAP[a], ms=2.6, lw=1.2,
                    label=LBL[a] if ax is axes[0] else None, zorder=3)
            k = knee(pts)
            if k[1] < cap:                      # a real knee, not the ceiling
                ax.plot(k[2], k[4], "*", color=CMAP[a], ms=11,
                        mec="black", mew=0.5, zorder=5)
                ax.annotate(f"{k[1]}", (k[2], k[4]), fontsize=6.2,
                            xytext=(3, -8), textcoords="offset points",
                            color=CMAP[a], fontweight="bold")

        ax.set_xlabel("PIM logic area (mm$^2$)", fontsize=8.5)
        ax.set_title(title, fontsize=9, pad=5)
        ax.grid(lw=0.4, color="0.88", zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7.5)

        # secondary axis: SPU count
        sec = ax.secondary_xaxis(
            "top", functions=(lambda v: v / 2.0339 * 8, lambda s: s / 8 * 2.0339))
        sec.set_xlabel("SPU count", fontsize=7.5, labelpad=2)
        sec.tick_params(labelsize=6.8)

    axes[0].set_ylabel("end-to-end speedup", fontsize=8.5)
    axes[0].legend(title=r"round trip $\alpha$", fontsize=7, title_fontsize=7.2,
                   loc="upper left", frameon=False)

    fig.text(0.5, -0.09,
             "★ = knee, the smallest design within 2% of the full card; the label is its SPU count. "
             "L40S shows no knee — at 24 channels\nit never reaches the network-bound regime, so its "
             "whole design space is worth building. B200 does: at 20 ms, 768 SPU delivers 98% of what "
             "2048 does\nfor 37% of the area. Curves for the same α are directly comparable; "
             "area is PIM logic only, not the DRAM it sits in.",
             ha="center", va="top", fontsize=6.4)

    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pareto.pdf")
