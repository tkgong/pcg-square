#!/usr/bin/env python3
"""Design-space exploration, both machines on one axis.

The SPU lane shrinks as 1/N_SPU while the NTT and NIC lanes do not move, so past
the point where the SPU lane drops below max(SM, NIC) extra SPUs buy nothing.
The knee is the smallest budget within 2% of the 8192-SPU limit.

Solid = buildable on that card (L40S 24 channels = 192 SPU, B200 256 = 2048).
Dashed = beyond the channel count; it would need a different memory
organisation, not just a bigger PIM budget.

Area per channel (8 SPU + 1 DRU) is 8 x 0.251120 + 0.024924 = 2.0339 mm^2 from
the synthesised breakdown. Both sides gang the communication throughout.

    python3 plot_dse.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (SPU, area mm2, speedup)
L40S = {
    100: [[8, 2.034, 0.1255], [16, 4.068, 0.2511], [32, 8.136, 0.5022], [64, 16.271, 1.0043], [128, 32.542, 1.8866], [192, 48.813, 2.5959], [256, 65.084, 3.085], [384, 97.627, 3.8691], [512, 130.169, 4.0983], [768, 195.253, 4.4444], [1024, 260.338, 4.7077], [1536, 390.506, 5.0621], [2048, 520.675, 5.1601], [3072, 781.013, 5.2848], [4096, 1041.351, 5.2848], [6144, 1562.026, 5.2848], [8192, 2082.701, 5.2848]],
    500: [[8, 2.034, 0.1265], [16, 4.068, 0.2529], [32, 8.136, 0.5058], [64, 16.271, 1.0116], [128, 32.542, 1.9003], [192, 48.813, 2.6148], [256, 65.084, 3.1075], [384, 97.627, 3.8973], [512, 130.169, 4.1281], [768, 195.253, 4.4768], [1024, 260.338, 4.7419], [1536, 390.506, 5.0989], [2048, 520.675, 5.1977], [3072, 781.013, 5.3232], [4096, 1041.351, 5.3232], [6144, 1562.026, 5.3232], [8192, 2082.701, 5.3232]],
    10000: [[8, 2.034, 0.1476], [16, 4.068, 0.2953], [32, 8.136, 0.5905], [64, 16.271, 1.181], [128, 32.542, 2.2184], [192, 48.813, 3.0526], [256, 65.084, 3.4751], [384, 97.627, 4.0438], [512, 130.169, 4.1205], [768, 195.253, 4.1205], [1024, 260.338, 4.1205], [1536, 390.506, 4.1205], [2048, 520.675, 4.1205], [3072, 781.013, 4.1205], [4096, 1041.351, 4.1205], [6144, 1562.026, 4.1205], [8192, 2082.701, 4.1205]],
    20000: [[8, 2.034, 0.1691], [16, 4.068, 0.3383], [32, 8.136, 0.6766], [64, 16.271, 1.3532], [128, 32.542, 2.4274], [192, 48.813, 3.0767], [256, 65.084, 3.2576], [384, 97.627, 3.4151], [512, 130.169, 3.4151], [768, 195.253, 3.4151], [1024, 260.338, 3.4151], [1536, 390.506, 3.4151], [2048, 520.675, 3.4151], [3072, 781.013, 3.4151], [4096, 1041.351, 3.4151], [6144, 1562.026, 3.4151], [8192, 2082.701, 3.4151]],
    30000: [[8, 2.034, 0.1902], [16, 4.068, 0.3803], [32, 8.136, 0.7606], [64, 16.271, 1.5161], [128, 32.542, 2.4459], [192, 48.813, 2.7635], [256, 65.084, 2.8519], [384, 97.627, 2.8519], [512, 130.169, 2.8519], [768, 195.253, 2.8519], [1024, 260.338, 2.8519], [1536, 390.506, 2.8519], [2048, 520.675, 2.8519], [3072, 781.013, 2.8519], [4096, 1041.351, 2.8519], [6144, 1562.026, 2.8519], [8192, 2082.701, 2.8519]],
}
B200 = {
    100: [[8, 2.034, 0.0377], [16, 4.068, 0.0754], [32, 8.136, 0.1509], [64, 16.271, 0.3017], [128, 32.542, 0.6035], [192, 48.813, 0.9052], [256, 65.084, 1.2069], [384, 97.627, 1.8104], [512, 130.169, 2.4139], [768, 195.253, 3.4434], [1024, 260.338, 4.3763], [1536, 390.506, 5.8348], [2048, 520.675, 7.0684], [3072, 781.013, 8.3398], [4096, 1041.351, 9.1791], [6144, 1562.026, 10.5075], [8192, 2082.701, 11.5649]],
    500: [[8, 2.034, 0.0384], [16, 4.068, 0.0768], [32, 8.136, 0.1536], [64, 16.271, 0.3072], [128, 32.542, 0.6143], [192, 48.813, 0.9215], [256, 65.084, 1.2286], [384, 97.627, 1.8429], [512, 130.169, 2.4572], [768, 195.253, 3.5052], [1024, 260.338, 4.4548], [1536, 390.506, 5.9396], [2048, 520.675, 7.1953], [3072, 781.013, 8.4895], [4096, 1041.351, 9.3439], [6144, 1562.026, 10.5865], [8192, 2082.701, 11.4672]],
    10000: [[8, 2.034, 0.0537], [16, 4.068, 0.1073], [32, 8.136, 0.2146], [64, 16.271, 0.4293], [128, 32.542, 0.8586], [192, 48.813, 1.2878], [256, 65.084, 1.6791], [384, 97.627, 2.3016], [512, 130.169, 2.7628], [768, 195.253, 3.257], [1024, 260.338, 3.5059], [1536, 390.506, 3.6003], [2048, 520.675, 3.6473], [3072, 781.013, 3.6473], [4096, 1041.351, 3.6473], [6144, 1562.026, 3.6473], [8192, 2082.701, 3.6473]],
    20000: [[8, 2.034, 0.0689], [16, 4.068, 0.1378], [32, 8.136, 0.2756], [64, 16.271, 0.5512], [128, 32.542, 1.0779], [192, 48.813, 1.4775], [256, 65.084, 1.7736], [384, 97.627, 2.1299], [512, 130.169, 2.2939], [768, 195.253, 2.3996], [1024, 260.338, 2.4378], [1536, 390.506, 2.4378], [2048, 520.675, 2.4378], [3072, 781.013, 2.4378], [4096, 1041.351, 2.4378], [6144, 1562.026, 2.4378], [8192, 2082.701, 2.4378]],
    30000: [[8, 2.034, 0.0838], [16, 4.068, 0.1675], [32, 8.136, 0.335], [64, 16.271, 0.6701], [128, 32.542, 1.1975], [192, 48.813, 1.519], [256, 65.084, 1.7262], [384, 97.627, 1.8837], [512, 130.169, 1.9449], [768, 195.253, 1.9758], [1024, 260.338, 1.9758], [1536, 390.506, 1.9758], [2048, 520.675, 1.9758], [3072, 781.013, 1.9758], [4096, 1041.351, 1.9758], [6144, 1562.026, 1.9758], [8192, 2082.701, 1.9758]],
}

ALPHA = [100, 500, 10000, 20000, 30000]
LBL = {100: "100 µs", 500: "500 µs", 10000: "10 ms", 20000: "20 ms", 30000: "30 ms"}
CMAP = {100: "#1F4E79", 500: "#2E86AB", 10000: "#7FB069", 20000: "#E0A458", 30000: "#C1666B"}
CAP = {"L40S": 192, "B200": 2048}
MARK = {"L40S": "o", "B200": "s"}


def knee(pts):
    full = pts[-1][2]
    return next(p for p in pts if p[2] >= 0.98 * full)


def main(outpath="dse.pdf"):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    for mach, D in (("L40S", L40S), ("B200", B200)):
        cap = CAP[mach]
        for a in ALPHA:
            pts = D[a]
            sol = [p for p in pts if p[0] <= cap]
            das = [p for p in pts if p[0] >= cap]
            ax.plot([p[1] for p in sol], [p[2] for p in sol], "-",
                    color=CMAP[a], lw=1.5, marker=MARK[mach], ms=3.2,
                    mec="none", zorder=4)
            ax.plot([p[1] for p in das], [p[2] for p in das], "--",
                    color=CMAP[a], lw=1.1, marker=MARK[mach], ms=2.6,
                    mfc="white", mec=CMAP[a], mew=0.7, alpha=0.85, zorder=3)
            k = knee(pts)
            ax.plot(k[1], k[2], "*", color=CMAP[a], ms=12,
                    mec="black", mew=0.5, zorder=6)
            ax.annotate(f"{k[0]}", (k[1], k[2]), fontsize=6.4,
                        xytext=(4, -9), textcoords="offset points",
                        color=CMAP[a], fontweight="bold", zorder=7)
        # the card's ceiling
        ce = [p for p in D[ALPHA[0]] if p[0] == cap][0][1]
        ax.axvline(ce, color="0.45", ls=(0, (2, 2)), lw=1.0, zorder=1)
        ax.text(ce, ax.get_ylim()[1], f" {mach} ceiling\n {cap} SPU",
                fontsize=6.6, color="0.35", va="top", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("PIM logic area (mm$^2$)   —   log scale", fontsize=9)
    ax.set_ylabel("end-to-end speedup", fontsize=9)
    ax.grid(lw=0.4, color="0.89", which="both", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)
    sec = ax.secondary_xaxis(
        "top", functions=(lambda v: v / 2.0339 * 8, lambda s: s / 8 * 2.0339))
    sec.set_xlabel("SPU count", fontsize=8, labelpad=2)
    sec.tick_params(labelsize=7)

    h = [plt.Line2D([], [], color=CMAP[a], lw=1.6, label=LBL[a]) for a in ALPHA]
    h += [plt.Line2D([], [], color="0.3", lw=1.5, marker="o", ms=3.4, mec="none",
                     label="L40S (circles)"),
          plt.Line2D([], [], color="0.3", lw=1.5, marker="s", ms=3.4, mec="none",
                     label="B200 (squares)"),
          plt.Line2D([], [], color="0.3", lw=1.1, ls="--", label="beyond the ceiling")]
    ax.legend(handles=h, fontsize=7, ncol=2, loc="upper left",
              frameon=False, labelspacing=0.3, columnspacing=1.2)

    fig.text(0.5, -0.02,
             "★ = knee, the smallest budget within 2% of the 8192-SPU limit, "
             "labelled with its SPU count. Every $\\alpha$ we care about places its "
             "knee\nbeyond what L40S can build (192 SPU): at 100 µs it wants 3072, "
             "16× the card. B200 reaches its knee within the card for "
             "$\\alpha\\geq$ 10 ms\nbut not below. Area is PIM logic only, "
             "not the DRAM it sits in.",
             ha="center", va="top", fontsize=6.6)

    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dse.pdf")
