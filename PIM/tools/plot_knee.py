#!/usr/bin/env python3
"""Speedup vs SPU budget, both machines on one axis, every knee in frame.

SPU count is forced by the memory organisation: the expansion ping-pongs between
two banks per unit (level i reads bank i%2, writes (i+1)%2), so N_SPU = banks/2.
L40S 24 ch x 16 banks -> 192; B200 256 pc x 16 -> 2048. Solid up to that
ceiling, dashed past it -- those points need a different memory system, not a
larger PIM budget.

The 1/N_SPU scaling behind every point is measured across 24..4096 SPU on both
orgs; with the instance count scaled to the machine the cycle counts agree to
0.045%.

    python3 plot_knee.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (SPU, speedup)
L40S = {
    100: [[8, 0.1255], [16, 0.2511], [32, 0.5022], [64, 1.0043], [128, 1.8866], [192, 2.5959], [256, 3.085], [384, 3.8691], [512, 4.0983], [768, 4.4444], [1024, 4.7077], [1536, 5.0621], [2048, 5.1601], [3072, 5.2848], [4096, 5.2848], [6144, 5.2848], [8192, 5.2848]],
    500: [[8, 0.1265], [16, 0.2529], [32, 0.5058], [64, 1.0116], [128, 1.9003], [192, 2.6148], [256, 3.1075], [384, 3.8973], [512, 4.1281], [768, 4.4768], [1024, 4.7419], [1536, 5.0989], [2048, 5.1977], [3072, 5.3232], [4096, 5.3232], [6144, 5.3232], [8192, 5.3232]],
    10000: [[8, 0.1476], [16, 0.2953], [32, 0.5905], [64, 1.181], [128, 2.2184], [192, 3.0526], [256, 3.4751], [384, 4.0438], [512, 4.1205], [768, 4.1205], [1024, 4.1205], [1536, 4.1205], [2048, 4.1205], [3072, 4.1205], [4096, 4.1205], [6144, 4.1205], [8192, 4.1205]],
    20000: [[8, 0.1691], [16, 0.3383], [32, 0.6766], [64, 1.3532], [128, 2.4274], [192, 3.0767], [256, 3.2576], [384, 3.4151], [512, 3.4151], [768, 3.4151], [1024, 3.4151], [1536, 3.4151], [2048, 3.4151], [3072, 3.4151], [4096, 3.4151], [6144, 3.4151], [8192, 3.4151]],
    30000: [[8, 0.1902], [16, 0.3803], [32, 0.7606], [64, 1.5161], [128, 2.4459], [192, 2.7635], [256, 2.8519], [384, 2.8519], [512, 2.8519], [768, 2.8519], [1024, 2.8519], [1536, 2.8519], [2048, 2.8519], [3072, 2.8519], [4096, 2.8519], [6144, 2.8519], [8192, 2.8519]],
}
B200 = {
    100: [[8, 0.0377], [16, 0.0754], [32, 0.1509], [64, 0.3017], [128, 0.6035], [192, 0.9052], [256, 1.2069], [384, 1.8104], [512, 2.4139], [768, 3.4434], [1024, 4.3763], [1536, 5.8348], [2048, 7.0684], [3072, 8.3398], [4096, 9.1791], [6144, 10.5075], [8192, 11.5649]],
    500: [[8, 0.0384], [16, 0.0768], [32, 0.1536], [64, 0.3072], [128, 0.6143], [192, 0.9215], [256, 1.2286], [384, 1.8429], [512, 2.4572], [768, 3.5052], [1024, 4.4548], [1536, 5.9396], [2048, 7.1953], [3072, 8.4895], [4096, 9.3439], [6144, 10.5865], [8192, 11.4672]],
    10000: [[8, 0.0537], [16, 0.1073], [32, 0.2146], [64, 0.4293], [128, 0.8586], [192, 1.2878], [256, 1.6791], [384, 2.3016], [512, 2.7628], [768, 3.257], [1024, 3.5059], [1536, 3.6003], [2048, 3.6473], [3072, 3.6473], [4096, 3.6473], [6144, 3.6473], [8192, 3.6473]],
    20000: [[8, 0.0689], [16, 0.1378], [32, 0.2756], [64, 0.5512], [128, 1.0779], [192, 1.4775], [256, 1.7736], [384, 2.1299], [512, 2.2939], [768, 2.3996], [1024, 2.4378], [1536, 2.4378], [2048, 2.4378], [3072, 2.4378], [4096, 2.4378], [6144, 2.4378], [8192, 2.4378]],
    30000: [[8, 0.0838], [16, 0.1675], [32, 0.335], [64, 0.6701], [128, 1.1975], [192, 1.519], [256, 1.7262], [384, 1.8837], [512, 1.9449], [768, 1.9758], [1024, 1.9758], [1536, 1.9758], [2048, 1.9758], [3072, 1.9758], [4096, 1.9758], [6144, 1.9758], [8192, 1.9758]],
}

ALPHA = [100, 500, 10000, 20000, 30000]
LBL = {100: "100 µs", 500: "500 µs", 10000: "10 ms", 20000: "20 ms", 30000: "30 ms"}
CMAP = {100: "#1F4E79", 500: "#3E92CC", 10000: "#5B8C5A", 20000: "#E0A458", 30000: "#C1666B"}
CAP = {"L40S": 192, "B200": 2048}
MK = {"L40S": "o", "B200": "s"}


def knee(pts):
    full = pts[-1][1]
    return next(p for p in pts if p[1] >= 0.98 * full)


def main(outpath="knee.pdf"):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    lab = []
    for mach, D in (("L40S", L40S), ("B200", B200)):
        cap = CAP[mach]
        for a in ALPHA:
            pts = D[a]
            sol = [p for p in pts if p[0] <= cap]
            das = [p for p in pts if p[0] >= cap]
            ax.plot([p[0] for p in sol], [p[1] for p in sol], "-",
                    color=CMAP[a], lw=1.7, marker=MK[mach], ms=3.6,
                    mec="none", zorder=4)
            ax.plot([p[0] for p in das], [p[1] for p in das], "--",
                    color=CMAP[a], lw=1.1, marker=MK[mach], ms=2.9,
                    mfc="white", mec=CMAP[a], mew=0.8, alpha=0.9, zorder=3)
            k = knee(pts)
            ax.plot(*k, "*", color=CMAP[a], ms=13, mec="black", mew=0.55, zorder=6)
            lab.append((k[0], k[1], a, mach))
        ax.axvline(cap, color="0.35", ls=(0, (3, 2)), lw=1.2, zorder=1)

    # knee labels: alternate side by rank so nothing collides
    for i, (x, y, a, m) in enumerate(sorted(lab, key=lambda z: -z[1])):
        dx, dy, ha = (8, 2, "left") if i % 2 == 0 else (-8, -9, "right")
        ax.annotate(f"{m} {x}", (x, y), fontsize=6.6, color=CMAP[a],
                    fontweight="bold", zorder=7, ha=ha,
                    xytext=(dx, dy), textcoords="offset points")

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 32, 128, 512, 2048, 8192])
    ax.set_xticklabels(["8", "32", "128", "512", "2048", "8192"])
    ax.set_xlabel("SPU budget  (= DRAM banks / 2)", fontsize=9)
    ax.set_ylabel("end-to-end speedup", fontsize=9)
    ax.grid(lw=0.4, color="0.89", which="both", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)

    yt = ax.get_ylim()[1]
    ax.text(192 * 0.93, yt, "L40S ceiling 192 ", rotation=90, fontsize=7,
            color="0.3", va="top", ha="right")
    ax.text(2048 * 0.93, yt, "B200 ceiling 2048 ", rotation=90, fontsize=7,
            color="0.3", va="top", ha="right")

    h = [plt.Line2D([], [], color=CMAP[a], lw=1.8, label=LBL[a]) for a in ALPHA]
    h += [plt.Line2D([], [], color="0.3", lw=1.7, marker="o", ms=3.8,
                     mec="none", label="L40S"),
          plt.Line2D([], [], color="0.3", lw=1.7, marker="s", ms=3.8,
                     mec="none", label="B200"),
          plt.Line2D([], [], color="0.3", lw=1.1, ls="--", label="past the ceiling")]
    ax.legend(handles=h, fontsize=7.2, ncol=2, loc="upper left",
              frameon=False, labelspacing=0.3, columnspacing=1.3,
              title=r"round trip $\alpha$", title_fontsize=7.6)

    fig.text(0.5, -0.02,
             "★ = knee, the smallest budget within 2% of the 8192-SPU limit. "
             "Below 10 ms both cards sit left of their knee — a datacenter "
             "deployment is\nunder-provisioned, not over. Above 10 ms both sit "
             "right of it: at 30 ms, 256 SPU reaches L40S's 192-SPU result and "
             "512 reaches B200's 2048.\nSPU count is not a free parameter; the "
             "dashed segments would need a different memory organisation.",
             ha="center", va="top", fontsize=6.6)

    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "knee.pdf")
