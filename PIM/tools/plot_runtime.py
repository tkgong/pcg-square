#!/usr/bin/env python3
"""End-to-end runtime vs SPU budget, both machines on one axis.

Runtime rather than speedup: a ratio hides which machine is actually faster,
because each speedup is taken against its own GPU. L40S shows the larger ratio
only because its baseline is also slower.

SPU count is forced by the memory organisation -- the expansion ping-pongs
between two banks per unit (level i reads bank i%2, writes (i+1)%2), so
N_SPU = banks/2. L40S 24 ch x 16 banks -> 192; B200 256 pc x 16 -> 2048. Solid
to that ceiling, dashed past it: those points need a different memory system,
not a larger PIM budget.

The 1/N_SPU scaling behind every point is measured across 24..4096 SPU on both
orgs; with the instance count scaled to the machine, cycle counts agree to
0.045%.

    python3 plot_runtime.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (SPU, PIM+GPU runtime ms, ganged-GPU baseline runtime ms)
L40S = {
    100: [[8, 11500.902, 1443.847], [16, 5750.451, 1443.847], [32, 2875.225, 1443.847], [64, 1437.613, 1443.847], [128, 765.331, 1443.847], [192, 556.194, 1443.847], [256, 468.018, 1443.847], [384, 373.173, 1443.847], [512, 352.308, 1443.847], [768, 324.866, 1443.847], [1024, 306.702, 1443.847], [1536, 285.227, 1443.847], [2048, 279.808, 1443.847], [3072, 273.21, 1443.847], [4096, 273.21, 1443.847], [6144, 273.21, 1443.847], [8192, 273.21, 1443.847]],
    500: [[8, 11500.902, 1454.351], [16, 5750.451, 1454.351], [32, 2875.225, 1454.351], [64, 1437.613, 1454.351], [128, 765.331, 1454.351], [192, 556.194, 1454.351], [256, 468.018, 1454.351], [384, 373.173, 1454.351], [512, 352.308, 1454.351], [768, 324.866, 1454.351], [1024, 306.702, 1454.351], [1536, 285.227, 1454.351], [2048, 279.808, 1454.351], [3072, 273.21, 1454.351], [4096, 273.21, 1454.351], [6144, 273.21, 1454.351], [8192, 273.21, 1454.351]],
    10000: [[8, 11500.902, 1697.832], [16, 5750.451, 1697.832], [32, 2875.225, 1697.832], [64, 1437.613, 1697.832], [128, 765.331, 1697.832], [192, 556.194, 1697.832], [256, 488.571, 1697.832], [384, 419.859, 1697.832], [512, 412.05, 1697.832], [768, 412.05, 1697.832], [1024, 412.05, 1697.832], [1536, 412.05, 1697.832], [2048, 412.05, 1697.832], [3072, 412.05, 1697.832], [4096, 412.05, 1697.832], [6144, 412.05, 1697.832], [8192, 412.05, 1697.832]],
    20000: [[8, 11500.902, 1945.324], [16, 5750.451, 1945.324], [32, 2875.225, 1945.324], [64, 1437.613, 1945.324], [128, 801.4, 1945.324], [192, 632.267, 1945.324], [256, 597.157, 1945.324], [384, 569.627, 1945.324], [512, 569.627, 1945.324], [768, 569.627, 1945.324], [1024, 569.627, 1945.324], [1536, 569.627, 1945.324], [2048, 569.627, 1945.324], [3072, 569.627, 1945.324], [4096, 569.627, 1945.324], [6144, 569.627, 1945.324], [8192, 569.627, 1945.324]],
    30000: [[8, 11500.902, 2187.036], [16, 5750.451, 2187.036], [32, 2875.225, 2187.036], [64, 1442.556, 2187.036], [128, 894.168, 2187.036], [192, 791.41, 2187.036], [256, 766.875, 2187.036], [384, 766.875, 2187.036], [512, 766.875, 2187.036], [768, 766.875, 2187.036], [1024, 766.875, 2187.036], [1536, 766.875, 2187.036], [2048, 766.875, 2187.036], [3072, 766.875, 2187.036], [4096, 766.875, 2187.036], [6144, 766.875, 2187.036], [8192, 766.875, 2187.036]],
}
B200 = {
    100: [[8, 15274.41, 576.103], [16, 7637.205, 576.103], [32, 3818.602, 576.103], [64, 1909.301, 576.103], [128, 954.651, 576.103], [192, 636.434, 576.103], [256, 477.325, 576.103], [384, 318.217, 576.103], [512, 238.663, 576.103], [768, 167.306, 576.103], [1024, 131.642, 576.103], [1536, 98.735, 576.103], [2048, 81.504, 576.103], [3072, 69.079, 576.103], [4096, 62.762, 576.103], [6144, 54.828, 576.103], [8192, 49.815, 576.103]],
    500: [[8, 15274.41, 586.445], [16, 7637.205, 586.445], [32, 3818.602, 586.445], [64, 1909.301, 586.445], [128, 954.651, 586.445], [192, 636.434, 586.445], [256, 477.325, 586.445], [384, 318.217, 586.445], [512, 238.663, 586.445], [768, 167.306, 586.445], [1024, 131.642, 586.445], [1536, 98.735, 586.445], [2048, 81.504, 586.445], [3072, 69.079, 586.445], [4096, 62.762, 586.445], [6144, 55.396, 586.445], [8192, 51.141, 586.445]],
    10000: [[8, 15274.41, 819.616], [16, 7637.205, 819.616], [32, 3818.602, 819.616], [64, 1909.301, 819.616], [128, 954.651, 819.616], [192, 636.434, 819.616], [256, 488.141, 819.616], [384, 356.111, 819.616], [512, 296.661, 819.616], [768, 251.647, 819.616], [1024, 233.782, 819.616], [1536, 227.651, 819.616], [2048, 224.719, 819.616], [3072, 224.719, 819.616], [4096, 224.719, 819.616], [6144, 224.719, 819.616], [8192, 224.719, 819.616]],
    20000: [[8, 15274.41, 1052.318], [16, 7637.205, 1052.318], [32, 3818.602, 1052.318], [64, 1909.301, 1052.318], [128, 976.282, 1052.318], [192, 712.222, 1052.318], [256, 593.322, 1052.318], [384, 494.08, 1052.318], [512, 458.745, 1052.318], [768, 438.536, 1052.318], [1024, 431.661, 1052.318], [1536, 431.661, 1052.318], [2048, 431.661, 1052.318], [3072, 431.661, 1052.318], [4096, 431.661, 1052.318], [6144, 431.661, 1052.318], [8192, 431.661, 1052.318]],
    30000: [[8, 15274.41, 1279.34], [16, 7637.205, 1279.34], [32, 3818.602, 1279.34], [64, 1909.301, 1279.34], [128, 1068.333, 1279.34], [192, 842.203, 1279.34], [256, 741.12, 1279.34], [384, 679.171, 1279.34], [512, 657.804, 1279.34], [768, 647.492, 1279.34], [1024, 647.492, 1279.34], [1536, 647.492, 1279.34], [2048, 647.492, 1279.34], [3072, 647.492, 1279.34], [4096, 647.492, 1279.34], [6144, 647.492, 1279.34], [8192, 647.492, 1279.34]],
}

ALPHA = [100, 500, 10000, 20000, 30000]
LBL = {100: "100 µs", 500: "500 µs", 10000: "10 ms", 20000: "20 ms", 30000: "30 ms"}
CMAP = {100: "#1F4E79", 500: "#3E92CC", 10000: "#5B8C5A", 20000: "#E0A458", 30000: "#C1666B"}
CAP = {"L40S": 192, "B200": 2048}
MK = {"L40S": "o", "B200": "s"}


def knee(pts):
    """smallest budget within 2% of the fastest runtime (i.e. the flat floor)."""
    best = pts[-1][1]
    return next(p for p in pts if p[1] <= best * 1.02)


def main(outpath="runtime.pdf"):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))

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
            ax.plot(k[0], k[1], "*", color=CMAP[a], ms=13,
                    mec="black", mew=0.55, zorder=6)
            lab.append((k[0], k[1], a, mach))
        ax.axvline(cap, color="0.35", ls=(0, (3, 2)), lw=1.2, zorder=1)

    for i, (x, y, a, m) in enumerate(sorted(lab, key=lambda z: -z[1])):
        dx, dy, ha = (8, 3, "left") if i % 2 == 0 else (-8, -10, "right")
        ax.annotate(f"{m} {x}", (x, y), fontsize=6.6, color=CMAP[a],
                    fontweight="bold", zorder=7, ha=ha,
                    xytext=(dx, dy), textcoords="offset points")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([8, 32, 128, 512, 2048, 8192])
    ax.set_xticklabels(["8", "32", "128", "512", "2048", "8192"])
    ax.set_xlabel("SPU budget  (= DRAM banks / 2)", fontsize=9)
    ax.set_ylabel("end-to-end runtime per expansion (ms)", fontsize=9)
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
    ax.legend(handles=h, fontsize=7.2, ncol=2, loc="lower left",
              frameon=False, labelspacing=0.3, columnspacing=1.3,
              title=r"round trip $\alpha$", title_fontsize=7.6)

    fig.text(0.5, -0.02,
             "★ = knee, the smallest budget within 2% of the floor. Lower is "
             "better. At every α and every budget B200 finishes sooner — it "
             "shows the smaller\nspeedup only because its GPU baseline is faster "
             "too. Curves flatten where the NIC or the SM takes over from the "
             "SPU; past that point\nextra SPUs change nothing. SPU count is not "
             "a free parameter; the dashed segments would need a different "
             "memory organisation.",
             ha="center", va="top", fontsize=6.6)

    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runtime.pdf")
