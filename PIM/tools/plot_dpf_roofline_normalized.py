#!/usr/bin/env python3
"""Normalized roofline (classic shape, ONE roofline for both devices).

Axes are normalized to each device's own hardware:
    x' = I / ridge(dev)      (intensity in units of the machine balance)
    y' = P / peak(dev)       (performance in units of the compute peak)
Every machine's roofline then collapses onto the SAME curve
    y' = min(x', 1)
so the figure has ONE roofline and the device gap on the y-axis disappears
by construction: a kernel that fully uses its machine sits ON the line,
anything below the line is hardware the kernel cannot use.

MEASURED (kernel-only nsys, matched shape n18_B256 = 67.1M leaves):
  L40S: expand 9.240 ms, convert 3.091, full 12.331
  B200: expand 2.526 ms, convert 2.673, full 5.200
Machines: L40S 864 GB/s, 89.5 G ChaCha/s (ridge 0.1035 ChaCha/B)
          B200 7.7 TB/s, 145.4 G ChaCha/s (ridge 0.0189 ChaCha/B)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = {"L40S": (864e9, 142 * 64 * 2.52e9 / 256, "#1a56db"),
       "B200": (7.7e12, 37.2e12 / 256, "#c2410c")}
FAM = {"Tree Expand": (1 / 112.0, "^"), "Full DPF": (3 / 152.0, "o"),
       "Leaf Convert": (2 / 40.0, "s")}
LEAVES = 67.108864e6
MS = {"L40S": dict(expand=9.240, conv=3.091),
      "B200": dict(expand=2.526, conv=2.673)}
SHARE = {("L40S", "Tree Expand"): 75, ("L40S", "Leaf Convert"): 25,
         ("B200", "Tree Expand"): 49, ("B200", "Leaf Convert"): 51}

def gblocks(dev, fam):
    t = MS[dev]
    if fam == "Tree Expand":  return LEAVES / t["expand"] / 1e6
    if fam == "Leaf Convert": return 2 * LEAVES / t["conv"] / 1e6
    return 3 * LEAVES / (t["expand"] + t["conv"]) / 1e6

fig, ax = plt.subplots(figsize=(7.2, 5.0))
xs = (3e-2, 30)
# the single normalized roofline: y = min(x, 1)
ax.plot([xs[0], 1], [xs[0], 1], "k-", lw=2.6)
ax.plot([1, xs[1]], [1, 1], "k-", lw=2.6)
ax.axvline(1, color="k", ls=":", lw=1.0, alpha=0.7)

for dev, (bw, pk, c) in DEV.items():
    ridge = pk / bw
    for fam, (I, m) in FAM.items():
        xn = I / ridge
        yn = gblocks(dev, fam) * 1e9 / pk
        ax.plot(xn, yn, m, ms=11, color=c, mec="k", mew=0.8, zorder=5)
        lab = SHARE.get((dev, fam))
        txt = f"{lab}% of DPF" if lab is not None else None
        if txt:
            ax.annotate(txt, xy=(xn, yn), xytext=(0, -15), ha="center",
                        textcoords="offset points", fontsize=8, color=c,
                        weight="bold")

for dev, (_, _, c) in DEV.items():
    ax.plot([], [], "o", color=c, mec="k", label=f"{dev} (measured)")
for fam, (_, m) in FAM.items():
    ax.plot([], [], m, color="w", mec="k", label=fam)
ax.plot([], [], "k-", lw=2.6, label="normalized roofline  y = min(x, 1)")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*xs); ax.set_ylim(6e-2, 2)
ax.set_xlabel("Operation Intensity / machine balance  (I / ridge)")
ax.set_ylabel("Attained / compute peak")
ax.legend(loc="lower right", fontsize=8.2, framealpha=0.95)
fig.tight_layout()

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "docs", "figures")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, "dpf_roofline_normalized." + ext), dpi=220)
print("wrote", os.path.join(outdir, "dpf_roofline_normalized.{pdf,png}"))
for dev, (bw, pk, _c) in DEV.items():
    ridge = pk / bw
    for fam, (I, _m) in FAM.items():
        xn, yn = I / ridge, gblocks(dev, fam) * 1e9 / pk
        roof = min(xn, 1)
        print(f"{dev} {fam:13s} x'={xn:5.2f}  y'={yn:5.3f}  ({100*yn/roof:5.1f}% of roof)")
