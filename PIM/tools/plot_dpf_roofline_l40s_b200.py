#!/usr/bin/env python3
"""DPF roofline, L40S vs B200, MEASURED per-kernel times on both devices.

Points = kernel-only nsys sums (GPU_baseline/run.sh -> roofline_points.csv):
  L40S : n18_B256 (67.1M lv) expand 9.240 ms / convert 3.091 / full 12.331
         n24_B16 (268.4M lv) expand 37.364 / convert 17.989 / full 55.352
  B200 : n18_B256            expand 2.526  / convert 2.673  / full 5.200
         n20_B256 (268.4M lv) expand 9.707 / convert 9.349  / full 19.056
  (expand = hash_kernel + expand_kernel; convert = out_sum + out_scatter)

Matched shape n18_B256 -> honest device speedups, annotated on the figure:
  expand 3.7x | convert 1.2x | full 2.4x  --  on 8.9x more bandwidth.

Roofs: L40S 864 GB/s + 89.5 G ChaCha/s (INT32 22.9 Tops / 256, known).
  B200 7.7 TB/s (measured stream) + 145.4 G/s (INT32 37.2 Tops, 128/SM/clk).
Intensity (x) is algorithm-fixed: expand 1/112, convert 2/40, full 3/152
ChaCha blocks per algorithmic-minimum DRAM byte -- machine-independent.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L40S_BW, L40S_PK = 864e9, 142 * 64 * 2.52e9 / 256
B200_BW = 7.7e12
B200_PK_LO, B200_PK_HI = 18.6e12 / 256, 37.2e12 / 256

EXPAND_I, CONV_I, FULL_I = 1 / 112.0, 2 / 40.0, 3 / 152.0

# L40S: full sweep of the DPF domain 2^18..2^24 at CONSTANT 268.4M leaves
# (B scaled inversely), so only the tree depth varies. kernel-only nsys.
L40S_SWEEP = {   # n -> (expand_ms, conv_ms)
    18: (37.553, 12.132), 19: (37.461, 15.009), 20: (37.332, 14.994),
    21: (37.255, 15.019), 22: (37.268, 15.085), 23: (37.216, 17.080),
    24: (37.360, 17.981),
}
# B200: same sweep pending (GPU_baseline/run.sh); the two shapes measured so
# far are kept until it lands.
B200_SWEEP = {18: (2.526, 2.673), 20: (9.707, 9.349)}
LEAVES_SWEEP = 268.435456e6
LEAVES_B200 = {18: 67.108864e6, 20: 268.435456e6}

def pts(d):
    ex, cv = d["expand"], d["conv"]
    return {"expand": d["leaves"] / ex / 1e6, "conv": 2 * d["leaves"] / cv / 1e6,
            "full": 3 * d["leaves"] / (ex + cv) / 1e6}

fig, ax = plt.subplots(figsize=(7.4, 5.2))
xs = (1e-3, 8)

# L40S roofline
r1 = L40S_PK / L40S_BW
ax.plot([xs[0], r1], [L40S_BW * xs[0] / 1e9, L40S_PK / 1e9], "k-", lw=2.4)
ax.plot([r1, xs[1]], [L40S_PK / 1e9] * 2, "k-", lw=2.4)
ax.axvline(r1, color="k", ls=":", lw=1.0, alpha=0.7)

# B200 roofline: memory roof + compute roof (128 INT32/SM/clk)
rhi = B200_PK_HI / B200_BW
ax.plot([xs[0], rhi], [B200_BW * xs[0] / 1e9, B200_PK_HI / 1e9],
        color="#7a7a7a", ls="-", lw=2.2)
ax.plot([rhi, xs[1]], [B200_PK_HI / 1e9] * 2, color="#7a7a7a", ls="-", lw=2.2)
ax.axvline(rhi, color="#7a7a7a", ls=":", lw=1.0, alpha=0.6)


STYLE = {"expand": ("^", EXPAND_I), "conv": ("s", CONV_I), "full": ("o", FULL_I)}
COL = {"L40S": "#1a56db", "B200": "#c2410c"}

def series(dev):
    """[(n, {family: G blocks/s})] darkest = largest domain."""
    out = []
    src = L40S_SWEEP if dev == "L40S" else B200_SWEEP
    for n, (ex, cv) in sorted(src.items()):
        L = LEAVES_SWEEP if dev == "L40S" else LEAVES_B200[n]
        out.append((n, {"expand": L / ex / 1e6, "conv": 2 * L / cv / 1e6,
                        "full": 3 * L / (ex + cv) / 1e6}))
    return out

import matplotlib.colors as mcolors
def shade(base, f):
    r, g, b = mcolors.to_rgb(base)
    w = 0.72 - 0.62 * f                       # light -> dark with size
    return (r + (1 - r) * w, g + (1 - g) * w, b + (1 - b) * w)

for dev in ("L40S", "B200"):
    ser = series(dev)
    for i, (n, p) in enumerate(ser):
        f = i / max(1, len(ser) - 1)
        for st, (m, I) in STYLE.items():
            ax.plot(I, p[st], m, ms=9, color=shade(COL[dev], f), mec="k",
                    mew=0.6, zorder=5)

ax.plot([], [], "k-", lw=2.4, label="L40S roofline")
ax.plot([], [], "-", color="#7a7a7a", lw=2.2, label="B200 roofline")
for _dev, _c in COL.items():
    ax.plot([], [], "o", color=_c, mec="k",
            label=f"{_dev}  (light\u2192dark: $2^{{18}}$\u2192$2^{{24}}$)")
for _m, _lab in (("^", "Tree Expand"), ("s", "Leaf Convert"), ("o", "Full DPF")):
    ax.plot([], [], _m, color="w", mec="k", label=_lab)

A = series("L40S")[-1][1]; B = series("B200")[-1][1]
# annotate each kernel family with its SHARE of full-DPF time (matched shape)
def share(ex, cv):
    return 100 * ex / (ex + cv), 100 * cv / (ex + cv)
exL, cvL = share(*L40S_SWEEP[24])
exB, cvB = share(*B200_SWEEP[20])
ax.annotate(f"Tree Expand\nL40S {exL:.0f}% of DPF", xy=(EXPAND_I, A["expand"]),
            xytext=(1.5e-3, 1.6), fontsize=8, color=COL["L40S"], weight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.9, color=COL["L40S"]))
ax.annotate(f"Tree Expand\nB200 {exB:.0f}% of DPF", xy=(EXPAND_I, B["expand"]),
            xytext=(1.6e-3, 60), fontsize=8, color=COL["B200"], weight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.9, color=COL["B200"]))
ax.annotate(f"Leaf Convert\nL40S {cvL:.0f}% of DPF", xy=(CONV_I, A["conv"]),
            xytext=(0.14, 9), fontsize=8, color=COL["L40S"], weight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.9, color=COL["L40S"]))
ax.annotate(f"Leaf Convert\nB200 {cvB:.0f}% of DPF", xy=(CONV_I, B["conv"]),
            xytext=(0.13, 200), fontsize=8, color=COL["B200"], weight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.9, color=COL["B200"]))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*xs); ax.set_ylim(0.1, 900)
ax.set_xlabel("Operation Intensity (ChaCha/byte)  [algorithm-fixed]")
ax.set_ylabel("Performance (Giga ChaCha/s)")
ax.legend(loc="lower right", fontsize=7.2, framealpha=0.95, ncol=1)
fig.tight_layout()

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "docs", "figures")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, "dpf_roofline_l40s_b200." + ext), dpi=220)
print("wrote", os.path.join(outdir, "dpf_roofline_l40s_b200.{pdf,png}"))
for dev in ("L40S", "B200"):
    for n, p in series(dev):
        print(f"{dev:5s} n={n:<3d} expand {p['expand']:6.2f}  conv {p['conv']:6.2f}  "
              f"full {p['full']:6.2f} Gblocks/s")
