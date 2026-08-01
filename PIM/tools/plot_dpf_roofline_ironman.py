#!/usr/bin/env python3
"""Ironman-Fig1(c)-style roofline from the L40S DPF profile: DPFs are
memory-bound.

Axes mirror Ironman (Giga AES/s vs AES/byte) with ChaCha as the primitive:
  y = Giga ChaCha blocks/s     x = ChaCha blocks per DRAM byte (algorithmic)

MEASURED inputs (this host, L40S):
  nsys kernel-only stage times, dpf_real_bench_v2 --B 256 (see
  plot_dpf_roofline.py provenance):
    n=18 (67.1M leaves): hash 3.056ms expand 6.168ms sums 1.544 scatter 1.551
    n=20 (268.4M):       hash 12.511  expand 24.783  sums 5.840  scatter 9.129
  full-pipeline plateau: 0.216-0.220 ns/leaf across 5 security sizes
  (endtoend/results_newalgo_gpu_l40s.md).
Roofs: DRAM 864 GB/s; ChaCha peak = INT32 22.9 Tops / 256 op = 89.5 G/s.
Blocks & bytes per unit:
  Tree Expand : 1 ChaCha block/node; 112 B/node (hash 48 + expand r/w 64)
  Leaf Convert: 2 ChaCha blocks/leaf (H' in sums + H' in scatter); 40 B/leaf
  Full DPF    : 3 blocks/leaf; 152 B/leaf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BW = 864e9
PEAK = 142 * 64 * 2.52e9 / 256          # 89.5 G ChaCha blocks/s
RIDGE = PEAK / BW                        # 0.1036 blocks/B

# (label, marker, intensity blocks/B, [(shade, Gblocks/s, size_label)])
EXPAND_I = 1 / 112.0
CONV_I = 2 / 40.0
FULL_I = 3 / 152.0
FAMS = [
    ("Tree Expand Ops.", "^", EXPAND_I, [
        (0.55, 67.1e6 / 9.224e-3 / 1e9, "n18"),        # 7.27
        (1.00, 268.4e6 / 37.294e-3 / 1e9, "n20"),      # 7.20
    ]),
    ("Leaf Convert Ops.", "s", CONV_I, [
        (0.55, 2 * 67.1e6 / 3.095e-3 / 1e9, "n18"),    # 43.4
        (1.00, 2 * 268.4e6 / 14.969e-3 / 1e9, "n20"),  # 35.9
    ]),
    # full pipeline at the 5 security sizes (plateau ns/leaf -> 3 blocks/leaf)
    ("Full DPF Ops.", "o", FULL_I, [
        (0.25, 3 / 0.216, "2^20"), (0.42, 3 / 0.217, "2^21"),
        (0.60, 3 / 0.218, "2^22"), (0.78, 3 / 0.219, "2^23"),
        (1.00, 3 / 0.220, "2^24"),
    ]),
]

fig, ax = plt.subplots(figsize=(6.4, 4.6))
xs = (1e-3, 8)
# roofline
ax.plot([xs[0], RIDGE], [BW * xs[0] / 1e9, PEAK / 1e9], "k-", lw=2.4)
ax.plot([RIDGE, xs[1]], [PEAK / 1e9, PEAK / 1e9], "k-", lw=2.4)
ax.axvline(RIDGE, color="k", ls="--", lw=1.2)
ax.text(2.2e-1, PEAK / 1e9 * 1.18, "Peak ChaCha Perf.", fontsize=11)
ax.text(1.35e-3, 3.2, "Peak Mem BD", fontsize=11, rotation=42)
ax.text(2.1e-3, 0.35, "Mem. Bound!", fontsize=13, weight="bold", color="#b91c1c")
ax.text(0.5, 6.0, "Comp. Bound!", fontsize=13, weight="bold", color="#b91c1c")

blues = plt.get_cmap("Blues")
for label, mark, inten, pts in FAMS:
    for i, (shade, perf, _sz) in enumerate(pts):
        ax.plot(inten, perf, mark, ms=10 if mark != "o" else 9,
                color=blues(0.35 + 0.55 * shade), mec="k", mew=0.8,
                label=label if i == len(pts) - 1 else None, zorder=5)

ax.annotate("94% of mem roof", xy=(EXPAND_I, 7.27), xytext=(2.6e-3, 12),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1.0))
ax.annotate("on the roof", xy=(CONV_I, 43.4), xytext=(0.11, 26),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1.0))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*xs); ax.set_ylim(0.15, 300)
ax.set_xlabel("Operation Intensity (ChaCha/byte)")
ax.set_ylabel("Performance (Giga ChaCha/s)")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(False)
fig.tight_layout()

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "docs", "figures")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, "dpf_roofline_ironman_style." + ext), dpi=220)
print("wrote", os.path.join(outdir, "dpf_roofline_ironman_style.{pdf,png}"))
for label, _m, inten, pts in FAMS:
    for _s, perf, sz in pts:
        roof = min(PEAK / 1e9, BW * inten / 1e9)
        print(f"{label:18s} {sz:5s} I={inten:.4f}  P={perf:6.2f} G/s "
              f"({100*perf/roof:5.1f}% of roof)")
