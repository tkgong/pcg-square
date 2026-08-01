#!/usr/bin/env python3
"""Combined motivation figure: Ironman-style roofline with the execution-time
breakdown annotated onto each kernel family, and BOTH memory classes
(GDDR: L40S roofline, HBM: B200 roofline) on one chart.

Why one chart works: operation intensity (x) is an ALGORITHM property --
identical on every machine. Only the roofs move.

ALL POINTS ARE KERNEL-ONLY (nsys cuda_gpu_kern_sum), one consistent caliber.
dpf_real_bench_v2 --B 256, 3 runs (1 warm + 2 iters), totals/3:
  n=18 B=256 ( 67.1M leaves, tree domain 2^18): expand 9.240 convert 3.091 ms
  n=24 B=16  (268.4M leaves, tree domain 2^24): expand 37.364 convert 17.989 ms
  (produced by GPU_baseline/run.sh -> roofline_points.csv; same two scales
   are run on every device so the comparison is apples-to-apples)
Size effect (4x leaves, 64x tree depth range): tree expand is flat (7.26 ->
7.18 G/s, 94 -> 93% of roof) -- the streaming half does not care; leaf
convert drops 43.4 -> 29.8 G/s (100 -> 69% of roof) as the scatter target
outgrows L2. Points slide DOWN the memory roof with size, never toward the
compute-bound region.
Phase-wall (0.216-0.220 ns/leaf) is 6-18% above kernel-only; that gap is the
per-level CW D2H/H2D + launch overhead, reported separately, not plotted.
B200: no new-algorithm silicon -- hollow marker, est from the old-algorithm
device ratio; run GPU_baseline/run.sh on a B200 host to replace it.
Roofs: L40S 864 GB/s + 89.5 G ChaCha/s (INT32 22.9 Tops/256).
       B200 8 TB/s + ~66 G ChaCha/s (INT32 ~17 Tops, est).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L40S_BW, L40S_PK = 864e9, 142 * 64 * 2.52e9 / 256          # 89.5 G/s
B200_BW, B200_PK = 8e12, 17e12 / 256                        # 66.4 G/s (est)

EXPAND_I, CONV_I, FULL_I = 1 / 112.0, 2 / 40.0, 3 / 152.0

fig, ax = plt.subplots(figsize=(7.2, 5.0))
xs = (1e-3, 8)

def roof(ax, bw, pk, color, ls, lw, label):
    ridge = pk / bw
    ax.plot([xs[0], ridge], [bw * xs[0] / 1e9, pk / 1e9], color=color,
            ls=ls, lw=lw, label=label)
    ax.plot([ridge, xs[1]], [pk / 1e9, pk / 1e9], color=color, ls=ls, lw=lw)
    ax.axvline(ridge, color=color, ls=":", lw=1.0, alpha=0.6)
    return ridge

r_l40s = roof(ax, L40S_BW, L40S_PK, "k", "-", 2.4, "L40S roofline (GDDR6, measured)")
r_b200 = roof(ax, B200_BW, B200_PK, "#7a7a7a", "--", 2.0,
              "B200 roofline (HBM3e, est. peaks)")

ax.text(2.4e-1, L40S_PK / 1e9 * 0.72, "Peak ChaCha Perf. (L40S)", fontsize=9)
ax.text(1.35e-3, 3.0, "Peak Mem BD\n(L40S 864 GB/s)", fontsize=8.5, rotation=40)
ax.text(1.1e-3, 22, "B200 8 TB/s", fontsize=8.5, rotation=40, color="#7a7a7a")
ax.text(1.6e-3, 0.32, "Mem. Bound!", fontsize=13, weight="bold", color="#b91c1c")
ax.text(0.9, 3.2, "Comp. Bound!", fontsize=13, weight="bold", color="#b91c1c")

blues = plt.get_cmap("Blues")
# --- kernel-only measurements, light shade = 67M leaves, dark = 268M --------
# roofline_points.csv, L40S (GPU_baseline/run.sh, kernel-only)
L = {"n18": 67.108864e6, "n24": 268.435456e6}
T = {"n18": dict(expand=9.240, conv=3.091),
     "n24": dict(expand=37.364, conv=17.989)}
PTS = {}
for k, t in T.items():
    ex, cv = t["expand"], t["conv"]
    PTS[k] = dict(expand=L[k] / ex / 1e6, conv=2 * L[k] / cv / 1e6,
                  full=3 * L[k] / (ex + cv) / 1e6,
                  share_ex=100 * ex / (ex + cv), share_cv=100 * cv / (ex + cv))

for k, shade in (("n18", 0.45), ("n24", 1.0)):
    p = PTS[k]
    ax.plot(EXPAND_I, p["expand"], "^", ms=11, color=blues(0.35 + 0.55 * shade),
            mec="k", mew=0.8, zorder=5,
            label="Tree Expand Ops." if k == "n24" else None)
    ax.plot(CONV_I, p["conv"], "s", ms=10, color=blues(0.35 + 0.55 * shade),
            mec="k", mew=0.8, zorder=5,
            label="Leaf Convert Ops." if k == "n24" else None)
    ax.plot(FULL_I, p["full"], "o", ms=9, color=blues(0.35 + 0.55 * shade),
            mec="k", mew=0.8, zorder=5,
            label="Full DPF Ops." if k == "n24" else None)

ax.annotate(f"Tree Expand: {PTS['n24']['share_ex']:.0f}% of DPF time,\n"
            f"{100*PTS['n24']['expand']/(L40S_BW*EXPAND_I/1e9):.0f}% of mem roof",
            xy=(EXPAND_I, PTS["n24"]["expand"]), xytext=(1.6e-3, 13.5),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.9))
ax.annotate(f"Leaf Convert: {PTS['n24']['share_cv']:.0f}% of DPF time\n"
            f"(L2 spill at 2^24: 100%→{100*PTS['n24']['conv']/(L40S_BW*CONV_I/1e9):.0f}% of roof)",
            xy=(CONV_I, PTS["n24"]["conv"]), xytext=(0.075, 8.5), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.annotate("light→dark = 2^18→2^24 domain\n(67M→268M leaves): points slide\nDOWN the roof, "
            "never toward Comp. Bound",
            xy=(FULL_I, PTS["n24"]["full"]), xytext=(2.2e-3, 1.1), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.9))

# B200 full-DPF point: ESTIMATE, not new-algorithm silicon.
# Provenance: L40S measured x 0.49 device ratio; the 0.49 was measured on the
# OLD-algorithm DPF bench on B200 -- its transfer to the new algorithm is an
# assumption (flagged est everywhere in the campaign tables). Re-measure with
# GPU_baseline/run.sh on a B200 host to replace this hollow marker.
ax.plot(FULL_I, PTS["n24"]["full"] / 0.49, "D", ms=9, mfc="none", mec="#c2410c", mew=1.6,
                label="Full DPF (B200, est. from old-algo ratio)", zorder=6)
ax.annotate("B200 est: L40S x measured old-algo\ndevice ratio (no new-algo B200\n"
            "silicon yet; see run.sh to measure)",
            xy=(FULL_I, PTS["n24"]["full"] / 0.49), xytext=(0.055, 90), fontsize=8,
            color="#c2410c",
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#c2410c"))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*xs); ax.set_ylim(0.15, 700)
ax.set_xlabel("Operation Intensity (ChaCha/byte)  [algorithm-fixed, machine-independent]")
ax.set_ylabel("Performance (Giga ChaCha/s)")
ax.legend(loc="lower right", fontsize=7.6, framealpha=0.95)
fig.tight_layout()

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "docs", "figures")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, "dpf_roofline_breakdown_combined." + ext), dpi=220)
print("wrote", os.path.join(outdir, "dpf_roofline_breakdown_combined.{pdf,png}"))
print(f"ridge L40S={r_l40s:.4f} blocks/B  ridge B200={r_b200:.5f} blocks/B  "
      f"(DPF intensities: expand {EXPAND_I:.4f}, full {FULL_I:.4f}, conv {CONV_I:.3f})")
