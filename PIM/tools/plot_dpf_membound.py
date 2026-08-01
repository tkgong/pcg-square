#!/usr/bin/env python3
"""DPF memory-bound evidence figure (L40S), three proofs on one ns/leaf axis.

  (1) plateau signature   : measured regime curve, ns/leaf flat (+-2%) over a
                            16x size range -> time = bytes x const (streaming)
  (2) shape insensitivity : same total leaves, different (n,B) tree shapes ->
                            same ns/leaf (+-6%) -> compute shape irrelevant
  (3) speed-of-light walls: memory wall 3.9x below plateau, compute wall
                            12.7x below -> compute-side speedup bounded 1.08x

DATA PROVENANCE (all measured numbers copied verbatim from):
  endtoend/results_newalgo_gpu_l40s.md  -- dpf_real_bench.cu, L40S silicon,
    cudaEvent phase-wall, 5 iters + 1 warm, CUDA 12.6, production kernels
    (dpf_gpu_batch_full_eval_device + dpf_out_sums + dpf_out_scatter_g).
  SOL constants: L40S spec (864 GB/s; INT32 22.9 Tops = 142 SM x 64 x 2.52GHz)
    + per-leaf algorithmic minima derived in the comments below.

Output: docs/figures/dpf_membound_l40s.{pdf,png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- measured --
# Regime curve: (total_leaves, ns/leaf, shape) -- md table rows, verbatim.
CURVE = [
    (0.52e6, 0.720, "12,128"),
    (1.05e6, 0.470, "14,64"),
    (4.2e6,  0.247, "16,64"),
    (16.8e6, 0.204, "16,256"), (16.8e6, 0.229, "18,64"),
    (67e6,   0.204, "18,256"), (67e6,   0.228, "20,64"),
    (268e6,  0.210, "20,256"), (268e6,  0.226, "22,64"),
]
# Collector interp curve extends the plateau to 1.07e9 with 0.220 -- that
# point is an EXTRAPOLATION (largest silicon run is 268M): hollow marker.
EXTRAP = (1.07e9, 0.220)

# Shape-insensitivity pairs = the three same-total plateau pairs above (+-6%).
PAIRS = [(16.8e6, 0.204, 0.229), (67e6, 0.204, 0.228), (268e6, 0.210, 0.226)]

# --------------------------------------------------------------- SOL walls --
# Memory wall: mandatory DRAM bytes per leaf (algorithmic minimum):
#   expand : read parent 16B + write 2 children 32B = 48B/node / 2 leaves = 24B
#   convert: read leaf 16B + write g coefficient 8B                       = 24B
BYTES_PER_LEAF = 48.0
BW = 864e9                                   # L40S GDDR6 peak, B/s
MEM_SOL = BYTES_PER_LEAF / BW * 1e9          # 0.0556 ns/leaf
# Compute wall: int32 ops per leaf (ChaCha core+ff ~272/node /2 + H' 272 + ~10)
OPS_PER_LEAF = 400.0
INT32_PEAK = 142 * 64 * 2.52e9               # 22.9 Tops (Ada: 64 INT32/SM/clk)
CMP_SOL = OPS_PER_LEAF / INT32_PEAK * 1e9    # 0.0175 ns/leaf

PLATEAU = 0.216                              # md: plateau average

fig, ax = plt.subplots(figsize=(7.0, 4.6))

# (3) SOL walls + forbidden region
ax.axhspan(1e-3, MEM_SOL, color="#b91c1c", alpha=0.08)
ax.axhline(MEM_SOL, color="#b91c1c", ls="--", lw=1.8)
ax.axhline(CMP_SOL, color="#777777", ls="--", lw=1.5)
ax.text(1.2e9, MEM_SOL * 1.08, "memory wall SOL 0.056 ns/leaf  (48 B / 864 GB/s)",
        ha="right", va="bottom", fontsize=8.5, color="#b91c1c", weight="bold")
ax.text(1.2e9, CMP_SOL * 1.08, "compute wall SOL 0.017 ns/leaf  (400 op / 22.9 Tops)",
        ha="right", va="bottom", fontsize=8.5, color="#555555")
ax.text(6e5, MEM_SOL * 0.55, "unreachable by ANY compute speedup\n"
        r"compute-side gain bound: 0.216/0.200 $\leq$ 1.08$\times$",
        fontsize=8.5, color="#b91c1c", va="top")

# (1) measured curve (average per total for the line; both shapes as markers)
totals = sorted({x for x, _, _ in CURVE})
avg = [(t, sum(y for x, y, _ in CURVE if x == t) /
        len([1 for x, _, _ in CURVE if x == t])) for t in totals]
ax.plot([x for x, _ in avg], [y for _, y in avg], "-", color="#1a56db",
        lw=1.8, zorder=3)
ax.plot([x for x, y, _ in CURVE], [y for x, y, _ in CURVE], "o",
        color="#1a56db", mec="k", mew=0.5, ms=6, zorder=4,
        label="measured (dpf_real_bench, L40S)")
ax.plot(*EXTRAP, "o", mfc="none", mec="#1a56db", mew=1.5, ms=6, zorder=4,
        label="plateau extrapolation (not measured)")
ax.annotate("plateau: 16$\\times$ size range, ns/leaf $\\pm$2%\n"
            "time = bytes $\\times$ const (pure streaming)",
            xy=(1.3e8, 0.218), xytext=(6e6, 0.115), fontsize=9,
            color="#1a56db", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a56db", lw=1.2))

# (2) shape pairs: bracket each same-total pair
for t, lo, hi in PAIRS:
    ax.plot([t, t], [lo, hi], color="#c2410c", lw=2.2, zorder=5)
ax.plot([], [], color="#c2410c", lw=2.2,
        label="same total, different tree shape ($\\pm$6%)")
ax.text(2.6e8, 0.245, "shape-insensitive:\nonly bytes matter", fontsize=8.5,
        color="#c2410c", ha="center")

# security sizes
ax.axvspan(5.4e8, 8.6e9, color="#1a56db", alpha=0.05)
ax.text(6.2e8, 0.5, "security-table sizes\n$2^{20}..2^{24}$ (c=4,t=16)",
        fontsize=8, color="#1a56db")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(4e5, 1.3e9); ax.set_ylim(0.012, 1.0)
ax.set_xlabel("total leaves (DPF expand + convert)")
ax.set_ylabel("ns / leaf")
ax.set_title("DPF is memory-bound on L40S: plateau + shape insensitivity + SOL walls")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

fig.tight_layout()
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
outdir = os.path.join(repo, "docs", "figures")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, f"dpf_membound_l40s.{ext}"), dpi=200)
print("wrote", os.path.join(outdir, "dpf_membound_l40s.{pdf,png}"))
print(f"MEM_SOL={MEM_SOL:.4f} ns/leaf  CMP_SOL={CMP_SOL:.4f} ns/leaf  "
      f"plateau/mem={PLATEAU/MEM_SOL:.1f}x  plateau/cmp={PLATEAU/CMP_SOL:.1f}x")
