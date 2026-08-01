#!/usr/bin/env python3
"""Per-stage roofline for the FULL DPF pipeline on L40S (one figure per size).

Stages (production kernels, Beaver output layer):
  hash    = hash_kernel_chacha8   : ChaCha8 PRG, parent seed -> 2 children
  expand  = expand_kernel         : EvalAll CW apply (lsb-predicated XOR)
  sums    = out_sum_kernel        : per-leaf ChaCha8 H' + mod-P convert + sums
  scatter = out_scatter_kernel    : H' + y=C+tau?CW + negacyclic fold (atomic)

DATA PROVENANCE (kernel-only, nsys cuda_gpu_kern_sum, this host, 2026-07-25):
  bench: endtoend/bin/dpf_real_bench_v2 --n {18,20} --B 256 --iters 2
         (built from current dpf_gpu.cu + leaf_convert_cuda.cu)
  nsys totals cover 3 runs (1 warm + 2 iters) -> divided by 3.
  n=18 B=256:  67,108,864 leaves, checksum 27cb4298e84744ed, 0.2022 ns/leaf
  n=20 B=256: 268,435,456 leaves, checksum 204a14f81d54aa7e, 0.2080 ns/leaf
Roofs: L40S DRAM 864 GB/s; INT32 peak 22.9 Tops (142 SM x 64 x 2.52 GHz).
Intensity: int32 ops / ALGORITHMIC-MINIMUM DRAM bytes per leaf (per stage).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# nsys raw kernel totals in ns over 3 runs, keyed by dataset ------- measured --
DATA = {
    "n18": dict(leaves=67108864, agg_ns=0.2022,
                raw={"hash": 9166586, "expand": 18504914,
                     "sums": 4632317, "scatter": 4651678}),
    "n20": dict(leaves=268435456, agg_ns=0.2080,
                raw={"hash": 37532232, "expand": 74347878,
                     "sums": 17518867, "scatter": 27387083}),
}
RUNS = 3
# per-leaf int32 ops / algorithmic-minimum DRAM bytes -------------- derived --
#  hash   : ChaCha core+ff 272 op; read parent 16B + write children 32B
#  expand : predicated CW xor ~20 op; read children 32B + write corrected 32B
#  sums   : H' 272 + mod-P reduce ~10; read leaf 16B (sums stay on-chip)
#  scatter: H' 272 + y/fold ~15; read leaf 16B + g update 8B
OPS = {"hash": 272, "expand": 20, "sums": 282, "scatter": 287}
BYT = {"hash": 48, "expand": 64, "sums": 16, "scatter": 24}

BW, PEAK = 864e9, 142 * 64 * 2.52e9
RIDGE = PEAK / BW

STYLE = {"hash":    ("o", "#1a56db", "ChaCha PRG (hash)"),
         "expand":  ("s", "#b91c1c", "EvalAll CW-apply (expand)"),
         "sums":    ("^", "#1f9d55", "H′ + convert (sums)"),
         "scatter": ("D", "#c2410c", "scatter g (fold)")}

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
outdir = os.path.join(repo, "docs", "figures")
os.makedirs(outdir, exist_ok=True)

for tag, ds in DATA.items():
    L = ds["leaves"]
    t_ms = {k: v / RUNS / 1e6 for k, v in ds["raw"].items()}
    tot = sum(t_ms.values())

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs = (3e-2, 60)
    ax.plot([xs[0], RIDGE], [BW * xs[0] / 1e12, PEAK / 1e12], "k-", lw=2.2)
    ax.plot([RIDGE, xs[1]], [PEAK / 1e12, PEAK / 1e12], "k-", lw=2.2)
    ax.axvline(RIDGE, color="#888", ls="--", lw=1.2)
    ax.text(RIDGE * 1.08, 0.045, f"ridge {RIDGE:.1f} op/B", fontsize=8, color="#555")
    ax.text(1.1, PEAK / 1e12 * 1.15, "peak INT32 22.9 Tops", fontsize=9, weight="bold")
    ax.text(4.4e-2, 6.5e-2, "peak DRAM\n864 GB/s", fontsize=9, weight="bold", rotation=38)
    ax.text(0.25, 8, "Mem. Bound", fontsize=12, weight="bold", color="#b91c1c")
    ax.text(33, 1.1, "Comp.\nBound", fontsize=11, weight="bold", color="#555", ha="center")

    stats = {}
    for k in ("hash", "expand", "sums", "scatter"):
        ops_s = OPS[k] * L / (t_ms[k] * 1e-3)
        bw = BYT[k] * L / (t_ms[k] * 1e-3)
        stats[k] = (ops_s, bw)
        m, c, lab = STYLE[k]
        ax.plot(OPS[k] / BYT[k], ops_s / 1e12, m, ms=9, color=c, mec="k",
                mew=0.7, zorder=5,
                label=f"{lab}: {100*ops_s/PEAK:.0f}% peak, "
                      f"{100*bw/BW:.0f}% algBW")

    agg_ops = sum(OPS.values())
    agg_perf = agg_ops / (ds["agg_ns"] * 1e-9) / 1e12
    ax.plot(agg_ops / sum(BYT.values()), agg_perf, "*", ms=17, color="#f5b301",
            mec="k", mew=0.8, zorder=6,
            label=f"full DPF pipeline: {100*agg_perf*1e12/PEAK:.0f}% peak")

    ex_ops, ex_bw = stats["expand"]
    ax.annotate(
        f"dominant stage ({100*t_ms['expand']/tot:.0f}% of time):\n"
        f"{100*ex_bw/BW:.0f}% of peak DRAM, {100*ex_ops/PEAK:.0f}% of peak compute",
        xy=(OPS["expand"] / BYT["expand"], ex_ops / 1e12),
        xytext=(0.045, 1.6), fontsize=8.5, color="#b91c1c",
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.1))
    h_ops, _ = stats["hash"]
    ax.annotate("above the DRAM line = producer->consumer\nreuse caught by L2 "
                "(hash->expand handoff)",
                xy=(OPS["hash"] / BYT["hash"], h_ops / 1e12),
                xytext=(1.6, 22), fontsize=8, color="#1a56db",
                arrowprops=dict(arrowstyle="->", color="#1a56db", lw=1.0))
    ax.text(3.2e-2, 1.35e-2,
            "every stage lies LEFT of the ridge: intensities are fixed by the\n"
            "algorithm (5.7 / 0.3 / 17.6 / 12 op/B) -- no schedule reaches "
            "Comp. Bound", fontsize=8.5, color="#111")

    n = tag[1:]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xs); ax.set_ylim(1e-2, 60)
    ax.set_xlabel("operational intensity (int32 op / algorithmic-min DRAM byte)")
    ax.set_ylabel("performance (int32 Tops)")
    ax.set_title(f"L40S roofline, all DPF stages "
                 f"(nsys kernel-only, n={n} B=256, {L/1e6:.0f}M leaves)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"dpf_roofline_stages_{tag}.{ext}"), dpi=200)
    plt.close(fig)

    print(f"== {tag}  ({L/1e6:.0f}M leaves)")
    for k in ("hash", "expand", "sums", "scatter"):
        ops_s, bw = stats[k]
        print(f"  {k:8s} t={t_ms[k]:7.3f}ms ({100*t_ms[k]/tot:4.1f}%)  "
              f"I={OPS[k]/BYT[k]:5.2f}  P={ops_s/1e12:5.2f}Tops "
              f"({100*ops_s/PEAK:4.1f}%)  algBW={bw/1e9:6.1f}GB/s "
              f"({100*bw/BW:5.1f}%)")
print("wrote", os.path.join(outdir, "dpf_roofline_stages_{n18,n20}.{pdf,png}"))
