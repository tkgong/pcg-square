#!/usr/bin/env python3
"""Execution-time breakdown of the CPU PCG baseline (Ironman Fig 1a style).

Stacked bars, one per security config, showing where wall time goes.
MEASURED phase zones (PCG_PROF_REPORT, build-prof, localhost 2PC, party 0):
  DPF_GEN            fused Gen + full-domain ChaCha expansion (the tree)
  DPF_EXPAND_LOCAL   output layer: H' hash + mod-p convert + scatter + poly
  PRE_KS/GILBOA/DLTSFT  preprocessing (Kogge-Stone + Gilboa OT + delta-shift)
  NET_WAIT + traced   all network exchange windows
FerretCOT setup is amortized out (TripleGen built once, reused across iters).
Source logs: comm_calib/*.log.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, total_s, gen, output, preproc(KS+Gil+Dlt), net)
CFG = [
    ("2^20\nc2 t64", 89.03, 45.21, 34.45, 0.72 + 0.50 + 7.48, 0.10),
    ("2^20\nc4 t16", 91.29, 44.42, 44.02, 0.43 + 0.13 + 1.53, 0.20),
    ("2^20\nc8 t8",  203.89, 87.90, 112.12, 1.03 + 0.45 + 1.01, 0.27),
    ("2^22\nc2 t64", 335.29, 181.20, 142.65, 0.94 + 1.16 + 7.52, 0.01),
    ("2^22\nc4 t16", 368.90, 178.42, 185.50, 0.83 + 0.13 + 1.75, 0.20),
    ("2^22\nc8 t8",  853.70, 354.35, 492.19, 0.51 + 0.13 + 1.76, 0.92),
]
CATS = ["DPF Tree Expand (ChaCha)", "Leaf Convert + Poly Output",
        "Preprocessing (OT)", "Network"]
COLORS = ["#3b5da8", "#7aa0d6", "#bcd0ec", "#9aa0a6"]

fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = range(len(CFG))
bot = [0.0] * len(CFG)
for ci, cat in enumerate(CATS):
    vals = []
    for _lab, tot, g, o, p, n in CFG:
        seg = [g, o, p, n][ci]
        vals.append(100.0 * seg / tot)
    ax.bar(x, vals, bottom=bot, width=0.66, color=COLORS[ci],
           edgecolor="white", linewidth=0.7, label=cat)
    bot = [b + v for b, v in zip(bot, vals)]

ax.set_xticks(list(x))
ax.set_xticklabels([c[0] for c in CFG], fontsize=9)
ax.set_ylabel("Percentage of Time (%)")
ax.set_ylim(0, 100)
ax.set_title("CPU PCG execution-time breakdown (L40S host, localhost 2PC)")
ax.legend(ncol=2, fontsize=8.5, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), frameon=False)
# annotate absolute wall on top
for xi, (_lab, tot, *_r) in zip(x, CFG):
    ax.text(xi, 101.5, f"{tot:.0f}s", ha="center", fontsize=7.5, color="#555")

fig.tight_layout()
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "docs", "figures")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, "cpu_exec_breakdown." + ext), dpi=220)
print("wrote", os.path.join(outdir, "cpu_exec_breakdown.{pdf,png}"))
for lab, tot, g, o, p, n in CFG:
    print(f"{lab.replace(chr(10),' '):12s} tot={tot:6.1f}s  "
          f"gen={100*g/tot:4.1f}% out={100*o/tot:4.1f}% "
          f"pre={100*p/tot:4.1f}% net={100*n/tot:4.2f}%")
