#!/usr/bin/env python3
"""Fig. 9 (end-to-end runtime vs SPU budget under six network latencies) regenerated from the final accounting.
Per cell: SPU lane(N) = per-instance SPU time (co-sim window, H' on the SPU, LSU overlap) x ceil(I / N) x SPU slowdown;
NTT lane = merge (L40S) / fused four-step + DRU (B200) x NTT slowdown; NIC = (n + 2) alpha; runtime = max of the three;
y = geomean over the suite. Area = N x 0.253970 mm^2 (SPU, Table III) + one DRU (0.024924) per 8 SPUs. Ceilings 192 / 2048
(banks / 2). Star = knee = smallest budget within 2% of the 8192-SPU runtime (as in the submission).
Usage: plot_fig9_budget.py WIN22_DIR OUT_PNG OUT_TXT"""
import io, contextlib, json, os, sys
from math import ceil, exp, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_
W, OUT, TXT = sys.argv[1], sys.argv[2], sys.argv[3]
def gm(v): v = list(v); return exp(sum(map(log, v)) / len(v))
DES = {"L40S": ("l40s", "merge", 192), "B200": ("b200", "f4dru", 2048)}
NS = [8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
ALPHA = [(100, "100 µs"), (500, "500 µs"), (10000, "10 ms"), (20000, "20 ms"), (30000, "30 ms"), (60000, "60 ms")]
COL = {100: "#1f3a93", 500: "#6a1b9a", 10000: "#e6a700", 20000: "#2e8b57", 30000: "#e8641b", 60000: "#c62828"}
AREA = lambda n: n * 0.253970 + (n / 8) * 0.024924
R = json.load(open(os.path.join(W, "e2e_nom.json")))
curves, lines = {}, []
for mach, (org, des, ceil_n) in DES.items():
    LN = L_(mach); rows = [r for r in R if r["org"] == org and r["design"] == des and r["tier"] == "fast"]
    for a_us, lab in ALPHA:
        ys = []
        for N in NS:
            v = []
            for r in rows:
                k = (r["c"], r["t"], r["logN"]); I = k[0] ** 2 * k[1] ** 2; n = LN[k]["n"]
                t_inst = r["spu_ms"] / ceil(I / ceil_n)                       # window lane = per-instance time x ceil(I / physical SPUs)
                spu = t_inst * ceil(I / N) * r["spu_slow"]; ntt = r["ntt_ms"] * r["ntt_slow"]; nic = (n + 2) * a_us / 1000.0
                v.append(max(spu, ntt, nic))
            ys.append(gm(v))
        knee = next(N for N, y in zip(NS, ys) if y <= 1.02 * ys[-1])
        curves[(mach, a_us)] = (ys, knee)
        lines.append(f"{mach} alpha {lab:6s}: runtime at ceiling ({ceil_n} SPU) {ys[NS.index(ceil_n)]:7.1f} ms | plateau (8192) {ys[-1]:7.1f} ms | knee {knee:5d} SPU ({'beyond' if knee > ceil_n else 'within'} the ceiling) | min(knee, ceiling) {min(knee, ceil_n)}: {ys[NS.index(min(knee, ceil_n))]:7.1f} ms")
open(TXT, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
fig, ax = plt.subplots(figsize=(4.2, 3.1))
for mach, mk in (("L40S", "o"), ("B200", "s")):
    ceil_n = DES[mach][2]
    for a_us, lab in ALPHA:
        ys, knee = curves[(mach, a_us)]
        xs = [AREA(n) for n in NS]; i = NS.index(ceil_n)
        ax.plot(xs[:i + 1], ys[:i + 1], "-", marker=mk, ms=3.2, lw=1.1, color=COL[a_us], mec="black", mew=0.3, label=lab if mach == "L40S" else None, zorder=3)
        ax.plot(xs[i:], ys[i:], "--", marker=mk, ms=3.2, lw=1.0, color=COL[a_us], mec="black", mew=0.3, alpha=0.75, zorder=3)
        kx, ky = AREA(knee), ys[NS.index(knee)]
        ax.plot([kx], [ky], "*", ms=9, color=COL[a_us], mec="black", mew=0.5, zorder=6)
        ax.annotate(str(knee), (kx, ky), fontsize=5.5, xytext=(3, 3), textcoords="offset points", zorder=7)
for mach, ceil_n, lab in (("L40S", 192, "L40S Ceiling"), ("B200", 2048, "B200 Ceiling")):
    ax.axvline(AREA(ceil_n), color="#c62828", ls="--", lw=0.9, zorder=1); ax.text(AREA(ceil_n) * 1.08, 2.2e4, lab, color="#c62828", fontsize=6.5, ha="left", va="top")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(AREA(8) / 1.3, AREA(8192) * 1.3); ax.set_ylim(15, 3e4)
tick_n = [8, 32, 128, 512, 2048, 8192]
ax.set_xticks([AREA(n) for n in tick_n]); ax.set_xticklabels([f"{AREA(n):.2f}" for n in tick_n], fontsize=6.5)
ax.set_xlabel("Area (mm$^2$)", fontsize=8); ax.set_ylabel("Runtime (ms)", fontsize=8); ax.tick_params(axis="y", labelsize=6.5)
top = ax.secondary_xaxis("top"); top.set_xscale("log"); top.set_xticks([AREA(n) for n in tick_n]); top.set_xticklabels([str(n) for n in tick_n], fontsize=6.5); top.set_xlabel("SPU count", fontsize=8)
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
extra = [Line2D([], [], color="0.3", marker="o", ms=3.5, lw=0, mec="black", mew=0.3), Line2D([], [], color="0.3", marker="s", ms=3.5, lw=0, mec="black", mew=0.3),
         Line2D([], [], color="0.3", ls="--", lw=1), Line2D([], [], color="#c62828", ls="--", lw=0.9), Line2D([], [], color="0.3", marker="*", ms=8, lw=0, mec="black", mew=0.5)]
ax.legend(h + extra, l + ["L40S", "B200", "Past the ceiling", "Card ceiling", "Optimal budget"], fontsize=5.5, ncol=2, loc="lower left", frameon=True, title="Network latency", title_fontsize=6)
ax.grid(True, which="major", lw=0.3, color="0.85"); ax.set_axisbelow(True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
