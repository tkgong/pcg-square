#!/usr/bin/env python3
"""One figure: geomean speedup of PCG^2 over the GPU baseline vs network bandwidth under the four co-scheduling
configurations (GPU serial / co-scheduled x PCG^2 serial / co-scheduled), both cards. Usage: plot_cosched_4config.py WIN22_DIR OUT_PNG"""
import io, contextlib, json, sys
from math import exp, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
sys.path.insert(0, "PIM/tools/contention/e2e")
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, gm
    from lanes import alpha_bw
W, OUT = sys.argv[1], sys.argv[2]
R = json.load(open(f"{W}/e2e_nom.json")); DES = {"L40S": ("l40s", "merge"), "B200": ("b200", "f4dru")}
GRID = [10 ** (x / 8) for x in range(8 * 7, 8 * 11 + 1)]
CFG = [("cc", "GPU co-sched / PCG$^2$ co-sched", "-"), ("sc", "GPU serial / PCG$^2$ co-sched", "--"), ("cs", "GPU co-sched / PCG$^2$ serial", "-."), ("ss", "GPU serial / PCG$^2$ serial", ":")]
COL = {"L40S": "#C8322B", "B200": "#C9950F"}
fig, ax = plt.subplots(figsize=(3.5, 2.6), gridspec_kw=dict(left=0.13, right=0.98, top=0.97, bottom=0.17))
for mach, (org, des) in DES.items():
    LN = L_(mach); rows = [r for r in R if r["org"] == org and r["design"] == des and r["tier"] == "fast"]
    curves = {k: [] for k, _, _ in CFG}
    for b in GRID:
        v = {k: [] for k in curves}
        for r in rows:
            k = (r["c"], r["t"], r["logN"]); tn = (LN[k]["n"] + 2) * alpha_bw(k[0], k[1], b)
            tg = r["gpu_ms"]; tp = max(r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"])
            v["ss"].append((tg + tn) / (tp + tn)); v["cs"].append(max(tg, tn) / (tp + tn)); v["sc"].append((tg + tn) / max(tp, tn)); v["cc"].append(max(tg, tn) / max(tp, tn))
        for k in curves: curves[k].append(gm(v[k]))
    for k, lab, ls in CFG:
        ax.plot(GRID, curves[k], ls, color=COL[mach], lw=1.4 if k == "cc" else 1.0, label=f"{mach}: {lab}")
for x in (40e9, 400e6): ax.axvline(x, color="0.8", lw=0.7, zorder=0)
ax.set_xscale("log"); ax.set_xlim(1e7, 1e11); ax.set_ylim(1, 8.5)
ax.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11]); ax.set_xticklabels(["10M", "100M", "1G", "10G", "100G"], fontsize=6.5); ax.xaxis.set_minor_locator(mt.NullLocator())
ax.tick_params(axis="y", labelsize=6.5); ax.set_xlabel("network bandwidth (bit/s)", fontsize=7.5, labelpad=1); ax.set_ylabel("speedup over GPU (geomean)", fontsize=7.5)
ax.grid(axis="y", lw=0.4, color="0.9"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=4.9, frameon=False, loc="upper left", ncol=2, handlelength=2.2, columnspacing=1.0, labelspacing=0.25)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
