#!/usr/bin/env python3
"""Fig. 9 combined: (top) end-to-end runtime vs SPU budget under six network latencies, memory system fixed (as in the
submission, regenerated from the final accounting); (bottom) on the same x axis, the DRU's gain when the memory grows
with the budget (8 SPUs + 1 DRU per channel, 48..512 channels on B200-class HBM3e) and the knee where the runtime
enables the DRU.    Usage: plot_fig9_two.py FINAL_WINDOW_DIR OUT_PNG"""
import io, contextlib, json, os, sys
from math import ceil, exp, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_
W, OUT = sys.argv[1], sys.argv[2]
def gm(v): v = list(v); return exp(sum(map(log, v)) / len(v))
def load(p): return json.load(open(p)) if os.path.exists(p) else []
AREA = lambda n: n * 0.253970 + (n / 8) * 0.024924
# ---------------- top: runtime vs SPU budget (memory fixed) ----------------
DES = {"L40S": ("l40s", "merge", 192), "B200": ("b200", "f4dru", 2048)}
NS = [8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
ALPHA = [(100, "100 µs"), (500, "500 µs"), (10000, "10 ms"), (20000, "20 ms"), (30000, "30 ms"), (60000, "60 ms")]
COL = {100: "#1f3a93", 500: "#6a1b9a", 10000: "#e6a700", 20000: "#2e8b57", 30000: "#e8641b", 60000: "#c62828"}
R = load(f"{W}/win22/e2e_nom.json"); curves = {}
for mach, (org, des, ceil_n) in DES.items():
    LN = L_(mach); rows = [r for r in R if r["org"] == org and r["design"] == des and r["tier"] == "fast"]
    for a_us, lab in ALPHA:
        ys = []
        for N in NS:
            v = []
            for r in rows:
                k = (r["c"], r["t"], r["logN"]); I = k[0] ** 2 * k[1] ** 2; n = LN[k]["n"]
                t_inst = r["spu_ms"] / ceil(I / ceil_n)
                v.append(max(t_inst * ceil(I / N) * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"], (n + 2) * a_us / 1000.0))
            ys.append(gm(v))
        curves[(mach, a_us)] = (ys, next(N for N, y in zip(NS, ys) if y <= 1.02 * ys[-1]))
# ---------------- bottom: DRU gain vs channel count (memory scaled) ----------------
def cells(rows, org, des): return {(r["c"], r["t"], r["logN"]): r for r in rows if r["org"] == org and r["design"] == des and r["tier"] == "fast"}
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]
MF = load(f"{W}/channel_sweep/e2e.json") + load(f"{W}/win22/e2e_nom.json"); SQ = load(f"{W}/fourstep_dru/e2e_chsweep_conservative_dru.json") + load(f"{W}/fourstep_dru/e2e_nom_conservative_dru.json")
CH = [48, 96, 128, 192, 256, 512]; gain, ratio = [], []
for ch in CH:
    org = "b200" if ch == 256 else f"b200_ch{ch}"; m, d, s = cells(MF, org, "merge"), cells(MF, org, "f4dru"), cells(SQ, org, "sq"); ks = sorted(d)
    gain.append(gm(s[k]["pcg_ms"] / d[k]["pcg_ms"] for k in ks)); ratio.append(gm(lane(m[k]) / lane(d[k]) for k in ks))
knee = next(exp(log(CH[i]) + (1 - ratio[i]) / (ratio[i + 1] - ratio[i]) * (log(CH[i + 1]) - log(CH[i]))) for i in range(len(CH) - 1) if ratio[i] < 1 <= ratio[i + 1])
CSTAR = 240 * (1 << 24) / 2.2766e-3 / (8.2e12 / 256 * (1 - 0.21))

# ---------------- bottom data: channel-scaled B200 runtime at 500 us ----------------
A500 = 500; LNB = L_("B200")
SPU_FIX = {k: r["spu_ms"] * r["spu_slow"] for k, r in cells(MF, "b200", "f4dru").items()}     # the B200 SPU array (2,048 SPUs) at every channel count
def rt(rowsd): return gm(max(SPU_FIX[k], r["ntt_ms"] * r["ntt_slow"], (LNB[k]["n"] + 2) * A500 / 1000.0) for k, r in rowsd.items())
t_no, t_dru, t_m = [], [], []
for ch in CH:
    org = "b200" if ch == 256 else f"b200_ch{ch}"
    t_no.append(rt(cells(SQ, org, "sq"))); t_dru.append(rt(cells(MF, org, "f4dru"))); t_m.append(rt(cells(MF, org, "merge")))
print("channels", CH); print("no DRU", [round(x, 1) for x in t_no]); print("merge ", [round(x, 1) for x in t_m]); print("DRU   ", [round(x, 1) for x in t_dru]); print("gain  ", [round(a / b, 3) for a, b in zip(t_no, t_dru)])
BW = [8.2 * c / 256 for c in CH]
# ---------------- draw: two panels, each with its own x axis ----------------
fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.2, 3.3), gridspec_kw=dict(height_ratios=[1.75, 1.0], hspace=0.62, left=0.14, right=0.98, top=0.9, bottom=0.11))
for mach, mk in (("L40S", "o"), ("B200", "s")):
    ceil_n = DES[mach][2]
    for a_us, lab in ALPHA:
        ys, kn = curves[(mach, a_us)]; xs = [AREA(n) for n in NS]; i = NS.index(ceil_n)
        ax.plot(xs[:i + 1], ys[:i + 1], "-", marker=mk, ms=2.6, lw=0.9, color=COL[a_us], mec="black", mew=0.25, label=lab if mach == "L40S" else None, zorder=3)
        ax.plot(xs[i:], ys[i:], "--", marker=mk, ms=2.6, lw=0.8, color=COL[a_us], mec="black", mew=0.25, alpha=0.75, zorder=3)
        ax.plot([AREA(kn)], [ys[NS.index(kn)]], "*", ms=7, color=COL[a_us], mec="black", mew=0.4, zorder=6)
        ax.annotate(str(kn), (AREA(kn), ys[NS.index(kn)]), fontsize=4.4, xytext=(3, 2), textcoords="offset points", zorder=7)
for n_, lab in ((192, "L40S Ceiling"), (2048, "B200 Ceiling")):
    ax.axvline(AREA(n_), color="#c62828", ls="--", lw=0.8, zorder=1); ax.text(AREA(n_) * 1.08, 2.2e4, lab, color="#c62828", fontsize=5.5, ha="left", va="top")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(AREA(8) / 1.3, AREA(8192) * 1.3); ax.set_ylim(15, 3e4)
tick_n = [8, 32, 128, 512, 2048, 8192]; ax.set_xticks([AREA(n) for n in tick_n]); ax.set_xticklabels([f"{AREA(n):.2f}" for n in tick_n], fontsize=6); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("Area (mm$^2$)", fontsize=7, labelpad=1); ax.set_ylabel("Runtime (ms)", fontsize=7); ax.tick_params(axis="y", labelsize=6); ax.grid(True, which="major", lw=0.3, color="0.85"); ax.set_axisbelow(True)
top = ax.secondary_xaxis("top"); top.set_xscale("log"); top.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([AREA(n) for n in tick_n])); top.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(n) for n in tick_n]))
top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); top.tick_params(labelsize=6); top.set_xlabel("SPU count", fontsize=7, labelpad=2)
h, l = ax.get_legend_handles_labels()
extra = [Line2D([], [], color="0.3", marker="o", ms=3.5, lw=0, mec="black", mew=0.3), Line2D([], [], color="0.3", marker="s", ms=3.5, lw=0, mec="black", mew=0.3),
         Line2D([], [], color="0.3", ls="--", lw=1), Line2D([], [], color="#c62828", ls="--", lw=0.9), Line2D([], [], color="0.3", marker="*", ms=8, lw=0, mec="black", mew=0.5)]
ax.legend(h + extra, l + ["L40S", "B200", "Past the ceiling", "Card ceiling", "Optimal budget"], fontsize=4.4, ncol=3, loc="lower left", frameon=True, title="Network latency", title_fontsize=4.8, columnspacing=0.8, handlelength=1.4, borderpad=0.3, labelspacing=0.2)
ax.text(0.02, 0.9, "(a) SPU budget, memory system fixed", transform=ax.transAxes, fontsize=5.8, va="top")
GOLD = "#C9950F"
bx.plot(CH, t_no, "-", marker="D", ms=3.4, lw=1.3, color="0.3", mec="black", mew=0.3, zorder=4, label="no DRU: four-step on the SMs")
bx.plot(CH, t_m, "-", marker="o", ms=3.0, lw=1.0, color="#C8322B", mec="black", mew=0.3, zorder=4, label="merge NTT (runtime fallback)")
off = [i for i, c in enumerate(CH) if c < knee]; on = [i for i, c in enumerate(CH) if c >= knee]; i0 = off[-1]
yk = exp(log(t_dru[i0]) + (log(t_dru[i0 + 1]) - log(t_dru[i0])) * (log(knee) - log(CH[i0])) / (log(CH[i0 + 1]) - log(CH[i0])))
bx.plot([CH[i] for i in off] + [knee], [t_dru[i] for i in off] + [yk], "--", color=GOLD, lw=1.2, zorder=5)
bx.plot([CH[i] for i in off], [t_dru[i] for i in off], "D", ms=3.4, mfc="white", mec=GOLD, mew=1.1, zorder=6, label="with DRU, not enabled (merge faster)")
bx.plot([knee] + [CH[i] for i in on], [yk] + [t_dru[i] for i in on], "-", color=GOLD, lw=1.5, zorder=5)
bx.plot([CH[i] for i in on], [t_dru[i] for i in on], "D", ms=3.4, color=GOLD, mec="black", mew=0.4, zorder=6, label="with DRU, enabled by the runtime")
bx.plot([knee], [yk], "*", ms=10, color="#C8322B", mec="black", mew=0.5, zorder=8)
for c, a, b in zip(CH, t_no, t_dru): bx.annotate(f"{a/b:.2f}×", (c, b), fontsize=4.4, color="#7a5a08", xytext=(0, -8), textcoords="offset points", ha="center", zorder=9)
bx.annotate(f"DRU knee {knee:.0f} ch ({8.2*knee/256:.1f} TB/s)", (knee, yk), fontsize=4.8, color="#C8322B", xytext=(50, 36), textcoords="data", ha="left", va="center", arrowprops=dict(arrowstyle="-", color="#C8322B", lw=0.6, shrinkB=5), zorder=9)
bx.axvline(256, color="#c62828", ls="--", lw=0.8, zorder=1); bx.text(256 * 1.04, 29, "B200", fontsize=5, color="#c62828", va="bottom")
bx.set_xscale("log"); bx.set_yscale("log"); bx.set_xticks(CH); bx.set_xticklabels([str(c) for c in CH], fontsize=6); bx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); bx.set_xlim(42, 590)
bx.set_ylim(28, 120); bx.set_yticks([30, 50, 100]); bx.set_yticklabels(["30", "50", "100"], fontsize=6); bx.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
bx.set_xlabel("memory channels (SPU array fixed at the B200 design, 2,048 SPUs)", fontsize=6.5, labelpad=1); bx.set_ylabel("Runtime (ms)", fontsize=7)
topb = bx.secondary_xaxis("top"); topb.set_xscale("log"); topb.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(CH)); topb.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{b:.1f}" for b in BW]))
topb.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); topb.tick_params(labelsize=5.5); topb.set_xlabel("aggregate bandwidth (TB/s)", fontsize=6.5, labelpad=2)
bx.grid(True, which="major", lw=0.3, color="0.85"); bx.set_axisbelow(True)
bx.legend(fontsize=4.4, frameon=False, loc="upper right", handlelength=1.4, labelspacing=0.2)
bx.text(0.02, 0.06, "(b) channel count, SPUs fixed, 500 µs", transform=bx.transAxes, fontsize=5.8, va="bottom")
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
