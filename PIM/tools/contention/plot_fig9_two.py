#!/usr/bin/env python3
"""Fig. 9 combined: (top) end-to-end runtime vs SPU budget under six network latencies, memory system fixed (as in the
submission, regenerated from the final accounting); (bottom) on the same x axis, the butterfly DRU design when the memory grows
with the budget (8 SPUs + 1 DRU per channel, 8..512 channels of each card's DRAM) against the real card with Cheddar's NTT
on the SMs, and the crossover where the runtime enables it.    Usage: plot_fig9_two.py RUN_DIR OUT_PNG"""
import io, contextlib, json, os, sys
from math import ceil, exp, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
sys.path.insert(0, HERE)
import final_design as FD
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_
W, OUT = sys.argv[1], sys.argv[2]
def gm(v): v = list(v); return exp(sum(map(log, v)) / len(v))
def load(p): return json.load(open(p)) if os.path.exists(p) else []
AREA = lambda n: FD.area_mm2(n, W)
# ---------------- top: runtime vs SPU budget (memory fixed) ----------------
DES = {M: (o, FD.des(o, W), FD.NSPU[o]) for o, M in FD.MACH.items()}
NS = [8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
ALPHA = [(100, "100 µs"), (500, "500 µs"), (10000, "10 ms"), (20000, "20 ms"), (30000, "30 ms"), (60000, "60 ms")]
COL = {100: "#1f3a93", 500: "#6a1b9a", 10000: "#e6a700", 20000: "#2e8b57", 30000: "#e8641b", 60000: "#c62828"}
R = load(f"{W}/e2e_nom.json"); curves = {}
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
# ---------------- bottom: the DRU design vs channel count (memory scaled), both cards ----------------
# channel count (= DRU count, bandwidth = count x per-channel), SPU array fixed per card: end-to-end runtime (alpha = 500 us)
# of the butterfly-DRU design (f4b) at each channel count vs the real card with Cheddar's NTT on the SMs (flat baseline)
def cells(rows, org, des): return {(r["c"], r["t"], r["logN"]): r for r in rows if r["org"] == org and r["design"] == des and r["tier"] == "fast"}
A500 = 500
ALL = load(f"{W}/e2e_nom.json") + load(f"{W}/channel_sweep.json")
CARD = {"B200": dict(org="b200", native=256, chs=[8, 16, 32, 48, 96, 128, 192, 256, 512], mk="s"), "L40S": dict(org="l40s", native=24, chs=[8, 16, 24, 48, 96, 128, 192, 256, 512], mk="o")}
ADRU = lambda n, org="b200": n * FD.dru_mm2(org, run_dir=W)
B = {}
for mach, c in CARD.items():
    LNm = L_(mach); nat = cells(ALL, c["org"], FD.SM_NTT); SPU_FIX = {k: r["spu_ms"] * r["spu_slow"] for k, r in nat.items()}
    def rt(rowsd): return gm(max(SPU_FIX[k], r["ntt_ms"] * r["ntt_slow"], (LNm[k]["n"] + 2) * A500 / 1000.0) for k, r in rowsd.items() if k in SPU_FIX)
    chs, t_dru, t_no, ratio = [], [], [], []
    for ch in c["chs"]:
        org = c["org"] if ch == c["native"] else f"{c['org']}_ch{ch}"
        d = cells(ALL, org, "f4b")
        if not d: continue
        chs.append(ch); t_dru.append(rt(d)); t_no.append(rt(nat)); ratio.append(t_no[-1] / t_dru[-1])
    kn = None
    for i in range(len(chs) - 1):
        if ratio[i] < 1 <= ratio[i + 1]: kn = exp(log(chs[i]) + (1 - ratio[i]) / (ratio[i + 1] - ratio[i]) * (log(chs[i + 1]) - log(chs[i]))); break
    B[mach] = dict(chs=chs, t=t_dru, t_no=t_no, ratio=ratio, knee=kn)
    print(mach, "channels", chs, "with butterfly DRU", [round(x, 1) for x in t_dru], "Cheddar on the SMs (real card)", [round(x, 1) for x in t_no], "ratio", [round(x, 3) for x in ratio], "crossover at", kn)
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
COLB = {"B200": "#C9950F", "L40S": "#C8322B"}
CHT = [8, 16, 24, 48, 96, 128, 192, 256, 512]; XD = [ADRU(n) for n in CHT]; LO, HI = 0.86, 1.30
# one panel, both cards: end-to-end runtime with the DRU normalized to the real card with the transposes in shared memory (= 1)
bx.axhline(1.0, ls=":", lw=1.3, color="0.25", zorder=4, label="Cheddar NTT on the SMs (real card)")
for mach, c in CARD.items():
    d = B[mach]
    if not d["chs"]: continue
    xs = [ADRU(n, c["org"]) for n in d["chs"]]; ys = [t / d["t_no"][0] for t in d["t"]]
    bx.plot(xs, ys, "-", marker=c["mk"], ms=3.2, lw=1.3, color=COLB[mach], mec="black", mew=0.35, zorder=5, label=f"{mach}, butterfly DRU")
    if d["knee"]:
        kn = d["knee"]; bx.plot([ADRU(kn)], [1.0], "*", ms=9, color=COLB[mach], mec="black", mew=0.5, zorder=8); bx.annotate(f"{kn:.0f}", (ADRU(kn), 1.0), fontsize=4.4, xytext=(3, 3), textcoords="offset points", zorder=9)
    bx.axvline(ADRU(c["native"]), color="#c62828", ls="--", lw=0.8, zorder=1); bx.text(ADRU(c["native"]) * 1.06, 1.09, f"{mach} Ceiling", fontsize=5.0, color="#c62828", va="top")
bx.set_xscale("log"); bx.set_xticks(XD); bx.set_xticklabels([f"{x:.2f}" for x in XD], fontsize=5.5); bx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); bx.set_xlim(ADRU(7), ADRU(590))
bx.set_ylim(LO, HI); bx.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3]); bx.tick_params(axis="y", labelsize=6)
bx.set_xlabel("DRU area (mm$^2$)", fontsize=7, labelpad=1); bx.set_ylabel("Normalized runtime", fontsize=7)
topb = bx.secondary_xaxis("top"); topb.set_xscale("log"); topb.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(XD)); topb.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(n) for n in CHT]))
topb.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); topb.tick_params(labelsize=6); topb.set_xlabel("DRU count (one per channel)", fontsize=7, labelpad=2)
bx.legend(fontsize=4.6, frameon=True, loc="upper right", handlelength=1.8, ncol=1, labelspacing=0.2, borderpad=0.3)
bx.grid(True, which="major", lw=0.3, color="0.85"); bx.set_axisbelow(True)

fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
