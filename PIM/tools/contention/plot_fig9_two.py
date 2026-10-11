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

# ---------------- bottom data: channel count (= DRU count, bandwidth = count x per-channel), SPU array fixed per card ----------------
A500 = 500
SQ_ALL = load(f"{W}/fourstep_dru/e2e_chsweep_conservative_dru.json") + load(f"{W}/fourstep_dru/e2e_nom_conservative_dru.json")
CARD = {"B200": dict(org="b200", native=256, chs=[8, 16, 32, 48, 96, 128, 192, 256, 512], mk="s",
                     rows=load(f"{W}/win22/e2e_nom.json") + load(f"{W}/win22/e2e_f4sm.json") + load(f"{W}/channel_sweep/e2e.json") + load(f"{W}/channel_sweep/e2e_f4sm.json") + load(f"{W}/channel_sweep/e2e_b200_low.json") + load(f"{W}/channel_sweep/e2e_f4sm_b200_low.json")),
        "L40S": dict(org="l40s", native=24, chs=[8, 16, 24, 48, 96, 128, 192, 256, 512], mk="o",
                     rows=load(f"{W}/win22/e2e_nom.json") + load(f"{W}/win22/e2e_f4sm.json") + load(f"{W}/channel_sweep/e2e_l40s.json") + load(f"{W}/channel_sweep/e2e_l40s_low.json") + load(f"{W}/channel_sweep/e2e_f4sm_l40s.json"))}
ADRU = lambda n: n * 0.024924
B = {}
for mach, c in CARD.items():
    LNm = L_(mach); nat = cells(c["rows"], c["org"], "f4dru"); SPU_FIX = {k: r["spu_ms"] * r["spu_slow"] for k, r in nat.items()}
    def rt(rowsd): return gm(max(SPU_FIX[k], r["ntt_ms"] * r["ntt_slow"], (LNm[k]["n"] + 2) * A500 / 1000.0) for k, r in rowsd.items() if k in SPU_FIX)
    chs, t_dru, t_no, ratio = [], [], [], []
    for ch in c["chs"]:
        org = c["org"] if ch == c["native"] else f"{c['org']}_ch{ch}"
        d, m, q = cells(c["rows"], org, "f4dru"), cells(c["rows"], org, "merge"), cells(c["rows"], org, "f4sm")   # no DRU = same four-step kernels, transposes on the SMs (shared-memory tile transpose)
        if not d: continue
        sm = cells(c["rows"], c["org"], "f4sm")   # end-to-end: DRU design at this channel count vs the real card with the transposes in shared memory (flat baseline)
        chs.append(ch); t_dru.append(rt(d)); t_no.append(rt(sm)); ratio.append(t_no[-1] / t_dru[-1])
    kn = None
    for i in range(len(chs) - 1):
        if ratio[i] < 1 <= ratio[i + 1]: kn = exp(log(chs[i]) + (1 - ratio[i]) / (ratio[i + 1] - ratio[i]) * (log(chs[i + 1]) - log(chs[i]))); break
    B[mach] = dict(chs=chs, t=t_dru, t_no=t_no, ratio=ratio, knee=kn)
    print(mach, "channels", chs, "with DRU", [round(x, 1) for x in t_dru], "no DRU", [round(x, 1) for x in t_no], "SM-transpose/DRU lane", [round(x, 3) for x in ratio], "crossover at", kn)
fig = plt.figure(figsize=(4.2, 3.3)); gs = fig.add_gridspec(2, 2, height_ratios=[1.75, 1.0], hspace=0.62, wspace=0.34, left=0.14, right=0.98, top=0.9, bottom=0.11)
ax = fig.add_subplot(gs[0, :]); BX = {"B200": fig.add_subplot(gs[1, 0]), "L40S": fig.add_subplot(gs[1, 1])}
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
YL = {"B200": (28, 62), "L40S": (224, 274)}; CHT = [8, 16, 24, 48, 96, 128, 192, 256, 512]
for mach, c in CARD.items():
    bx = BX[mach]; d = B[mach]; xs = [ADRU(n) for n in d["chs"]]; i = d["chs"].index(c["native"])
    bx.plot(xs[:i + 1], d["t"][:i + 1], "-", marker=c["mk"], ms=3.2, lw=1.3, color=COLB[mach], mec="black", mew=0.35, zorder=5, label="with DRU", clip_on=True)
    bx.plot(xs[i:], d["t"][i:], "--", marker=c["mk"], ms=3.2, lw=1.1, color=COLB[mach], mec="black", mew=0.35, alpha=0.8, zorder=5)
    bx.axhline(d["t_no"][0], ls=":", lw=1.3, color=COLB[mach], zorder=4, label="transpose in shared memory")
    if d["knee"]:
        i0 = max(j for j, n in enumerate(d["chs"]) if n < d["knee"]); kn = d["knee"]
        bx.plot([ADRU(kn)], [d["t_no"][0]], "*", ms=9, color=COLB[mach], mec="black", mew=0.5, zorder=8); bx.annotate(f"{kn:.0f}", (ADRU(kn), d["t_no"][0]), fontsize=4.4, xytext=(3, 3), textcoords="offset points", zorder=9)
    bx.axvline(ADRU(c["native"]), color="#c62828", ls="--", lw=0.8, zorder=1)
    lo, hi = YL[mach]; bx.text(ADRU(c["native"]) * 1.08, hi - 0.03 * (hi - lo), f"{mach} Ceiling", fontsize=5.0, color="#c62828", va="top")
    # off-scale points: mark the first on-scale point's left neighbour with its value
    for x, n, t in zip(xs, d["chs"], d["t"]):
        if t > hi: bx.annotate(f"{t:.0f}", (x, hi), fontsize=4.2, xytext=(0, -7), textcoords="offset points", ha="center", color=COLB[mach], zorder=9)
    XD = [ADRU(n) for n in CHT]; bx.set_xscale("log"); bx.set_xticks(XD); bx.set_xticklabels([f"{x:.1f}" if x >= 1 else f"{x:.2f}" for x in XD], fontsize=4.6, rotation=45)
    bx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); bx.set_xlim(ADRU(7), ADRU(590)); bx.set_ylim(lo, hi); bx.tick_params(axis="y", labelsize=6)
    bx.set_xlabel("DRU area (mm$^2$)", fontsize=6.5, labelpad=1)
    topb = bx.secondary_xaxis("top"); topb.set_xscale("log"); topb.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(XD)); topb.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(n) for n in CHT]))
    topb.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); topb.tick_params(labelsize=4.6); topb.set_xlabel("DRU count (one per channel)", fontsize=6, labelpad=1)
    bx.legend(fontsize=4.6, frameon=True, loc="lower left", handlelength=1.8, labelspacing=0.2, borderpad=0.3, title=mach, title_fontsize=5.2)
    bx.grid(True, which="major", lw=0.3, color="0.85"); bx.set_axisbelow(True)
BX["B200"].set_ylabel("Runtime (ms)", fontsize=7)

fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
