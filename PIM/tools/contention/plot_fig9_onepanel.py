#!/usr/bin/env python3
"""Fig. 9 on one set of axes: end-to-end runtime vs SPU budget under six network latencies with the memory system fixed
(the submission's figure, regenerated from the final accounting), plus, at 500 us, the B200 runtime when the memory grows with the
budget (8 SPUs + 1 DRU per channel, 48..512 channels): without the DRU (submission four-step on the SMs) and with it; the
gap is the DRU gain and the star is the knee where the DRU design overtakes the merge NTT and the runtime enables it.
Usage: plot_fig9_onepanel.py FINAL_WINDOW_DIR OUT_PNG"""
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

# channel-scaled B200 runtimes at 500 us: without the DRU (sq) and with it (f4dru)
A500 = 500
def rt(rowsd, LN):
    return gm(max(r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"], (LN[(r["c"], r["t"], r["logN"])]["n"] + 2) * A500 / 1000.0) for r in rowsd.values())
LNB = L_("B200"); t_no, t_dru = [], []
for ch in CH:
    org = "b200" if ch == 256 else f"b200_ch{ch}"
    t_no.append(rt(cells(SQ, org, "sq"), LNB)); t_dru.append(rt(cells(MF, org, "f4dru"), LNB))
fig, ax = plt.subplots(figsize=(4.2, 3.3), gridspec_kw=dict(left=0.14, right=0.98, top=0.88, bottom=0.14))
for mach, mk in (("L40S", "o"), ("B200", "s")):
    ceil_n = DES[mach][2]
    for a_us, lab in ALPHA:
        ys, kn = curves[(mach, a_us)]; xs = [AREA(n) for n in NS]; i = NS.index(ceil_n)
        ax.plot(xs[:i + 1], ys[:i + 1], "-", marker=mk, ms=2.8, lw=0.9, color=COL[a_us], mec="black", mew=0.25, label=lab if mach == "L40S" else None, zorder=3, alpha=0.9)
        ax.plot(xs[i:], ys[i:], "--", marker=mk, ms=2.8, lw=0.8, color=COL[a_us], mec="black", mew=0.25, alpha=0.6, zorder=3)
        ax.plot([AREA(kn)], [ys[NS.index(kn)]], "*", ms=7, color=COL[a_us], mec="black", mew=0.4, zorder=6)
        ax.annotate(str(kn), (AREA(kn), ys[NS.index(kn)]), fontsize=4.6, xytext=(3, 3), textcoords="offset points", zorder=7)
for n_, lab in ((192, "L40S Ceiling"), (2048, "B200 Ceiling")):
    ax.axvline(AREA(n_), color="#c62828", ls="--", lw=0.8, zorder=1); ax.text(AREA(n_) * 1.08, 2.2e4, lab, color="#c62828", fontsize=6, ha="left", va="top")
GOLD = "#C9950F"; xs = [AREA(8 * c) for c in CH]
ax.plot(xs, t_no, "-", marker="D", ms=4.2, lw=1.6, color="0.25", mec="black", mew=0.3, zorder=5, label="B200, memory scaled, no DRU (500 µs)")
off = [i for i, c in enumerate(CH) if c < knee]; on = [i for i, c in enumerate(CH) if c >= knee]; i0 = off[-1]
yk = exp(log(t_dru[i0]) + (log(t_dru[i0 + 1]) - log(t_dru[i0])) * (log(knee) - log(CH[i0])) / (log(CH[i0 + 1]) - log(CH[i0]))); xk = AREA(8 * knee)
ax.plot([xs[i] for i in off] + [xk], [t_dru[i] for i in off] + [yk], "--", color=GOLD, lw=1.4, zorder=5)
ax.plot([xs[i] for i in off], [t_dru[i] for i in off], "D", ms=4.2, mfc="white", mec=GOLD, mew=1.2, zorder=6, label="B200, memory scaled, DRU design (not enabled: merge faster)")
ax.plot([xk] + [xs[i] for i in on], [yk] + [t_dru[i] for i in on], "-", color=GOLD, lw=1.8, zorder=5)
ax.plot([xs[i] for i in on], [t_dru[i] for i in on], "D", ms=4.2, color=GOLD, mec="black", mew=0.4, zorder=6, label="B200, memory scaled, DRU enabled by the runtime")
ax.plot([xk], [yk], "*", ms=12, color="#C8322B", mec="black", mew=0.6, zorder=8)
for j, (c, x, a, b) in enumerate(zip(CH, xs, t_no, t_dru)):
    ax.annotate(f"{c} ch\n{a/b:.2f}×", (x, b), fontsize=4.8, color="#7a5a08", xytext=(0, -20 if j == 3 else -7), textcoords="offset points", ha="center", va="top", zorder=9)
ax.annotate(f"DRU knee: {knee:.0f} channels ({8.2*knee/256:.1f} TB/s),\nDRU design overtakes the merge NTT", (xk, yk), fontsize=5.4, color="#C8322B", xytext=(AREA(8 * 26), 2300), textcoords="data", ha="left", va="center", arrowprops=dict(arrowstyle="-", color="#C8322B", lw=0.6, shrinkB=6), zorder=9)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(AREA(8) / 1.3, AREA(8192) * 1.3); ax.set_ylim(15, 3e4)
ax.set_ylabel("Runtime (ms)", fontsize=8); ax.set_xlabel("Area (mm$^2$)", fontsize=8); ax.tick_params(axis="y", labelsize=6.5)
tick_n = [8, 32, 128, 512, 2048, 8192]; ax.set_xticks([AREA(n) for n in tick_n]); ax.set_xticklabels([f"{AREA(n):.2f}" for n in tick_n], fontsize=6.5); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
top = ax.secondary_xaxis("top"); top.set_xscale("log"); top.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([AREA(n) for n in tick_n])); top.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(n) for n in tick_n]))
top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); top.tick_params(labelsize=6.5); top.set_xlabel("SPU count", fontsize=8)
ax.grid(True, which="major", lw=0.3, color="0.85"); ax.set_axisbelow(True)
h, l = ax.get_legend_handles_labels()
extra = [Line2D([], [], color="0.3", marker="o", ms=3.5, lw=0, mec="black", mew=0.3), Line2D([], [], color="0.3", marker="s", ms=3.5, lw=0, mec="black", mew=0.3),
         Line2D([], [], color="0.3", ls="--", lw=1), Line2D([], [], color="#c62828", ls="--", lw=0.9), Line2D([], [], color="0.3", marker="*", ms=8, lw=0, mec="black", mew=0.5)]
hl = list(zip(h, l)); lat = [(a, b) for a, b in hl if "memory scaled" not in b]; mem = [(a, b) for a, b in hl if "memory scaled" in b]
leg1 = ax.legend([a for a, _ in lat] + extra, [b for _, b in lat] + ["L40S", "B200", "Past the ceiling", "Card ceiling", "Optimal budget"], fontsize=4.9, ncol=2, loc="lower left", frameon=True, title="Memory fixed, network latency", title_fontsize=5.2)
ax.add_artist(leg1)
ax.legend([a for a, _ in mem], [b for _, b in mem], fontsize=4.9, loc="upper right", bbox_to_anchor=(0.995, 0.9), frameon=True, title="Memory scaled with the budget", title_fontsize=5.2)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.02); print("wrote", OUT)
print("channel-scaled 500us: no DRU", [round(x, 1) for x in t_no], "DRU", [round(x, 1) for x in t_dru], "gain", [round(a / b, 3) for a, b in zip(t_no, t_dru)])
