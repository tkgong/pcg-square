#!/usr/bin/env python3
"""Summary of cheddar_interfere_<tag>.csv (silicon only, no model): per logN and victim, the net slowdown
(victim next to the aggressor / victim next to the sleep-only control at the same blocks) at the SPUs' share of DRAM
bandwidth (interfere_at_spu.py: L40S 10-15%, B200 16-21% of peak), stream and SPU-like row-burst aggressor; and the
four-step SM lane with its DRU transposes stood in by extra streaming bandwidth (96 N B per multiply at the lane's own
co-run speed, solved as a fixed point), giving the silicon NTT lane ratio Cheddar / four-step (gather excluded).
Usage: interfere_summary.py TAG   (writes results_<tag>/cheddar_interfere_<tag>.txt)"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1]
RES = os.path.join(HERE, "..", f"results_{TAG}")
CARD = {"l40s": dict(peak=864.0, spu=(0.10, 0.15)), "b200": dict(peak=8000.0, spu=(0.16, 0.21))}[TAG]
rows = list(csv.DictReader(open(f"{RES}/cheddar_interfere_{TAG}.csv")))
bad = sum(r["mismatches"] != "0" for r in rows)


def curve(lg, vic, mode):
    R = [r for r in rows if r["logN"] == str(lg) and r["victim"] == vic]
    ctrl = {r["blocks"]: float(r["victim_ms"]) for r in R if r["mode"] == "sleep_only"}
    return sorted((float(r["agg_GBps"]) / CARD["peak"], float(r["victim_ms"]) / ctrl[r["blocks"]]) for r in R if r["mode"] == mode and r["blocks"] in ctrl)


def interp(pts, x):   # piecewise linear through the measured points, 1 at zero aggressor bandwidth
    pts = [(0.0, 1.0)] + pts
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return float("nan")   # beyond the largest measured aggressor bandwidth


def base(lg, vic):
    v = [float(r["victim_ms"]) for r in rows if r["logN"] == str(lg) and r["victim"] == vic and r["mode"] == "none"]
    return v[0] if v else float("nan")


lo, hi = CARD["spu"]; mid = (lo + hi) / 2
out = [f"{TAG}: silicon interference, batch 16, ms per polynomial multiply; {bad} of {len(rows)} rows have mismatches",
       f"net slowdown = victim beside aggressor / victim beside sleep-only control (same blocks), interpolated at the SPUs' share {lo:.0%}-{hi:.0%} of {CARD['peak']:.0f} GB/s",
       "", "logN victim    alone ms | stream @SPU    | row burst @SPU | max measured aggressor share"]
LG = sorted({int(r["logN"]) for r in rows})
for lg in LG:
    for vic in ("merge", "cheddar", "f4c_sm", "f4c_full"):
        s, rb = curve(lg, vic, "stream"), curve(lg, vic, "rowburst")
        if not s: continue
        out.append(f"  {lg}  {vic:8s} {base(lg, vic):8.4f} | {interp(s, lo):.3f}-{interp(s, hi):.3f}    | {interp(rb, lo):.3f}-{interp(rb, hi):.3f}    | {max(x for x, _ in s):.0%}")
out += ["", "four-step SM lane with the DRU's 96 N B per multiply added as streaming bandwidth (fixed point on the lane's co-run time);",
        "lane ratio Cheddar / four-step on silicon (> 1: four-step faster), Cheddar under the SPU aggressor alone:",
        "logN | DRU share of peak | f4c_sm slowdown (SPU+DRU) | Cheddar slowdown (SPU) | alone ratio | co-run ratio (stream) | co-run ratio (row burst SPU + stream DRU)"]
for lg in LG:
    t_c, t_f = base(lg, "cheddar"), base(lg, "f4c_sm"); s_f, s_c, rb_c, rb_f = curve(lg, "f4c_sm", "stream"), curve(lg, "cheddar", "stream"), curve(lg, "cheddar", "rowburst"), curve(lg, "f4c_sm", "rowburst")
    if not s_f or not s_c: continue
    dru_gb = 96 * (1 << lg) / 1e9   # GB per polynomial multiply
    dru_at = lambda S: dru_gb / (t_f * S / 1e3) / CARD["peak"]   # DRU share of peak when the lane runs S x slower

    def fixed(g):   # S = g(S) by bisection; g undefined (beyond the measured bandwidth) counts as "S too small"
        a, b = 1.0, 8.0
        for _ in range(60):
            m = (a + b) / 2; v = g(m)
            if v != v or v > m: a = m
            else: b = m
        return b if b < 7.99 else float("nan")
    S = fixed(lambda S: interp(s_f, mid + dru_at(S)))
    # row burst for the SPU part, stream for the DRU part: the SPU-level row-burst penalty times the stream penalty of the DRU bytes
    S_mix = fixed(lambda S: interp(rb_f, mid) * interp(s_f, dru_at(S)))
    dru = dru_at(S)
    out.append(f"  {lg} | {dru:6.1%} | {S:.3f} | {interp(s_c, mid):.3f} | {t_c / t_f:.3f} | {t_c * interp(s_c, mid) / (t_f * S):.3f} | {t_c * interp(rb_c, mid) / (t_f * S_mix):.3f}")
txt = "\n".join(out) + "\n"; open(f"{RES}/cheddar_interfere_{TAG}.txt", "w").write(txt); print(txt)
