#!/usr/bin/env python3
"""Summary of cheddar_interfere_<tag>.csv (silicon only): per logN and measured config (each design's best big-N config for
each column M1), the net slowdown (victim next to the aggressor / victim next to the sleep-only control at the same blocks)
at the SPUs' share of DRAM bandwidth; and the four-step SM lane against Cheddar, each at its fastest config alone and at
its fastest config under that interference.
SPU share = pim_allbank_slot_cycles / memory_system_cycles of the co-simulation's cells (results/contention/final_window/
cheddar/win22): L40S 14.8-15.3% in every cell; B200 12.7% ((8,4), (8,8) at 2^24) to 20.6% ((2,64), (2,128)).  The slowdown
at a share is the linear interpolation between the two measured aggressor points around it.
Usage: interfere_summary.py TAG   (writes results_<tag>/cheddar_interfere_<tag>.txt)"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1]
RES = os.path.join(HERE, "..", f"results_{TAG}")
CARD = {"l40s": dict(peak=864.0, shares=(0.15,)), "b200": dict(peak=8000.0, shares=(0.127, 0.164, 0.205))}[TAG]
rows = list(csv.DictReader(open(f"{RES}/cheddar_interfere_{TAG}.csv")))
bad = sum(r["mismatches"] != "0" for r in rows)
cfg = lambda r: (r["P1"], r["M1"], r["B1"], r["M2"], r["maxrreg"])
cname = lambda c: f"{c[0]}+ M{c[1]} B{c[2]} M2 {c[3]} R{c[4]}"


def interp(pts, x):   # piecewise linear through the measured points, 1 at zero aggressor bandwidth
    pts = [(0.0, 1.0)] + pts
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return float("nan")   # beyond the largest measured aggressor bandwidth


def runs(lg, vic):
    """per config of this victim: (alone ms, batch, {mode: [(share, net slowdown)]})"""
    out = {}
    for c in {cfg(r) for r in rows if r["logN"] == str(lg) and r["victim"] == vic}:
        R = [r for r in rows if r["logN"] == str(lg) and r["victim"] == vic and cfg(r) == c]
        ctrl = {r["blocks"]: float(r["victim_ms"]) for r in R if r["mode"] == "sleep_only"}
        cur = {m: sorted((float(r["agg_GBps"]) / CARD["peak"], float(r["victim_ms"]) / ctrl[r["blocks"]]) for r in R if r["mode"] == m and r["blocks"] in ctrl)
               for m in ("rowburst", "stream")}
        alone = [r for r in R if r["mode"] == "none"][0]
        out[c] = (float(alone["victim_ms"]), alone["batch"], cur)
    return out


SH = CARD["shares"]
out = [f"{TAG}: silicon interference, ms per polynomial multiply; {bad} of {len(rows)} rows have mismatches",
       "net slowdown = victim beside aggressor / victim beside sleep-only control (same blocks), at the SPUs' share of "
       f"{CARD['peak']:.0f} GB/s: " + ", ".join(f"{s:.1%}" for s in SH),
       "co-run ms = alone ms x row-burst net slowdown", "",
       "logN batch | design   config (P1 M1 B1 M2 cap)  | alone ms | row burst slowdown | stream slowdown | co-run ms (row burst)"]
LG = sorted({int(r["logN"]) for r in rows})
for lg in LG:
    for vic in ("merge", "cheddar", "f4c_sm", "f4c_full", "f4b_sm"):
        for c, (t, b, cur) in sorted(runs(lg, vic).items(), key=lambda kv: kv[1][0]):
            out.append(f"  {lg} {b:>3s} | {vic:8s} {'-' if vic == 'merge' else cname(c):24s} | {t:8.4f} | "
                       + " ".join(f"{interp(cur['rowburst'], s):.3f}" for s in SH) + " | " + " ".join(f"{interp(cur['stream'], s):.3f}" for s in SH)
                       + " | " + " ".join(f"{t * interp(cur['rowburst'], s):.4f}" for s in SH))
for title, vic in (("four-step SM lane (transposes on the DRU)", "f4c_sm"), ("butterfly-DRU SM lane (column pass in the DRU, row kernel only)", "f4b_sm")):
    out += ["", f"{title} vs Cheddar, each design at its fastest measured config:",
            f"time saved = 1 - {vic} / Cheddar; alone = best standalone configs; co-run = best co-run configs beside the SPU-like",
            "row-burst aggressor at each share", f"logN batch | Cheddar alone | {vic} alone | saved alone | " + " | ".join(f"saved co-run @{s:.1%}" for s in SH)]
    for lg in LG:
        C, F = runs(lg, "cheddar"), runs(lg, vic)
        if not C or not F: continue
        ca, fa = min(v[0] for v in C.values()), min(v[0] for v in F.values())
        co = lambda D, s: min(t * interp(cur["rowburst"], s) for t, b, cur in D.values())
        out.append(f"  {lg} {next(iter(C.values()))[1]:>3s} | {ca:.4f} | {fa:.4f} | {1 - fa / ca:+.1%} | "
                   + " | ".join(f"{1 - co(F, s) / co(C, s):+.1%} ({co(C, s):.4f} vs {co(F, s):.4f})" for s in SH))
txt = "\n".join(out) + "\n"; open(f"{RES}/cheddar_interfere_{TAG}.txt", "w").write(txt); print(txt)
