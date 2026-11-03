#!/usr/bin/env python3
"""Summary of cheddar_interfere_cfg_<tag>.csv (silicon only): Cheddar's NTT configs within reach of the standalone optimum,
each timed alone and beside the SPU-like row-burst aggressor. Per logN: the best standalone config, the best config at the
SPUs' DRAM share (co-run ms = alone ms x net slowdown interpolated at the share, as interfere_summary.py), the gain of
choosing by co-run time, and the slowdown spread by column radix M1, warp batching B1 and row radix M2.
Usage: interfere_cfg_summary.py TAG   (writes results_<tag>/cheddar_interfere_cfg_<tag>.txt)"""
import csv, os, sys
from statistics import mean
HERE = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1]
RES = os.path.join(HERE, "..", f"results_{TAG}")
CARD = {"l40s": dict(peak=864.0, shares=(0.15,)), "b200": dict(peak=8000.0, shares=(0.127, 0.164, 0.205))}[TAG]
import glob
rows = [r for f in sorted(glob.glob(f"{RES}/cheddar_interfere_cfg_{TAG}*.csv")) for r in csv.DictReader(open(f)) if r["victim"] == "cheddar"]   # original sweep + extensions
bad = sum(r["mismatches"] != "0" for r in rows)
cfg = lambda r: (r["P1"], r["M1"], r["B1"], r["M2"], r["maxrreg"])
cname = lambda c: f"{c[0]}+ M{c[1]} B{c[2]} M2 {c[3]} R{c[4]}"


def interp(pts, x):
    pts = [(0.0, 1.0)] + pts
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return float("nan")


def runs(lg):
    out = {}
    for c in {cfg(r) for r in rows if r["logN"] == str(lg)}:
        R = [r for r in rows if r["logN"] == str(lg) and cfg(r) == c]
        ctrl = {r["blocks"]: float(r["victim_ms"]) for r in R if r["mode"] == "sleep_only"}
        cur = sorted((float(r["agg_GBps"]) / CARD["peak"], float(r["victim_ms"]) / ctrl[r["blocks"]]) for r in R if r["mode"] == "rowburst" and r["blocks"] in ctrl)
        alone = [r for r in R if r["mode"] == "none"]
        if not alone or not cur: continue
        out[c] = (float(alone[0]["victim_ms"]), cur)
    return out


SH = CARD["shares"]
out = [f"{TAG}: Cheddar NTT configs under SPU-like DRAM interference, ms per polynomial multiply (batch 16); {bad} of {len(rows)} rows have mismatches",
       f"net slowdown = beside row-burst aggressor / beside sleep-only control (same blocks), at the SPUs' share of {CARD['peak']:.0f} GB/s: "
       + ", ".join(f"{s:.1%}" for s in SH), ""]
for s in SH:
    out += [f"share {s:.1%}: best standalone config vs best co-run config (chosen by alone ms x slowdown)",
            "logN | configs | best alone: config, ms, slowdown, co-run ms | best co-run: config, ms, slowdown, co-run ms | co-run gain | slowdown min..max"]
    for lg in sorted({int(r["logN"]) for r in rows}):
        D = runs(lg)
        if not D: continue
        co = {c: (t, interp(cur, s), t * interp(cur, s)) for c, (t, cur) in D.items()}
        co = {c: v for c, v in co.items() if v[1] == v[1]}          # drop configs with no aggressor point beyond the share
        if not co: continue
        a = min(co, key=lambda c: co[c][0]); b = min(co, key=lambda c: co[c][2])
        sl = [v[1] for v in co.values()]
        out.append(f"  {lg} | {len(co):3d} | {cname(a):24s} {co[a][0]:.4f} x{co[a][1]:.3f} = {co[a][2]:.4f} | "
                   f"{cname(b):24s} {co[b][0]:.4f} x{co[b][1]:.3f} = {co[b][2]:.4f} | {co[a][2] / co[b][2] - 1:+.1%} | {min(sl):.3f}..{max(sl):.3f}")
    out.append("")
    out += ["slowdown by knob (mean over the configs with that value), per logN:", "logN | M1=3 M1=4 M1=5 M1=6 | B1=0 B1=1 B1=2 B1=3 B1=4 | M2=3 M2=4 M2=5 M2=6 | R0 R64 R128"]
    for lg in sorted({int(r["logN"]) for r in rows}):
        D = runs(lg); co = {c: interp(cur, s) for c, (t, cur) in D.items()}; co = {c: v for c, v in co.items() if v == v}
        if not co: continue
        g = lambda i, val: (lambda v: f"{mean(v):.3f}" if v else "  -  ")([x for c, x in co.items() if c[i] == val])
        out.append(f"  {lg} | " + " ".join(g(1, m) for m in "3456") + " | " + " ".join(g(2, b) for b in "01234") + " | "
                   + " ".join(g(3, m) for m in "3456") + " | " + " ".join(g(4, r) for r in ("0", "64", "128")))
    out.append("")
txt = "\n".join(out) + "\n"; open(f"{RES}/cheddar_interfere_cfg_{TAG}.txt", "w").write(txt); print(txt)
