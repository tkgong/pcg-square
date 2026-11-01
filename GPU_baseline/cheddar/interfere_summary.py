#!/usr/bin/env python3
"""Summary of cheddar_interfere_<tag>.csv (silicon only): per logN, each design at its best big-N config, the net slowdown
(victim next to the aggressor / victim next to the sleep-only control at the same blocks) at the SPUs' share of DRAM
bandwidth, and the four-step SM lane against Cheddar with and without that interference.
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


def curve(lg, vic, mode):
    R = [r for r in rows if r["logN"] == str(lg) and r["victim"] == vic]
    ctrl = {r["blocks"]: float(r["victim_ms"]) for r in R if r["mode"] == "sleep_only"}
    return sorted((float(r["agg_GBps"]) / CARD["peak"], float(r["victim_ms"]) / ctrl[r["blocks"]]) for r in R if r["mode"] == mode and r["blocks"] in ctrl)


def interp(pts, x):   # piecewise linear through the measured points, 1 at zero aggressor bandwidth
    pts = [(0.0, 1.0)] + pts
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return float("nan")   # beyond the largest measured aggressor bandwidth


def one(lg, vic):
    R = [r for r in rows if r["logN"] == str(lg) and r["victim"] == vic and r["mode"] == "none"]
    return R[0] if R else None


out = [f"{TAG}: silicon interference, ms per polynomial multiply; {bad} of {len(rows)} rows have mismatches",
       "net slowdown = victim beside aggressor / victim beside sleep-only control (same blocks), at the SPUs' share of "
       f"{CARD['peak']:.0f} GB/s: " + ", ".join(f"{s:.1%}" for s in CARD["shares"]),
       "", "logN batch | design   config (P1 M1 B1 M2 cap) | alone ms | row burst @share | stream @share"]
LG = sorted({int(r["logN"]) for r in rows})
for lg in LG:
    for vic in ("merge", "cheddar", "f4c_sm", "f4c_full"):
        r = one(lg, vic)
        if not r: continue
        rb, st = curve(lg, vic, "rowburst"), curve(lg, vic, "stream")
        cfg = "-" if vic == "merge" else f"{r['P1']} {r['M1']} {r['B1']} {r['M2']} R{r['maxrreg']}"
        out.append(f"  {lg} {r['batch']:>3s} | {vic:8s} {cfg:18s} | {float(r['victim_ms']):8.4f} | "
                   + " ".join(f"{interp(rb, s):.3f}" for s in CARD["shares"]) + " | " + " ".join(f"{interp(st, s):.3f}" for s in CARD["shares"]))
out += ["", "four-step SM lane (transposes on the DRU) vs Cheddar: time saved = 1 - four-step / Cheddar, alone and beside the",
        "SPU-like row-burst aggressor at each share", "logN batch | Cheddar ms | four-step ms | alone | " + " | ".join(f"@{s:.1%}" for s in CARD["shares"])]
for lg in LG:
    c, f = one(lg, "cheddar"), one(lg, "f4c_sm")
    if not c or not f: continue
    tc, tf = float(c["victim_ms"]), float(f["victim_ms"]); rc, rf = curve(lg, "cheddar", "rowburst"), curve(lg, "f4c_sm", "rowburst")
    out.append(f"  {lg} {c['batch']:>3s} | {tc:.4f} | {tf:.4f} | {1 - tf / tc:+.1%} | "
               + " | ".join(f"{1 - tf * interp(rf, s) / (tc * interp(rc, s)):+.1%}" for s in CARD["shares"]))
txt = "\n".join(out) + "\n"; open(f"{RES}/cheddar_interfere_{TAG}.txt", "w").write(txt); print(txt)
