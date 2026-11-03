#!/usr/bin/env python3
"""Summary of cheddar_interfere_ole_<tag>.csv (silicon): the PCG-output NTT (ole: natural layout, ole_t: the SPUs' column-
contiguous layout) beside the SPU-like row-burst and the streaming aggressor, per logN at batch 16 (c = 4), ms per g_ij;
net slowdown at the SPUs' share as interfere_summary.py. Beside it Cheddar's old algorithm from cheddar_interfere_cfg_<tag>.csv
(if present) or cheddar_interfere_<tag>.csv: 2 multiplies per g_ij at Cheddar's best co-run config.
Usage: interfere_ole_summary.py TAG"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1]
RES = os.path.join(HERE, "..", f"results_{TAG}")
CARD = {"l40s": dict(peak=864.0, shares=(0.15,)), "b200": dict(peak=8000.0, shares=(0.127, 0.164, 0.205))}[TAG]
rows = list(csv.DictReader(open(f"{RES}/cheddar_interfere_ole_{TAG}.csv")))
chp = f"{RES}/cheddar_interfere_cfg_{TAG}.csv"
ch = [r for r in csv.DictReader(open(chp if os.path.exists(chp) else f"{RES}/cheddar_interfere_{TAG}.csv")) if r["victim"] == "cheddar"]
bad = sum(r["mismatches"] != "0" for r in rows)
cfg = lambda r: (r["P1"], r["M1"], r["B1"], r["M2"], r["maxrreg"])
cname = lambda c: f"{c[0]}+ M{c[1]} B{c[2]} M2 {c[3]} R{c[4]}"


def interp(pts, x):
    pts = [(0.0, 1.0)] + pts
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return float("nan")


def runs(R, lg, vic):
    out = {}
    for c in {cfg(r) for r in R if r["logN"] == str(lg) and r["victim"] == vic}:
        Q = [r for r in R if r["logN"] == str(lg) and r["victim"] == vic and cfg(r) == c]
        ctrl = {r["blocks"]: float(r["victim_ms"]) for r in Q if r["mode"] == "sleep_only"}
        cur = {m: sorted((float(r["agg_GBps"]) / CARD["peak"], float(r["victim_ms"]) / ctrl[r["blocks"]]) for r in Q if r["mode"] == m and r["blocks"] in ctrl)
               for m in ("rowburst", "stream")}
        alone = [r for r in Q if r["mode"] == "none"]
        if alone and cur["rowburst"]: out[c] = (float(alone[0]["victim_ms"]), cur)
    return out


SH = CARD["shares"]
out = [f"{TAG}: PCG-output NTT under DRAM interference, ms per g_ij (batch 16 = c^2, c = 4); {bad} of {len(rows)} rows have mismatches",
       "net slowdown = beside aggressor / beside sleep-only control (same blocks), at the SPUs' share of "
       f"{CARD['peak']:.0f} GB/s: " + ", ".join(f"{s:.1%}" for s in SH),
       "Cheddar (old algorithm) = 2 multiplies per g_ij at its best co-run config (row burst) of the config sweep", ""]
for s in SH:
    out += [f"share {s:.1%}:", "logN | design  config                   | alone ms | row burst | stream | co-run ms | 2 x Cheddar co-run ms (config) | gain"]
    for lg in sorted({int(r["logN"]) for r in rows}):
        C = runs(ch, lg, "cheddar")
        if C:
            cb = min(C, key=lambda c: C[c][0] * interp(C[c][1]["rowburst"], s)); cco = 2 * C[cb][0] * interp(C[cb][1]["rowburst"], s)
        for vic in ("ole", "ole_t"):
            for c, (t, cur) in sorted(runs(rows, lg, vic).items(), key=lambda kv: kv[1][0]):
                rb, st = interp(cur["rowburst"], s), interp(cur["stream"], s); co = t * rb
                out.append(f"  {lg} | {vic:6s} {cname(c):24s} | {t:8.4f} | {rb:7.3f} | {st:6.3f} | {co:9.4f} | "
                           + (f"{cco:.4f} ({cname(cb)}) | {cco / co:.2f}x" if C else "-"))
    out.append("")
txt = "\n".join(out) + "\n"; open(f"{RES}/cheddar_interfere_ole_{TAG}.txt", "w").write(txt); print(txt)
