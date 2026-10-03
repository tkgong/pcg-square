#!/usr/bin/env python3
"""Joint fit of an NTT's DRAM access pattern (reads in flight W, columns per row
visit R) to silicon on BOTH (1) standalone per-mul time (host_jobs kernels with
measured compute floors) and (2) the interference curve (calib_ntt_patterns.py).
Usage: fit_ntt_pattern.py --sim BIN --out DIR --silicon interfere_ntt_L40S_22.csv --design merge|f4dru"""
import argparse, concurrent.futures as cf, csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ntt_jobs import DEV, ntt_table, phases, write_jobs
PIM = os.path.dirname(os.path.dirname(HERE))
YAML = os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml")
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--silicon", required=True); ap.add_argument("--design", default="merge"); ap.add_argument("--jobs", type=int, default=80)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
VIC = {"merge": "merge", "f4dru": "f4g_sm"}[a.design]
RD = {"merge": 88 / 168, "f4dru": 88 / 144}[a.design]
rows = list(csv.DictReader(open(a.silicon)))
ctrl = {(r["victim"], r["blocks"]): float(r["victim_ms"]) for r in rows if r["mode"] == "sleep_only"}
SIL = sorted((float(r["agg_GBps"]) / 864.0, float(r["victim_ms"]) / ctrl[(r["victim"], r["blocks"])])
             for r in rows if r["mode"] == "stream" and r["victim"] == VIC and float(r["agg_GBps"]) / 864 <= 0.25)
def sim(args, out):
    if not (os.path.exists(out) and "memory_system_cycles" in open(out).read()):
        with open(out, "w") as f:
            subprocess.run([a.sim, "-f", YAML, "-t", eoc, "-p", "MemorySystem.DRAM.org.channel=2", "-p", "Frontend.issue_width=2",
                            "-p", "MemorySystem.Controller.wr_max_age=1000", "-p", "MemorySystem.Controller.class_priority=1,3,0,2"] + args,
                           stdout=f, stderr=subprocess.STDOUT)
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(out).read(), re.M)}
def standalone(W, R):   # per-mul ms at logN 22 / 24, batch 16 vs silicon
    err = []
    for lg in (22, 24):
        tm, ts = ntt_table("l40s")[(lg, 16)][:2]
        target = tm if a.design == "merge" else ts
        muls = 8; sc = 1.5e6 / (muls * target * 1e6 / DEV["l40s"]["tck"])
        ph = phases("l40s", a.design, lg, 16, scale=sc, window=W, run=R)
        if a.design == "f4dru": ph = [p for p in ph if p[0] == 0]      # SM lane only (L40S has no DRU)
        jp = os.path.join(a.out, f"sa_{W}_{R}_{lg}.jobs"); write_jobs(jp, [(muls, 0, ph)])
        d = sim(["-p", f"MemorySystem.host_jobs={jp}"], os.path.join(a.out, f"sa_{W}_{R}_{lg}.out"))
        err.append(d["pipe0_finish_max"] * DEV["l40s"]["tck"] / 1e6 / muls / sc / target - 1)
    return err
GAPS = [3, 4, 6, 8, 12, 16, 24, 32, 48]
def curve(W, R):
    """Victim = the same kernel pipeline (with compute floors) as the standalone check,
    next to the aggressor stream; slowdown vs its own standalone time, x = aggressor
    bandwidth while the victim ran."""
    lg = 22; tm, ts = ntt_table("l40s")[(lg, 16)][:2]; target = tm if a.design == "merge" else ts
    muls = 8; sc = 1.5e6 / (muls * target * 1e6 / DEV["l40s"]["tck"])
    ph = phases("l40s", a.design, lg, 16, scale=sc, window=W, run=R)
    if a.design == "f4dru": ph = [p for p in ph if p[0] == 0]
    jp = os.path.join(a.out, f"cv_{W}_{R}.jobs"); write_jobs(jp, [(muls, 0, ph)])
    base = sim(["-p", f"MemorySystem.host_jobs={jp}"], os.path.join(a.out, f"sa_{W}_{R}_{lg}.out"))["pipe0_finish_max"]
    pts = []
    for g in GAPS:
        sp = os.path.join(a.out, f"cv_{W}_{R}_g{g}.streams")
        open(sp, "w").write(f"3 40000000 0.5 128 {g} 32 40000 0\n")
        d = sim(["-p", f"MemorySystem.host_jobs={jp}", "-p", f"MemorySystem.host_streams={sp}"], os.path.join(a.out, f"cv_{W}_{R}_g{g}.out"))
        t = d["pipe0_finish_max"]; pts.append((d["stream0_cols_at_p0_done"] * 32 / t / 16.0, t / base))
    return sorted(pts)
def interp(pts, x):
    if x <= pts[0][0]: return 1 + (pts[0][1] - 1) * x / pts[0][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]
grid = [(W, R) for W in (32, 48, 64, 96, 128, 192, 256) for R in (4, 8, 16, 24, 32, 64)]
def ev(g):
    W, R = g; e = standalone(W, R); c = curve(W, R)
    rms = (sum((interp(c, x) - y) ** 2 for x, y in SIL) / len(SIL)) ** 0.5
    return W, R, e, rms
res = []
with cf.ThreadPoolExecutor(a.jobs) as ex:
    for r in ex.map(ev, grid): res.append(r)
res.sort(key=lambda r: max(abs(r[2][0]), abs(r[2][1])) * 2 + r[3])
print(f"{a.design}: silicon interference points {[(round(x,3), round(y,3)) for x, y in SIL]}")
for W, R, e, rms in res[:10]:
    print(f"  W={W:3d} R={R:2d}  standalone err logN22 {e[0]*100:+.1f}%  logN24 {e[1]*100:+.1f}%   interference rms {rms:.3f}")
