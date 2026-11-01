#!/usr/bin/env python3
"""Silicon vs Ramulator2 NTT slowdown under the synthetic interference kernel, evaluated at the interference level that
matches the SPUs' column-command occupancy of each card, for the designs of the final accounting: Cheddar's NTT and the
SM lanes of the two DRU designs (f4c_sm, f4b_sm), from GPU_baseline/cheddar/cheddar_interfere_<tag>.csv (logN 22, batch 16,
each design at its best measured config). Silicon slowdown is normalized by the same kernel beside sleeping aggressor blocks
(removes the SM-occupancy part); the simulated curve uses each design's fitted access pattern (ntt_jobs.PATTERN).
Usage: interfere_at_spu.py --sim BIN --out DIR"""
import argparse, concurrent.futures as cf, csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ntt_jobs import DEV, mul_ms, phases, write_jobs
PIM = os.path.dirname(os.path.dirname(HERE)); REPO = os.path.dirname(PIM)
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
CARD = {"l40s": dict(yaml="sim/test/gddr6_dpf_24ch.yaml", peak=864.0, csv="GPU_baseline/results_l40s/cheddar_interfere_l40s.csv", spu=(0.10, 0.15)),
        "b200": dict(yaml="sim/test/hbm3e_dpf_b200d.yaml", peak=8000.0, csv="GPU_baseline/results_b200/cheddar_interfere_b200.csv", spu=(0.13, 0.21))}
DESIGNS = (("cheddar", "cheddar"), ("f4c", "f4c_sm"), ("f4b", "f4b_sm"))   # (co-sim design, silicon victim)
GAPS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
def sim(org, args, out):
    if not (os.path.exists(out) and "memory_system_cycles" in open(out).read()):
        with open(out, "w") as f:
            subprocess.run([a.sim, "-f", os.path.join(PIM, CARD[org]["yaml"]), "-t", eoc, "-p", "MemorySystem.DRAM.org.channel=2", "-p", "Frontend.issue_width=2",
                            "-p", "MemorySystem.Controller.wr_max_age=1000", "-p", "MemorySystem.Controller.class_priority=1,3,0,2"] + args, stdout=f, stderr=subprocess.STDOUT)
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(out).read(), re.M)}
def curve(org, design):
    lg = 22; target = mul_ms(org, design, lg, 16)
    muls = 8; sc = 1.5e6 / (muls * target * 1e6 / DEV[org]["tck"])
    ph = [p for p in phases(org, design, lg, 16, scale=sc) if p[0] == 0]      # the SM lane (silicon times the SM kernels)
    jp = os.path.join(a.out, f"{org}_{design}.jobs"); write_jobs(jp, [(muls, 0, ph)])
    base = sim(org, ["-p", f"MemorySystem.host_jobs={jp}"], os.path.join(a.out, f"{org}_{design}_alone.out"))["pipe0_finish_max"]
    def one(g):
        sp = os.path.join(a.out, f"{org}_{design}_g{g}.streams"); open(sp, "w").write(f"3 40000000 0.5 128 {g} 32 40000 0\n")
        d = sim(org, ["-p", f"MemorySystem.host_jobs={jp}", "-p", f"MemorySystem.host_streams={sp}"], os.path.join(a.out, f"{org}_{design}_g{g}.out"))
        t = d["pipe0_finish_max"]; return (d["stream0_cols_at_p0_done"] * 32 / t / 16.0, t / base)
    with cf.ThreadPoolExecutor(len(GAPS)) as ex: return sorted(ex.map(one, GAPS))
def interp(pts, x):
    if x <= pts[0][0]: return 1 + (pts[0][1] - 1) * x / pts[0][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]
for org, c in CARD.items():
    p = os.path.join(REPO, c["csv"])
    if not os.path.exists(p): print(f"{org.upper()}: {c['csv']} missing"); continue
    rows = [r for r in csv.DictReader(open(p)) if r["logN"] == "22" and r["batch"] == "16"]
    for design, vic in DESIGNS:
        V = [r for r in rows if r["victim"] == vic]
        if not V: print(f"{org.upper()} {design}: no {vic} rows at 2^22 batch 16"); continue
        alone = min(V, key=lambda r: float(r["victim_ms"]) if r["mode"] == "none" else 1e9)      # the best config alone
        cfg = (alone["P1"], alone["M1"], alone["B1"], alone["M2"], alone["maxrreg"])
        V = [r for r in V if (r["P1"], r["M1"], r["B1"], r["M2"], r["maxrreg"]) == cfg]
        ctrl = {r["blocks"]: float(r["victim_ms"]) for r in V if r["mode"] == "sleep_only"}
        sil = sorted((float(r["agg_GBps"]) / c["peak"], float(r["victim_ms"]) / ctrl[r["blocks"]]) for r in V
                     if r["mode"] == "stream" and r["blocks"] in ctrl and float(r["agg_GBps"]) / c["peak"] <= 0.25)
        if not sil: print(org, design, "no silicon points"); continue
        try: pts = curve(org, design)
        except KeyError as e: print(f"{org.upper()} {design}: no co-sim table entry ({e})"); continue
        lo, hi = c["spu"]
        near = [(x, y, interp(pts, x)) for x, y in sil if lo - 0.03 <= x <= hi + 0.03]
        allerr = [(interp(pts, x) - y) for x, y in sil]
        print(f"{org.upper()} {design} (silicon {vic}, config P1 {cfg[0]} M1 {cfg[1]} B1 {cfg[2]} M2 {cfg[3]} R{cfg[4]}): SPU column-command occupancy {lo:.0%}-{hi:.0%}; silicon points in [{lo-0.03:.0%},{hi+0.03:.0%}]:")
        for x, y, s in near: print(f"   occupancy {x:5.1%}: silicon {y:.3f}  sim {s:.3f}  error {s-y:+.3f} ({(s-y)/y*100:+.1f}%)")
        if near: print(f"   at the SPU's level: mean |error| {sum(abs(s-y) for x,y,s in near)/len(near):.3f}; over all points <=25%: rms {(sum(e*e for e in allerr)/len(allerr))**0.5:.3f}")
