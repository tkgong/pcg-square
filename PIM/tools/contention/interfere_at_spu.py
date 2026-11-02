#!/usr/bin/env python3
"""Silicon vs Ramulator2 NTT slowdown under the synthetic interference kernel, evaluated at the interference level that
matches the SPUs' column-command occupancy of each card (final access pattern W=48, R=16).
Silicon slowdown is normalized by the same kernel with the interference blocks only sleeping (removes the SM-occupancy part).
Usage: interfere_at_spu.py --sim BIN --out DIR"""
import argparse, concurrent.futures as cf, csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ntt_jobs import DEV, ntt_table, phases, write_jobs
PIM = os.path.dirname(os.path.dirname(HERE)); REPO = os.path.dirname(PIM)
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
CARD = {"l40s": dict(yaml="sim/test/gddr6_dpf_24ch.yaml", peak=864.0, csv="GPU_baseline/fused4/results/interfere_ntt_L40S_22.csv", spu=(0.10, 0.15)),
        "b200": dict(yaml="sim/test/hbm3e_dpf_b200d.yaml", peak=8000.0, csv="GPU_baseline/fused4/results/interfere_ntt_B200_22.csv", spu=(0.16, 0.21))}
GAPS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
def sim(org, args, out):
    if not (os.path.exists(out) and "memory_system_cycles" in open(out).read()):
        with open(out, "w") as f:
            subprocess.run([a.sim, "-f", os.path.join(PIM, CARD[org]["yaml"]), "-t", eoc, "-p", "MemorySystem.DRAM.org.channel=2", "-p", "Frontend.issue_width=2",
                            "-p", "MemorySystem.Controller.wr_max_age=1000", "-p", "MemorySystem.Controller.class_priority=1,3,0,2"] + args, stdout=f, stderr=subprocess.STDOUT)
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(out).read(), re.M)}
def curve(org, design):
    lg = 22; tm, ts = ntt_table(org)[(lg, 16)][:2]; target = tm if design == "merge" else ts
    muls = 8; sc = 1.5e6 / (muls * target * 1e6 / DEV[org]["tck"])
    ph = phases(org, design, lg, 16, scale=sc, window=48, run=16)
    if design == "f4dru": ph = [p for p in ph if p[0] == 0]
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
    rows = list(csv.DictReader(open(os.path.join(REPO, c["csv"]))))
    for design, vic in (("merge", "merge"), ("f4dru", "f4g_sm")):
        ctrl = {r["blocks"]: float(r["victim_ms"]) for r in rows if r["mode"] == "sleep_only" and r["victim"] == vic}
        none = [float(r["victim_ms"]) for r in rows if r["mode"] == "none" and r["victim"] == vic]
        sil = sorted((float(r["agg_GBps"]) / c["peak"], float(r["victim_ms"]) / ctrl[r["blocks"]]) for r in rows
                     if r["mode"] == "stream" and r["victim"] == vic and r["blocks"] in ctrl and float(r["agg_GBps"]) / c["peak"] <= 0.25)
        if not sil: print(org, design, "no silicon points"); continue
        pts = curve(org, design); lo, hi = c["spu"]
        near = [(x, y, interp(pts, x)) for x, y in sil if lo - 0.03 <= x <= hi + 0.03]
        allerr = [(interp(pts, x) - y) for x, y in sil]
        print(f"{org.upper()} {design}: SPU column-command occupancy {lo:.0%}-{hi:.0%}; silicon points in [{lo-0.03:.0%},{hi+0.03:.0%}]:")
        for x, y, s in near: print(f"   occupancy {x:5.1%}: silicon {y:.3f}  sim {s:.3f}  error {s-y:+.3f} ({(s-y)/y*100:+.1f}%)")
        if near: print(f"   at the SPU's level: mean |error| {sum(abs(s-y) for x,y,s in near)/len(near):.3f}; over all points <=25%: rms {(sum(e*e for e in allerr)/len(allerr))**0.5:.3f}")
