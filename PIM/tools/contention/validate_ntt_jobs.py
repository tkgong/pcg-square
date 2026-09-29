#!/usr/bin/env python3
"""Standalone NTT in the simulator (no SPU) vs silicon: merge and f4+DRU per-mul time."""
import argparse, concurrent.futures as cf, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntt_jobs import DEV, ntt_table, phases, write_jobs
PIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YAML = {"l40s": os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml"), "b200": os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml")}
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dru-window", type=int, default=4); ap.add_argument("--muls", type=int, default=4)
ap.add_argument("--jobs", type=int, default=32)
ap.add_argument("--target-ck", type=float, default=1.5e6, help="scale each run to about this many CK")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
def run(dev, design, lg, b):
    tag = f"{dev}_{design}_{lg}_{b}"; jp = os.path.join(a.out, tag + ".jobs"); out = os.path.join(a.out, tag + ".out")
    tm = ntt_table(dev)[(lg, b)][0]
    sc = min(1.0, a.target_ck / (a.muls * tm * 1e6 / DEV[dev]["tck"]))   # linear model: shrink bytes+floors alike
    write_jobs(jp, [(a.muls, 0, phases(dev, design, lg, b, scale=sc, dru_window=a.dru_window))])
    cmd = [a.sim, "-f", YAML[dev], "-t", eoc, "-p", "MemorySystem.DRAM.org.channel=2", "-p", "Frontend.issue_width=2",
           "-p", f"MemorySystem.host_jobs={jp}", "-p", "MemorySystem.Controller.wr_max_age=1000"]
    with open(out, "w") as f: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    t = float(re.search(r"pipe0_finish_mean:\s*([0-9.eE+]+)", open(out).read()).group(1))
    return dev, design, lg, b, t * DEV[dev]["tck"] / 1e6 / a.muls / sc
J = [(d, g, lg, b) for d in ("l40s", "b200") for g in ("merge", "f4dru") for lg in range(20, 25) for b in (4, 16, 64)
     if (lg, b) in ntt_table(d)]
res = {}
with cf.ThreadPoolExecutor(a.jobs) as ex:
    for d, g, lg, b, ms in ex.map(lambda j: run(*j), J): res[(d, g, lg, b)] = ms
print("dev logN batch | silicon merge  f4g_sm | sim merge  f4dru | silicon merge/f4g_sm  sim merge/f4dru")
for d in ("l40s", "b200"):
    T = ntt_table(d)
    for (lg, b), (tm, ts) in sorted(T.items()):
        if (d, "merge", lg, b) not in res: continue
        sm, sf = res[(d, "merge", lg, b)], res[(d, "f4dru", lg, b)]
        print(f"{d} {lg} {b:3d} | {tm:7.3f} {ts:7.3f} | {sm:7.3f} {sf:7.3f} | {tm / ts:6.2f}x  {sm / sf:6.2f}x")
