#!/usr/bin/env python3
"""Per-config effective DRAM-traffic fraction of each NTT's SM kernels: find f in (0, 1]
such that the simulated standalone per-mul time (kernels with measured compute floors,
fitted access pattern) equals silicon. f < 1 where the kernel's working set partly lives
in L2 (the simulator has no L2); f = 1 where the kernel is compute-bound. Writes
ntt_dram_fraction.json next to this script."""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ntt_jobs import DEV, ntt_table, phases, write_jobs
PIM = os.path.dirname(os.path.dirname(HERE))
YAML = {"l40s": os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml"), "b200": os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml")}
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--devs", default="l40s,b200"); ap.add_argument("--jobs", type=int, default=60)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
MULS = 8
def t_sim(dev, design, lg, b, f):
    tm, ts = ntt_table(dev)[(lg, b)]; target = tm if design == "merge" else ts
    sc = 1.5e6 / (MULS * target * 1e6 / DEV[dev]["tck"])
    ph = phases(dev, design, lg, b, scale=sc, frac=f)
    if design == "f4dru": ph = [p for p in ph if p[0] == 0]         # SM lane (silicon f4g_sm)
    tag = f"{dev}_{design}_{lg}_{b}_{f:.4f}"; jp = os.path.join(a.out, tag + ".jobs"); out = os.path.join(a.out, tag + ".out")
    write_jobs(jp, [(MULS, 0, ph)])
    if not (os.path.exists(out) and "pipe0_finish_max" in open(out).read()):
        with open(out, "w") as fo:
            subprocess.run([a.sim, "-f", YAML[dev], "-t", eoc, "-p", "MemorySystem.DRAM.org.channel=2", "-p", "Frontend.issue_width=2",
                            "-p", f"MemorySystem.host_jobs={jp}", "-p", "MemorySystem.Controller.wr_max_age=1000"], stdout=fo, stderr=subprocess.STDOUT)
    t = float(re.search(r"pipe0_finish_max:\s*([0-9.eE+]+)", open(out).read()).group(1))
    return t * DEV[dev]["tck"] / 1e6 / MULS / sc / target          # sim / silicon
def fit(key):
    dev, design, lg, b = key
    r1 = t_sim(dev, design, lg, b, 1.0)
    if r1 <= 1.003: return key, 1.0, r1, r1
    lo, hi = 0.0, 1.0                                               # bisection on f (time is monotone in f)
    for _ in range(9):
        mid = (lo + hi) / 2
        if t_sim(dev, design, lg, b, mid) > 1.003: hi = mid      # tolerance: last-phase latency, rounding
        else: lo = mid
    f = (lo + hi) / 2
    return key, round(f, 4), r1, t_sim(dev, design, lg, b, round(f, 4))
keys = [(d, g, lg, b) for d in a.devs.split(",") for g in ("merge", "f4dru") for (lg, b) in sorted(ntt_table(d))]
res = {}
with cf.ThreadPoolExecutor(a.jobs) as ex:
    for key, f, r1, rf in ex.map(fit, keys):
        res[key] = (f, r1, rf)
        print(f"{key}: sim/silicon at f=1 {r1:.3f} -> f={f:.3f} -> {rf:.3f}", flush=True)
json.dump({f"{d}/{g}/{lg}/{b}": v[0] for (d, g, lg, b), v in res.items()},
          open(os.path.join(HERE, "ntt_dram_fraction.json"), "w"), indent=1, sort_keys=True)
