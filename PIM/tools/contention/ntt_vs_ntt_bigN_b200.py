#!/usr/bin/env python3
"""DRU design (four-step, transposes executed by the DRU) against the GPU-NTT merge kernel, NTT lane alone,
logN 24..28 on the L40S memory system (native 24 channels and a 96-channel variant), 16 multiplies pipelined.
Per-multiply kernel times are silicon at every N (GPU_baseline/results_l40s/fused4_bigN_l40s.csv, batch 1, bit-exact).
Usage: ntt_vs_ntt_bigN.py --sim BIN --out DIR"""
import argparse, concurrent.futures as cf, csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ntt_jobs
from ntt_jobs import DEV, write_jobs
PIM = os.path.dirname(os.path.dirname(HERE)); ROOT = os.path.dirname(PIM)
YAML = os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml")
CTRL = ["-p", "MemorySystem.Controller.fpu_gate_issue=false", "-p", "MemorySystem.Controller.wr_max_age=1000", "-p", "MemorySystem.Controller.pim_row_wait=0",
        "-p", "MemorySystem.Controller.class_priority=0,1,3,2", "-p", "MemorySystem.Controller.class_min_run=64", "-p", "MemorySystem.DRAM.org.channel=2",
        "-p", "Frontend.issue_width=2", "-p", "MemorySystem.Controller.dru_bus_slot=2"]
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--muls", type=int, default=16); ap.add_argument("--kernel-ck", type=int, default=40000); ap.add_argument("--jobs", type=int, default=40)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
parse = lambda p: {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}
# B200 silicon, batch 1, min of 3 reps (AICR big-N job; values as reported by the job, not yet in the repo CSV):
# merge_ms_mul, f4g_sm (GPU-NTT sub-transform kernels), f4_sm (own single-kernel sub-transforms, 2^14 tiles)
B = {24: (2.4433, 2.3126, 3.1722), 25: (5.0387, 4.7639, 6.1119), 26: (10.4447, 9.8094, 11.8502),
     27: (22.0121, 20.7046, 23.2100), 28: (49.3586, 42.5815, 45.5454)}
TAB = {}
for lg, (m, gs, os_) in B.items():
    TAB[("headline", lg)] = (m, gs, gs)
    TAB[("own", lg)] = (m, os_, os_)
orig = ntt_jobs.ntt_table
SCALE = {}
def prep(w):   # serial: ntt_table is monkeypatched per job
    var, dev, lg, d = w; tag = f"{var}_{dev}_{lg}_{d}"; jp = os.path.join(a.out, tag + ".jobs")
    ntt_jobs.ntt_table = lambda _dev: {(lg, 4): TAB[(var, lg)]}
    tck = DEV[dev]["tck"]; tm = TAB[(var, lg)][0]
    nph = len([p for p in ntt_jobs.phases(dev, "merge", lg, 4) if p[0] == 0]); s = a.kernel_ck * nph / (tm * 1e6 / tck)
    write_jobs(jp, [(a.muls, 0, ntt_jobs.phases(dev, d, lg, 4, scale=s))]); SCALE[w] = s
def run(w):
    var, dev, lg, d = w; tag = f"{var}_{dev}_{lg}_{d}"; jp = os.path.join(a.out, tag + ".jobs"); out = os.path.join(a.out, tag + ".out"); s = SCALE[w]
    if not (os.path.exists(out) and "pipe0_finish_max" in open(out).read()):
        with open(out, "w") as f: subprocess.run([a.sim, "-f", YAML, "-t", eoc, "-p", f"MemorySystem.host_jobs={jp}", "-p", f"MemorySystem.DRAM.org.channel=2"] + CTRL, stdout=f, stderr=subprocess.STDOUT)
    r = parse(out); return (var, dev, lg, d), (r["pipe0_finish_max"], s, TAB[(var, lg)])
work = [(v, dev, lg, d) for v in ("headline", "own") for dev in ("b200",) for lg in range(24, 29) if (v, lg) in TAB for d in ("merge", "f4dru")]
for w in work: prep(w)
with cf.ThreadPoolExecutor(a.jobs) as ex: R = dict(ex.map(run, work))
lines = ["DRU design vs GPU-NTT merge, NTT lane alone, B200 HBM3e memory system (256 channels), 16 multiplies pipelined. Ratio > 1 = DRU design faster.",
         "silicon inputs per multiply (ms, batch 1, B200 AICR big-N job): merge, four-step SM lane (headline = GPU-NTT kernels, own = single-kernel sub-transforms).",
         f"{'variant':>8s} {'org':>10s} logN | merge ms  SM-lane ms | sim merge/DRU | sim merge vs silicon | DRU lane vs its SM lane"]
for (var, dev, lg, d), (fin, s, t) in sorted(R.items()):
    if d != "merge": continue
    fd = R[(var, dev, lg, "f4dru")][0]; tck = DEV[dev]["tck"]
    meas_m = t[0] * 1e6 / tck * a.muls * s; meas_f = t[1] * 1e6 / tck * a.muls * s
    lines.append(f"{var:>8s} {dev:>10s} {lg:4d} | {t[0]:8.3f} {t[1]:11.3f} | {fin/fd:13.3f} | {fin/meas_m:20.3f} | {fd/meas_f:22.3f}")
open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
