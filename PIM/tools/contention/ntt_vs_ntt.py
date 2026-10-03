#!/usr/bin/env python3
"""NTT against NTT on B200: the DRU design (four-step on GPU-NTT kernels, transposes on the DRU)
against the SOTA GPU-NTT merge kernel, per (logN, batch), with a DEEP pipeline (16 multiplies, every
kernel sized to ~40k CK) so the DRU transposes pipeline as they do in a real batch -- alone, and next
to the SPUs (SPU trace long enough to cover the whole NTT pipeline).
Usage: ntt_vs_ntt.py --sim BIN --out DIR [--muls 16] [--kernel-ck 40000] [--win-inst 40]"""
import argparse, concurrent.futures as cf, os, re, subprocess, sys
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ntt_jobs import DEV, ntt_table, phases, write_jobs
PIM = os.path.dirname(os.path.dirname(HERE))
YAML = os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml"); GEN = os.path.join(PIM, "tools", "gen_dpf_tree_trace.py")
CTRL = ["-p", "MemorySystem.Controller.fpu_gate_issue=false", "-p", "MemorySystem.Controller.wr_max_age=1000", "-p", "MemorySystem.Controller.pim_row_wait=0",
        "-p", "MemorySystem.Controller.class_priority=0,1,3,2", "-p", "MemorySystem.Controller.class_min_run=64", "-p", "MemorySystem.DRAM.org.channel=2",
        "-p", "Frontend.issue_width=2", "-p", "MemorySystem.Controller.dru_bus_slot=2"]
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--muls", type=int, default=16); ap.add_argument("--kernel-ck", type=int, default=40000); ap.add_argument("--win-inst", type=int, default=40)
ap.add_argument("--jobs", type=int, default=60)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
def gm(v): v = list(v); return exp(sum(map(log, v)) / len(v))
def parse(p): return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}
eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
spu = os.path.join(a.out, f"spu_b200_i{a.win_inst}.trace")
if not os.path.exists(spu):
    subprocess.run([sys.executable, GEN, "-n", "12", "-C", "2", "-P", "8", "-I", str(16 * a.win_inst), "--mode", "instances", "--seed-bits", "128", "--reread",
                    "--broadcast", "--cl", "155", "--reduce", "chacha", "--convert-cl", "465", "--modmul-cl", "32", "-o", spu], check=True, stderr=subprocess.DEVNULL)
T = ntt_table("b200"); tck = DEV["b200"]["tck"]
work = []
for (lg, b) in sorted(k for k in T if k[0] >= 20):
    s = a.kernel_ck * 10 / (T[(lg, b)][0] * 1e6 / tck)          # same scale for both designs: merge's kernels ~kernel_ck
    for d in ("merge", "f4dru"):
        jp = os.path.join(a.out, f"b200_{lg}_{b}_{d}.jobs"); write_jobs(jp, [(a.muls, 0, phases("b200", d, lg, b, scale=s))])
        for tag, tr in (("alone", eoc), ("spu", spu)):
            work.append((lg, b, d, tag, jp, tr, s))
def run(w):
    lg, b, d, tag, jp, tr, s = w; out = jp[:-5] + f"_{tag}.out"
    if not (os.path.exists(out) and "pipe0_finish_max" in open(out).read()):
        with open(out, "w") as f: subprocess.run([a.sim, "-f", YAML, "-t", tr, "-p", f"MemorySystem.host_jobs={jp}"] + CTRL, stdout=f, stderr=subprocess.STDOUT)
    r = parse(out); return (lg, b, d, tag), (r["pipe0_finish_max"], r.get("pim_done_cycles", -1), s)
with cf.ThreadPoolExecutor(a.jobs) as ex: R = dict(ex.map(run, work))
lines = [f"B200 NTT vs NTT: {a.muls} multiplies pipelined, kernels ~{a.kernel_ck} CK, SPU trace {a.win_inst} instances/SPU. Ratios > 1 = DRU design faster.",
         f"{'logN/batch':>10s} | {'merge sim/meas':>14s} {'DRU sim/meas':>12s} | {'alone: merge/DRU':>16s} | {'slowdown merge':>14s} {'slowdown DRU':>12s} | {'next to SPUs: merge/DRU':>23s} | {'vs merge alone (GPU baseline)':>29s} | SPU covered"]
A, B, C = [], [], []
for (lg, b) in sorted(k for k in T if k[0] >= 20):
    ma, ms = R[(lg, b, "merge", "alone")], R[(lg, b, "merge", "spu")]; fa, fs = R[(lg, b, "f4dru", "alone")], R[(lg, b, "f4dru", "spu")]
    s = ma[2]; meas_m = T[(lg, b)][0] * 1e6 / tck * a.muls * s; meas_f = T[(lg, b)][1] * 1e6 / tck * a.muls * s
    sm, sf = ms[0] / ma[0], fs[0] / fa[0]
    r_alone, r_spu, r_base = ma[0] / fa[0], ms[0] / fs[0], ma[0] / fs[0]
    cov = "yes" if min(ms[1], fs[1]) >= max(ms[0], fs[0]) else f"NO ({min(ms[1], fs[1])/max(ms[0], fs[0]):.2f})"
    A.append(r_alone); B.append(r_spu); C.append(r_base)
    lines.append(f"{lg:>6d}/{b:<3d} | {ma[0]/meas_m:14.3f} {fa[0]/meas_f:12.3f} | {r_alone:16.3f} | {sm:14.3f} {sf:12.3f} | {r_spu:23.3f} | {r_base:29.3f} | {cov}")
lines.append(f"geomean: alone {gm(A):.3f} (range {min(A):.2f}-{max(A):.2f}) | next to SPUs {gm(B):.3f} ({min(B):.2f}-{max(B):.2f}) | DRU next to SPUs vs merge alone {gm(C):.3f} ({min(C):.2f}-{max(C):.2f})")
open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
