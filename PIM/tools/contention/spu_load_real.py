#!/usr/bin/env python3
"""Per-SPU busy time and per-DRU transpose volume at the REAL card configuration: L40S 24 channels x 8 SPUs = 192 SPUs,
B200 one die-domain = 128 pseudo-channels x 8 SPUs = 1024 SPUs (the card is two symmetric die-domains, so the cell's
instances are halved). Instances per cell = c^2 t^2 (static partition, the generator's own mapping
m -> channel m % C, SPU (m // C) % 8, round m // (8C)); window depth 10 (1024 leaves/instance). Alone and co-run with the
cell's GPU NTT stream (final design: B200 four-step + DRU, L40S merge), sized as in e2e_window.py.
Usage: spu_load_real.py --sim BIN --out DIR [--jobs N]"""
import argparse, concurrent.futures as cf, json, math, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ntt_jobs import DEV, mul_ms, phases, write_jobs
PIM = os.path.dirname(os.path.dirname(HERE)); GEN = os.path.join(PIM, "tools", "gen_dpf_tree_trace.py")
YAML = {"l40s": os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml"), "b200": os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml")}
CH = {"l40s": 24, "b200": 128}; HALF = {"l40s": 1, "b200": 2}; P = 8; DEPTH = 10; KERNEL_MIN_CK = 20000
DES = {"l40s": "merge", "b200": "f4dru"}
CELLS = {"l40s": [(8, 4, 22), (4, 16, 22), (2, 128, 22), (8, 8, 24)], "b200": [(8, 4, 24), (4, 16, 24), (2, 128, 24), (8, 8, 24)]}
CTRL = ["-p", "MemorySystem.Controller.fpu_gate_issue=false", "-p", "MemorySystem.Controller.wr_max_age=1000", "-p", "MemorySystem.Controller.pim_row_wait=0",
        "-p", "MemorySystem.Controller.class_priority=0,1,3,2", "-p", "MemorySystem.Controller.class_min_run=64", "-p", "Frontend.issue_width=2",
        "-p", "MemorySystem.Controller.dru_bus_slot=2"]
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=12)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
parse = lambda p: {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}
eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
E2E = {(r["org"], r["c"], r["t"], r["logN"], r["design"]): r for r in json.load(open(os.path.join(PIM, "results/contention/final_window/win22/e2e_nom.json"))) if r["tier"] == "fast"}

def trace(org, c, t):
    I = c * c * t * t // HALF[org]; tr = os.path.join(a.out, f"spu_{org}_{c}_{t}.trace")
    if not os.path.exists(tr):
        subprocess.run([sys.executable, GEN, "-n", str(DEPTH), "-C", str(CH[org]), "-P", str(P), "-I", str(I), "--mode", "instances", "--seed-bits", "128",
                        "--reread", "--broadcast", "--cl", "155", "--reduce", "chacha", "--convert-cl", "465", "--modmul-cl", "32", "--add64m-cl", "4", "-o", tr],
                       check=True, stderr=subprocess.DEVNULL)
    return tr, I
def run(tag, org, tr, jp=None):
    out = os.path.join(a.out, tag + ".out")
    if not (os.path.exists(out) and "pim_done_cycles" in open(out).read()):
        cmd = [a.sim, "-f", YAML[org], "-t", tr] + ([] if jp is None else ["-p", f"MemorySystem.host_jobs={jp}"]) + CTRL
        with open(out, "w") as fo: subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT)
    return parse(out)
def alone(w):
    org, c, t, lg = w; tr, I = trace(org, c, t); return w, (tr, I, run(f"spu_{org}_{c}_{t}_alone", org, tr))
work = [(org, c, t, lg) for org in CELLS for (c, t, lg) in CELLS[org]]
with cf.ThreadPoolExecutor(a.jobs) as ex: A = dict(ex.map(alone, work))
def corun(w):
    org, c, t, lg = w; tr, I, al = A[w]; TS = al["pim_done_cycles"]; d = DES[org]; r0 = E2E[(org, c, t, lg, d)]
    tm = mul_ms(org, d, lg, c * c); mul_ck = tm * 1e6 / DEV[org]["tck"]; r = tm * 2 * c * c / r0["spu_ms"]
    n = max(1, int(r * TS / (10 * KERNEL_MIN_CK))); s = r * TS / (n * mul_ck)
    ph = phases(org, d, lg, c * c, scale=s)
    m = max(1, min(t, (c * c * t * t) // (8 * DEV[org]["ch"])))                     # DPF->NTT gather with the SPU presum, as in the main run
    gb = int(round(8 * (2 * (1 << lg) * t) / 2 / m / DEV[org]["ch"] * s)); ph = [(0, 1, gb, 1.0, 48, 0, 16, 20000, 0)] + ph
    jp = os.path.join(a.out, f"{org}_{c}_{t}_{lg}_{d}.jobs"); write_jobs(jp, [(n, 0, ph)])
    return w, run(f"{org}_{c}_{t}_{lg}_{d}_corun", org, tr, jp)
with cf.ThreadPoolExecutor(a.jobs) as ex: Cr = dict(ex.map(corun, work))
L = []
for w in work:
    org, c, t, lg = w; tr, I, al = A[w]; co = Cr[w]; C = CH[org]; tck = DEV[org]["tck"]; nspu = C * P
    cnt = [0] * nspu
    for m in range(I): cnt[(m % C) + C * ((m // C) % P)] += 1          # SPU index = ch + C*pu
    chmax = [max(cnt[ch + C * pu] for pu in range(P)) for ch in range(C)]
    for name, d in (("alone", al), ("co-run", co)):
        busy_ch = [d.get(f"CH{ch}_fpu_busy_cycles", 0) * tck / 1e6 for ch in range(C)]
        done_ch = [d.get(f"CH{ch}_pim_last_depart", 0) * tck / 1e6 for ch in range(C)]
        busy_spu = [busy_ch[i % C] * (cnt[i] / chmax[i % C] if chmax[i % C] else 0) for i in range(nspu)]
        mean = sum(busy_spu) / nspu; var = sum((x - mean) ** 2 for x in busy_spu) / nspu
        dm = sum(done_ch) / C; dv = sum((x - dm) ** 2 for x in done_ch) / C
        L.append(f"{org.upper()} ({c},{t}) 2^{lg} {name:6s}: {nspu} SPUs x{HALF[org]}, {I*HALF[org]} instances | per-SPU busy (ms) mean {mean:.3f} var {var:.5f} std {var**.5:.4f} min {min(busy_spu):.3f} max {max(busy_spu):.3f} max/mean {max(busy_spu)/mean if mean else 0:.3f} idle SPUs {sum(1 for x in cnt if x == 0)} | per-channel completion (ms) mean {dm:.3f} std {dv**.5:.4f} min {min(done_ch):.3f} max {max(done_ch):.3f} | SPU done {d['pim_done_cycles']*tck/1e6:.3f} ms"
                 + (f" | slowdown vs alone {co['pim_done_cycles']/al['pim_done_cycles']:.3f}" if name == "co-run" else ""))
    if org == "b200":
        dru = [max(0, co.get(f"CH{ch}_num_RD_commands", 0) + co.get(f"CH{ch}_num_WR_commands", 0) - co.get(f"CH{ch}_gpu_rd_cols", 0) - co.get(f"CH{ch}_gpu_wr_cols", 0)) for ch in range(C)]
        m = sum(dru) / C; L.append(f"   DRU columns per DRU over {C} DRUs: mean {m:.0f} std {(sum((x-m)**2 for x in dru)/C)**.5:.1f} min {min(dru)} max {max(dru)}")
open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
