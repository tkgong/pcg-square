#!/usr/bin/env python3
"""Q4, second half: what each contention-mitigation mechanism of the controller buys, on the final co-sim
window (e2e_window.py run dir). For a cell, co-run the SPU trace with the NTT pipeline under the final
arbitration and under each alternative, and report the SPU and NTT slowdowns (vs alone) and the GPU
queueing p99.  Usage: arbitration_ablation.py --sim BIN --run WIN22_DIR --out DIR [--cells b200:4_16_24:f4dru,l40s:4_16_22:merge]"""
import argparse, concurrent.futures as cf, glob, os, re, subprocess
PIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YAML = {"l40s": os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml"), "b200": os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml")}
TCK = {"l40s": 0.444, "b200": 0.5}
BASE = ["-p", "MemorySystem.Controller.fpu_gate_issue=false", "-p", "MemorySystem.DRAM.org.channel=2", "-p", "Frontend.issue_width=2", "-p", "MemorySystem.Controller.dru_bus_slot=2"]
FINAL = dict(wr_max_age=1000, pim_row_wait=0, class_priority="0,1,3,2", class_min_run=64, pim_strict="false")
POLICIES = {
    "final: class-fair, GPU 64-cycle burst window, PIM row protection, age write drain": {},
    "strict PIM priority": dict(pim_strict="true"),
    "strict GPU priority": dict(class_priority="1,3,2,0", class_min_run=0, pim_row_wait=-1),
    "final without the GPU burst window (class_min_run=0)": dict(class_min_run=0),
    "final without PIM row protection": dict(pim_row_wait=-1),
    "final without the age-triggered write drain": dict(wr_max_age=0),
}
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cells", default="b200:4_16_24:f4dru,b200:4_16_24:merge,l40s:4_16_22:merge"); ap.add_argument("--jobs", type=int, default=24)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
if a.cells == "all":   # every cell of the suite with the final design
    import json
    a.cells = ",".join(sorted({f'{r["org"]}:{r["c"]}_{r["t"]}_{r["logN"]}:{"f4dru" if r["org"] == "b200" else "merge"}' for r in json.load(open(os.path.join(a.run, "e2e.json"))) if r["tier"] == "fast"}))
def parse(p): return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}
work = []
for cell in a.cells.split(","):
    org, c, des = cell.split(":")
    for name, ov in POLICIES.items():
        prm = dict(FINAL, **ov); tag = re.sub(r"[^a-z0-9]+", "_", name.lower())[:40]
        out = os.path.join(a.out, f"{org}_{c}_{des}_{tag}.out")
        args = [a.sim, "-f", YAML[org], "-t", sorted(glob.glob(os.path.join(a.run, f"spu_{org}_*clk.trace")))[-1], "-p", f"MemorySystem.host_jobs={os.path.join(a.run, f'{org}_{c}_{des}.jobs')}"] + BASE + [x for k, v in prm.items() for x in ("-p", f"MemorySystem.Controller.{k}={v}")]
        work.append((org, c, des, name, out, args))
def run(w):
    org, c, des, name, out, args = w
    if not (os.path.exists(out) and ("pim_done_cycles" in open(out).read() or "watchdog" in open(out).read())):
        with open(out, "w") as f: subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    return w, parse(out)
with cf.ThreadPoolExecutor(a.jobs) as ex: R = list(ex.map(run, work))
lines = []
for cell in a.cells.split(","):
    org, c, des = cell.split(":"); tck = TCK[org]
    spu = parse(sorted(glob.glob(os.path.join(a.run, f"spu_{org}_*clk.out")))[-1]); alone = parse(os.path.join(a.run, f"{org}_{c}_{des}_alone.out"))
    lines.append(f"\n{org.upper()} ({c.replace('_', ',', 1).replace('_', ') 2^')}, {des}: SPU slowdown | NTT slowdown | GPU queueing mean/p99 ns | GPU read latency p99 ns | SPU queueing mean ns")
    for (o, cc, d, name, out, args), r in R:
        if (o, cc, d) != (org, c, des): continue
        if "pim_done_cycles" not in r:
            lines.append(f"  {name:72s} LIVELOCK: the controller's watchdog fired (reads wait behind posted writes that never reach the drain watermark)"); continue
        lines.append(f"  {name:72s} SPU x{r['pim_done_cycles']/spu['pim_done_cycles']:.3f} | NTT x{r['pipe0_finish_max']/alone['pipe0_finish_max']:.3f} | "
                     f"{r.get('CH0_gpu_q_mean',0)*tck:5.0f}/{r.get('CH0_gpu_q_p99',0)*tck:6.0f} | {r.get('CH0_gpu_rdlat_p99',0)*tck:6.0f} | {r.get('CH0_pim_q_mean',0)*tck:6.0f}")
# per-policy geomean over all cells (SPU / NTT slowdown), plus the NTT-bound subset
from math import exp, log
def gm(v): v = list(v); return exp(sum(map(log, v)) / len(v)) if v else float("nan")
agg = {}
for (o, cc, d, name, out, args), r in R:
    if "pim_done_cycles" not in r: continue
    spu = parse(sorted(glob.glob(os.path.join(a.run, f"spu_{o}_*clk.out")))[-1]); alone = parse(os.path.join(a.run, f"{o}_{cc}_{d}_alone.out"))
    agg.setdefault((o, name), []).append((r["pim_done_cycles"] / spu["pim_done_cycles"], r["pipe0_finish_max"] / alone["pipe0_finish_max"]))
lines.append("\nGeomean over all cells, SPU slowdown / NTT slowdown (max over cells in brackets):")
for (o, name), v in sorted(agg.items()):
    lines.append(f"  {o.upper()} {name:72s} SPU x{gm(x for x, _ in v):.3f} [{max(x for x, _ in v):.3f}] | NTT x{gm(y for _, y in v):.3f} [{max(y for _, y in v):.3f}]  ({len(v)} cells)")
for (o, name) in [k for k in agg if k[1].startswith("final:")]:
    pass
open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
