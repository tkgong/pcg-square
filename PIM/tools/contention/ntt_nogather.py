#!/usr/bin/env python3
"""NTT lane alone with the DPF->NTT gather phase removed, for the channel-count sweep: re-simulates every *_alone NTT
pipeline (four-step + DRU, four-step on the SMs) of the given window directories without its first (gather) phase and
rescales the lane: lane_nogather = ntt_ms * finish(no gather) / finish(with gather).
Usage: ntt_nogather.py OUT_DIR WINDOW_DIR [WINDOW_DIR ...]"""
import concurrent.futures as cf, glob, os, re, subprocess, sys
PIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIM = os.path.join(PIM, "sim/build/ramulator2")
YAML = {"l40s": os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml"), "b200": os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml")}
CTRL = ["-p", "MemorySystem.Controller.fpu_gate_issue=false", "-p", "MemorySystem.Controller.wr_max_age=1000", "-p", "MemorySystem.Controller.pim_row_wait=0",
        "-p", "MemorySystem.Controller.class_priority=0,1,3,2", "-p", "MemorySystem.Controller.class_min_run=64", "-p", "MemorySystem.DRAM.org.channel=2",
        "-p", "Frontend.issue_width=2", "-p", "MemorySystem.Controller.dru_bus_slot=2"]
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
eoc = os.path.join(OUT, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
parse = lambda p: {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}
work = []
for d in sys.argv[2:]:
    for a in glob.glob(os.path.join(d, "*_alone.out")):
        tag = os.path.basename(a)[:-len("_alone.out")]
        if not tag.endswith(("_f4dru", "_sq")): continue
        jp = os.path.join(d, tag + ".jobs")
        if not os.path.exists(jp): continue
        lines = open(jp).read().splitlines(keepends=True)
        out, dropped = [], False
        for ln in lines:
            if ln.startswith("phase") and not dropped:
                p = ln.split()
                if p[1] == "0" and p[2] == "1" and p[4].startswith("1.0") and p[8] == "20000":   # gather: class-1 pure read at row 20000
                    dropped = True; continue
            out.append(ln)
        if not dropped: continue
        njp = os.path.join(OUT, tag + "_nogather.jobs"); open(njp, "w").writelines(out)
        work.append((tag, a, njp, os.path.join(OUT, tag + "_nogather.out")))
def run(w):
    tag, a, njp, o = w
    if not (os.path.exists(o) and "pipe0_finish_max" in open(o).read()):
        org = tag.split("_")[0]
        with open(o, "w") as f: subprocess.run([SIM, "-f", YAML[org], "-t", eoc, "-p", f"MemorySystem.host_jobs={njp}"] + CTRL, stdout=f, stderr=subprocess.STDOUT)
    return tag, parse(a)["pipe0_finish_max"], parse(o)["pipe0_finish_max"]
with cf.ThreadPoolExecutor(90) as ex:
    res = list(ex.map(run, work))
open(os.path.join(OUT, "ratios.txt"), "w").write("".join(f"{t} {b:.0f} {n:.0f} {n/b:.5f}\n" for t, b, n in res))
print(len(res), "pipelines re-simulated")
