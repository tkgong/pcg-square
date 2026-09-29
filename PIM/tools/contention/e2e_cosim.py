#!/usr/bin/env python3
"""End-to-end PCG^2 per Fig. 8 cell, computed IN the simulator.

For each (machine, c, t, logN) cell and design, one Ramulator2 run holds the SPU
DPF trace and the cell's 2c^2 NTT poly-muls as a host job pipeline (GPU kernels
with measured compute floors, DRU transposes as bandwidth-limited phases; see
ntt_jobs.py). Steady state of the PCG pipeline: the NTTs of expansion i run
while the SPU expands i+1, so all NTT jobs start at 0 next to the SPU trace.
The run is scaled so that the simulated SPU time equals the cell's SPU lane
(bytes and compute floors scale together), and

   PCG^2(cell) = max( sim both-done, NIC lane )      [NIC = (n+2) alpha, paper]
   baseline    = GPU DPF + merge NTT (2c^2 poly-muls, measured) + NIC

Designs: merge (GPU-NTT merge on the SMs) and f4dru (fused four-step on GPU-NTT
kernels, transposes on the DRU). Also runs each NTT pipeline without the SPU and
reports both-done / max(alone) per cell.
Usage: e2e_cosim.py --sim BIN --out DIR [--clock nom|1.0|...] [--dru-window 4]
"""
import argparse, concurrent.futures as cf, io, contextlib, json, math, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "e2e"))
from ntt_jobs import DEV, ntt_table, phases, write_jobs
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, LOGN, alpha_bw, gm
    from fig8 import BETA
PIM = os.path.dirname(os.path.dirname(HERE))
GEN = os.path.join(PIM, "tools", "gen_dpf_tree_trace.py")
YAML = {"l40s": os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml"), "b200": os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml")}
FNOM = {"l40s": 2.25, "b200": 2.0}
MACH = {"l40s": "L40S", "b200": "B200"}


def parse(p):
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--clock", default="nom", help="SPU clock in GHz or 'nom' (= DRAM clock)")
    ap.add_argument("--dru-window", type=int, default=4)
    ap.add_argument("--orgs", default="l40s,b200"); ap.add_argument("--designs", default="merge,f4dru")
    ap.add_argument("--jobs", type=int, default=60)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
    ctrl = ["-p", "MemorySystem.Controller.fpu_gate_issue=true", "-p", "MemorySystem.Controller.wr_max_age=1000",
            "-p", "MemorySystem.Controller.pim_row_wait=0", "-p", "MemorySystem.Controller.class_priority=0,1,3,2",
            "-p", "MemorySystem.Controller.class_min_run=64", "-p", "MemorySystem.DRAM.org.channel=2",
            "-p", "Frontend.issue_width=2"]
    spu_T = {}
    for org in a.orgs.split(","):
        f = FNOM[org] if a.clock == "nom" else float(a.clock)
        tr = os.path.join(a.out, f"spu_{org}.trace")
        if not os.path.exists(tr):
            subprocess.run([sys.executable, GEN, "-n", "12", "-C", "2", "-P", "8", "-I", "64", "--mode", "instances",
                            "--seed-bits", "128", "--reread", "--broadcast", "--cl", str(round(155 * FNOM[org] / f)),
                            "--reduce", "chacha", "--convert-cl", str(round(465 * FNOM[org] / f)), "-o", tr],
                           check=True, stderr=subprocess.DEVNULL)
        o = os.path.join(a.out, f"spu_{org}.out")
        if not (os.path.exists(o) and "pim_done_cycles" in open(o).read()):
            with open(o, "w") as fo:
                subprocess.run([a.sim, "-f", YAML[org], "-t", tr] + ctrl, stdout=fo, stderr=subprocess.STDOUT)
        spu_T[org] = parse(o)["pim_done_cycles"]
        # SPU slowdown of this clock vs nominal (for the lane): nominal trace time
        if a.clock != "nom":
            trn = os.path.join(a.out, f"spu_{org}_nom.trace"); on = os.path.join(a.out, f"spu_{org}_nom.out")
            if not os.path.exists(on):
                subprocess.run([sys.executable, GEN, "-n", "12", "-C", "2", "-P", "8", "-I", "64", "--mode", "instances",
                                "--seed-bits", "128", "--reread", "--broadcast", "--cl", "155", "--reduce", "chacha",
                                "--convert-cl", "465", "-o", trn], check=True, stderr=subprocess.DEVNULL)
                with open(on, "w") as fo:
                    subprocess.run([a.sim, "-f", YAML[org], "-t", trn] + ctrl, stdout=fo, stderr=subprocess.STDOUT)
            spu_T[org + "_nom"] = parse(on)["pim_done_cycles"]

    jobs = []; cells = {}
    for org in a.orgs.split(","):
        LN = L_(MACH[org]); tck = DEV[org]["tck"]
        clk_scale = spu_T[org] / spu_T[org + "_nom"] if a.clock != "nom" else 1.0
        for (c, t) in CFG:
            for lg in LOGN:
                L = LN.get((c, t, lg))
                if not L or (lg, c * c) not in ntt_table(org): continue
                spu_ms = L["spu"] * clk_scale
                s = spu_T[org] * tck / (spu_ms * 1e6)                 # real -> sim scale
                cells[(org, c, t, lg)] = dict(L=L, spu_ms=spu_ms, s=s)
                for d in a.designs.split(","):
                    ph = phases(org, d, lg, c * c, scale=s, dru_window=a.dru_window)
                    tag = f"{org}_{c}_{t}_{lg}_{d}"
                    jp = os.path.join(a.out, tag + ".jobs"); write_jobs(jp, [(2 * c * c, 0, ph)])
                    jobs.append((tag, os.path.join(a.out, f"spu_{org}.trace"), jp, org))
                    jobs.append((tag + "_alone", eoc, jp, org))

    def run(tag, trace, jp, org):
        out = os.path.join(a.out, tag + ".out")
        if not (os.path.exists(out) and "memory_system_cycles" in open(out).read()):
            with open(out, "w") as fo:
                subprocess.run([a.sim, "-f", YAML[org], "-t", trace, "-p", f"MemorySystem.host_jobs={jp}"] + ctrl,
                               stdout=fo, stderr=subprocess.STDOUT)
        return tag, parse(out)
    R = {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for tag, d in ex.map(lambda j: run(*j), jobs): R[tag] = d

    rows = []
    for (org, c, t, lg), cd in cells.items():
        L, s = cd["L"], cd["s"]; tck = DEV[org]["tck"]; mm = ntt_table(org)[(lg, c * c)][0]
        for tier in ("fast", "slow"):
            nic = (L["n"] + 2) * alpha_bw(c, t, BETA[tier])
            base = L["gpu_dpf_g"] + mm * 2 * c * c + nic
            for d in a.designs.split(","):
                x = R[f"{org}_{c}_{t}_{lg}_{d}"]; y = R[f"{org}_{c}_{t}_{lg}_{d}_alone"]
                both = max(x["pim_done_cycles"], x["pipe0_finish_max"]) * tck / 1e6 / s
                ntt_alone = y["pipe0_finish_max"] * tck / 1e6 / s
                ideal = max(cd["spu_ms"], ntt_alone)
                rows.append(dict(org=org, c=c, t=t, logN=lg, tier=tier, design=d, spu_ms=cd["spu_ms"],
                                 ntt_alone_ms=ntt_alone, both_ms=both, nic_ms=nic, base_ms=base,
                                 vs_max=both / ideal, speedup=base / max(both, nic),
                                 speedup_nocont=base / max(ideal, nic)))
    json.dump(rows, open(os.path.join(a.out, "e2e.json"), "w"), indent=1)
    print(f"clock={a.clock} dru_window={a.dru_window}   (best (c,t) / geomean over all cells; baseline = GPU DPF + merge NTT)")
    for org in a.orgs.split(","):
        for tier in ("fast", "slow"):
            for d in a.designs.split(","):
                rr = [r for r in rows if r["org"] == org and r["tier"] == tier and r["design"] == d]
                per = {}
                for r in rr: per.setdefault((r["c"], r["t"]), []).append(r)
                best = max(gm(x["speedup"] for x in v) for v in per.values())
                best0 = max(gm(x["speedup_nocont"] for x in v) for v in per.values())
                vm = [r["vs_max"] for r in rr]
                print(f"  {MACH[org]} {tier} {d:6s}: no contention {best0:.2f}/{gm(r['speedup_nocont'] for r in rr):.2f}"
                      f"  -> co-sim {best:.2f}/{gm(r['speedup'] for r in rr):.2f}   both/max {min(vm):.2f}-{max(vm):.2f}")


if __name__ == "__main__":
    main()
