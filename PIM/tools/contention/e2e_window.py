#!/usr/bin/env python3
"""End-to-end PCG^2 per Fig. 8 cell, contention measured in a steady-state WINDOW.

e2e_cosim.py shrank every NTT kernel by the cell's time scale (as small as ~1/500 on
L40S), so fixed per-kernel DRAM latency inflated the simulated NTT lane 1.5-2.3x.
Here kernels keep a size the standalone validation reproduces (>= ~20k CK each): the
window is one run of the SPU trace, and the NTT pipeline inside it is sized so that
NTT-alone / SPU-alone matches the cell's real lane ratio r. The window gives the two
lanes' co-run slowdowns; the cell is then

   PCG^2 = max( SPU lane x SPU slowdown, NTT lane x NTT slowdown, NIC )

with the SPU lane from the paper's anchor (scaled by the SPU clock) and the NTT lane
(incl. the DPF->NTT gather) from the simulator's standalone per-mul time.
Usage: e2e_window.py --sim BIN --out DIR [--clock nom|1.0|0.5] [--gather --spu-presum]
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
FNOM = {"l40s": 2.25, "b200": 2.0}; MACH = {"l40s": "L40S", "b200": "B200"}; NSPU = {"l40s": 192, "b200": 2048}
KERNEL_MIN_CK = 20000


def parse(p):
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--clock", default="nom"); ap.add_argument("--orgs", default="l40s,b200")
    ap.add_argument("--designs", default="merge,f4dru"); ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--gather", action="store_true"); ap.add_argument("--spu-presum", action="store_true")
    ap.add_argument("--spu-lane", default="paper", choices=["paper", "sim"],
                    help="paper: Fig. 8 anchor (718,660 CK, --reduce mau: no per-leaf H' on the SPU) scaled by T(clock)/T(nominal); "
                         "sim: this run's SPU trace time (--reduce chacha: expand + H' + mod-p on the SPU) x ceil(I/NSPU) x leaves/instance")
    ap.add_argument("--reduce", default="chacha", choices=["chacha", "mau", "fused"],
                    help="SPU trace leaf conversion: chacha = expand + per-leaf ChaCha8 hash H' + mod-p on the SPU (Sec. IV); "
                         "mau = the paper's Fig. 8 anchor mode (conversion as a channel-level column stream, H' not charged)")
    ap.add_argument("--alu-mult", type=float, default=1.0,
                    help="SPU datapath width multiplier (2 = two VADD/VXORL pairs, or 16 lanes): divides the per-op cycle counts")
    ap.add_argument("--fpu-gate", default="true", help="fpu_gate_issue: true = campaign (serial PU, no load/compute overlap), false = LSU overlap as described in Sec. IV")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
    ctrl = ["-p", f"MemorySystem.Controller.fpu_gate_issue={a.fpu_gate}", "-p", "MemorySystem.Controller.wr_max_age=1000",
            "-p", "MemorySystem.Controller.pim_row_wait=0", "-p", "MemorySystem.Controller.class_priority=0,1,3,2",
            "-p", "MemorySystem.Controller.class_min_run=64", "-p", "MemorySystem.DRAM.org.channel=2", "-p", "Frontend.issue_width=2"]

    def spu_trace(org, f, tag):
        tr = os.path.join(a.out, f"spu_{org}_{a.reduce}_x{a.alu_mult:g}_{tag}.trace"); o = tr[:-6] + ".out"
        if not os.path.exists(tr):
            subprocess.run([sys.executable, GEN, "-n", "12", "-C", "2", "-P", "8", "-I", "64", "--mode", "instances", "--seed-bits", "128",
                            "--reread", "--broadcast", "--cl", str(round(155 * FNOM[org] / f / a.alu_mult)), "--reduce", a.reduce,
                            "--convert-cl", str(round(465 * FNOM[org] / f / a.alu_mult)), "-o", tr], check=True, stderr=subprocess.DEVNULL)
        if not (os.path.exists(o) and "pim_done_cycles" in open(o).read()):
            with open(o, "w") as fo: subprocess.run([a.sim, "-f", YAML[org], "-t", tr] + ctrl, stdout=fo, stderr=subprocess.STDOUT)
        return tr, parse(o)["pim_done_cycles"]

    orgs = a.orgs.split(","); TR, TS, CLK = {}, {}, {}
    for org in orgs:
        f = FNOM[org] if a.clock == "nom" else float(a.clock)
        TR[org], TS[org] = spu_trace(org, f, "clk")
        CLK[org] = TS[org] / spu_trace(org, FNOM[org], "nom")[1]          # SPU lane scale vs the anchor clock

    def build(org, d, lg, c, t, r):
        """kernel pipeline for the window; returns (jobs file, n muls, scale)"""
        tm = ntt_table(org)[(lg, c * c)][0 if d == "merge" else 1]
        mul_ck = tm * 1e6 / DEV[org]["tck"]                                  # real per-mul CK
        n = max(1, int(r * TS[org] / (10 * KERNEL_MIN_CK)))
        s = r * TS[org] / (n * mul_ck)
        ph = phases(org, d, lg, c * c, scale=s)
        if a.gather:
            m = max(1, min(t, (c * c * t * t) // NSPU[org])) if a.spu_presum else 1
            gb = int(round(8 * (2 * (1 << lg) * t) / 2 / m / DEV[org]["ch"] * s))
            ph = [(0, 1, gb, 1.0, 48, 0, 16, 20000, 0)] + ph
        jp = os.path.join(a.out, f"{org}_{c}_{t}_{lg}_{d}.jobs"); write_jobs(jp, [(n, 0, ph)])
        return jp, n, s

    def sim(tag, trace, jp, org):
        out = os.path.join(a.out, tag + ".out")
        if not (os.path.exists(out) and "memory_system_cycles" in open(out).read()):
            with open(out, "w") as fo:
                subprocess.run([a.sim, "-f", YAML[org], "-t", trace, "-p", f"MemorySystem.host_jobs={jp}"] + ctrl, stdout=fo, stderr=subprocess.STDOUT)
        return parse(out)

    cells, work = {}, []
    for org in orgs:
        LN = L_(MACH[org])
        for (c, t) in CFG:
            for lg in LOGN:
                L = LN.get((c, t, lg))
                if not L or (lg, c * c) not in ntt_table(org): continue
                if a.spu_lane == "sim":
                    I = c * c * t * t; n_leaf = 2 * (1 << lg) // t
                    spu_ms = TS[org] / 4 * DEV[org]["tck"] / 1e6 * math.ceil(I / NSPU[org]) * n_leaf / 4096   # sim: 4 instances of 4096 leaves per SPU
                else:
                    spu_ms = L["spu"] * CLK[org]
                for d in a.designs.split(","):
                    tm = ntt_table(org)[(lg, c * c)][0 if d == "merge" else 1]
                    r = tm * 2 * c * c / spu_ms                                   # real lane ratio (pre-gather)
                    jp, n, s = build(org, d, lg, c, t, r)
                    cells[(org, c, t, lg, d)] = dict(L=L, spu_ms=spu_ms, n=n, s=s, tm=tm)
                    work += [(f"{org}_{c}_{t}_{lg}_{d}", TR[org], jp, org), (f"{org}_{c}_{t}_{lg}_{d}_alone", eoc, jp, org)]
    R = {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for (tag, *_), d in zip(work, ex.map(lambda w: sim(*w), work)): R[tag] = d

    rows = []
    for (org, c, t, lg, d), cd in cells.items():
        tck = DEV[org]["tck"]; x = R[f"{org}_{c}_{t}_{lg}_{d}"]; y = R[f"{org}_{c}_{t}_{lg}_{d}_alone"]
        ntt_lane = y["pipe0_finish_max"] / cd["s"] / cd["n"] * 2 * c * c * tck / 1e6   # real ms, incl. gather
        spu_slow = x["pim_done_cycles"] / TS[org]; ntt_slow = x["pipe0_finish_max"] / y["pipe0_finish_max"]
        mm = ntt_table(org)[(lg, c * c)][0]
        for tier in ("fast", "slow"):
            nic = (cd["L"]["n"] + 2) * alpha_bw(c, t, BETA[tier])
            gpu = cd["L"]["gpu_dpf_g"] + mm * 2 * c * c
            pcg = max(cd["spu_ms"] * spu_slow, ntt_lane * ntt_slow, nic)
            rows.append(dict(org=org, c=c, t=t, logN=lg, tier=tier, design=d, spu_ms=cd["spu_ms"], ntt_ms=ntt_lane,
                             ntt_meas_ms=cd["tm"] * 2 * c * c, spu_slow=spu_slow, ntt_slow=ntt_slow, nic_ms=nic,
                             pcg_ms=pcg, pcg_nocont_ms=max(cd["spu_ms"], ntt_lane, nic), gpu_ms=gpu,
                             base_serial_ms=gpu + nic, base_overlap_ms=max(gpu, nic),
                             base_paper_ms=cd["L"]["gpu_dpf_g"] + cd["L"]["ntt_dev"] + nic))
    json.dump(rows, open(os.path.join(a.out, "e2e.json"), "w"), indent=1)
    print(f"clock={a.clock}  (best (c,t) / geomean over all cells; SOTA baseline = GPU DPF + merge NTT)")
    for org in orgs:
        for tier in ("fast", "slow"):
            for d in a.designs.split(","):
                rr = [r for r in rows if r["org"] == org and r["tier"] == tier and r["design"] == d]
                def hb(key):
                    per = {}
                    for r in rr: per.setdefault((r["c"], r["t"]), []).append(r[key] / r["pcg_ms"])
                    allv = [r[key] / r["pcg_ms"] for r in rr]
                    return max(gm(v) for v in per.values()), gm(allv)
                nc = [r["base_serial_ms"] / r["pcg_nocont_ms"] for r in rr]
                ntt_err = gm(r["ntt_ms"] / r["ntt_meas_ms"] for r in rr)
                print(f"  {MACH[org]} {tier} {d:6s}: serial-net base {hb('base_serial_ms')[0]:.2f}/{hb('base_serial_ms')[1]:.2f}"
                      f" | overlap-net base {hb('base_overlap_ms')[0]:.2f}/{hb('base_overlap_ms')[1]:.2f}"
                      f" | paper base {hb('base_paper_ms')[0]:.2f}/{hb('base_paper_ms')[1]:.2f}"
                      f" | no-cont geo {gm(nc):.2f} | SPU slow {min(r['spu_slow'] for r in rr):.2f}-{max(r['spu_slow'] for r in rr):.2f}"
                      f" NTT slow {min(r['ntt_slow'] for r in rr):.2f}-{max(r['ntt_slow'] for r in rr):.2f} | sim NTT lane/measured {ntt_err:.2f}")


if __name__ == "__main__":
    main()
