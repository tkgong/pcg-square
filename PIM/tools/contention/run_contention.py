#!/usr/bin/env python3
"""GPU-PIM contention co-simulation (reviewers B/D/E).

Three configurations, all in ONE Ramulator2 DRAM model (no max() stitching):
  (a) GPU-only NTT          GPU stream = full fused four-step (butterflies + transposes)
  (b) GPU + SPU             same GPU stream, concurrent with the SPU DPF trace
  (c) GPU + SPU + DRU       GPU stream = SM lane only, transposes as a DRU stream,
                            concurrent with the SPU DPF trace
plus references: SPU alone, GPU SM lane alone, DRU alone.

GPU work is sized so that GPU-alone time / SPU-alone time = r (the lane balance of
a real PCG configuration) and is split into 4 chunks that start at the 4 SPU
rounds, so the overlap samples every SPU phase (EXTEND levels, fused CONVERT,
output pass). Per poly-mul traffic of fused4 (GPU_baseline/fused4):
  SM lane   : 144 N B, 61% reads  (K1,K2 x2 incl. twiddle table, PW, K3, K4)
  transposes:  96 N B, 50% reads  (6 transposes)
so the DRU stream is 2/3 of the SM-lane bytes and the full GPU NTT is 5/3.

Usage: run_contention.py --org l40s|b200 --out DIR [--r 0.12,0.25,0.5,1.0] [--jobs N]
"""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIM = os.path.dirname(os.path.dirname(HERE))
GEN = os.path.join(PIM, "tools", "gen_dpf_tree_trace.py")
ORGS = {
    # yaml, channels simulated (subset: contention is per channel and symmetric), tCK ns
    "l40s": dict(yaml=os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml"), C=8, tck=0.444),
    "b200": dict(yaml=os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml"), C=16, tck=0.500),
}
SM_RD, FULL_RD, DRU_RD = 88 / 144, 136 / 240, 0.5
GPU_ROW, DRU_ROW = 16384, 32768
PEAK_B_PER_CK = 16.0          # 32 B column per nBL = 2 CK, per channel


def sh(cmd, out):
    with open(out, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def parse(path):
    txt = open(path).read()
    d = {}
    for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", txt, re.M):
        d[m.group(1)] = float(m.group(2))
    if "watchdog" in txt or "terminate" in txt:
        d["_error"] = txt[-400:]
    return d


def write_streams(path, lines):
    with open(path, "w") as f:
        f.write("# class bytes_per_ch rd_frac window gap run row_base start\n")
        for l in lines:
            f.write(" ".join(str(x) for x in l) + "\n")


def chunks(cls, total_bytes, rd, window, gap, run, row, T, n=4):
    per = int(total_bytes // n)
    return [(cls, per, f"{rd:.4f}", window, gap, run, row, int(k * T / n)) for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, choices=ORGS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim", required=True, help="ramulator2 binary built from this branch")
    ap.add_argument("--r", default="0.12,0.25,0.5,1.0")
    ap.add_argument("--gap", type=int, default=0, help="GPU think time per request (CK)")
    ap.add_argument("--window", type=int, default=64, help="GPU outstanding reads per channel")
    ap.add_argument("--run", type=int, default=8, help="GPU columns per row visit (256 B)")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--policies", default="pim,fair,gpu")
    ap.add_argument("--dru-window", type=int, default=4,
                    help="DRU blocks in flight x 4 beats; 4 = the paper's single 1024-bit register file")
    ap.add_argument("--min-run", type=int, default=64, help="fair policy: cycles a class keeps the channel while it has ready requests")
    ap.add_argument("--spu-cl", type=int, default=155, help="EXTEND compute latency (DRAM CK); 155 = PU at the DRAM clock")
    ap.add_argument("--spu-convert-cl", type=int, default=465, help="CONVERT compute latency (DRAM CK)")
    ap.add_argument("--channels", type=int, default=0,
                    help="channels simulated (0 = org default); channels are symmetric")
    ap.add_argument("--row-hold", type=int, default=-1, help="PIM row hold after a PIM column access (CK; -1 = off)")
    ap.add_argument("--anticipate", type=int, default=-1, help="host stops opening rows this many CK before the PIM FPU frees (-1 = off)")
    ap.add_argument("--row-wait", type=int, default=-1,
                    help="PIM row-change protection (cycles; -1 = off): host classes stop opening rows")
    ap.add_argument("--wr-age", type=int, default=1000,
                    help="controller drains posted writes once the oldest is this many CK old (0 = watermarks only)")
    a = ap.parse_args()
    O = ORGS[a.org]; C = a.channels or O["C"]
    os.makedirs(a.out, exist_ok=True)
    base = [a.sim, "-f", O["yaml"], "-p", "MemorySystem.Controller.fpu_gate_issue=true",
            "-p", f"Frontend.issue_width={C}", "-p", f"MemorySystem.DRAM.org.channel={C}",
            "-p", f"MemorySystem.Controller.wr_max_age={a.wr_age}",
            "-p", f"MemorySystem.Controller.pim_row_wait={a.row_wait}",
            "-p", f"MemorySystem.Controller.pim_row_hold={a.row_hold}",
            "-p", f"MemorySystem.Controller.pim_anticipate={a.anticipate}"]
    spu_trace = os.path.join(a.out, f"spu_{a.org}_C{C}.trace")
    if not os.path.exists(spu_trace):
        subprocess.run([sys.executable, GEN, "-n", "12", "-C", str(C), "-P", "8", "-I", str(C * 8 * 4),
                        "--mode", "instances", "--seed-bits", "128", "--reread", "--broadcast",
                        "--cl", str(a.spu_cl), "--reduce", "chacha", "--convert-cl", str(a.spu_convert_cl), "-o", spu_trace],
                       check=True, stderr=subprocess.DEVNULL)
    eoc = os.path.join(a.out, "eoc.trace")
    open(eoc, "w").write("AiM EOC\n")
    # arbitration policies between the PIM (all-bank) and host (GPU/DRU) classes
    POL = {
        "pim": ["-p", "MemorySystem.Controller.class_priority=0,1,3,2",
                "-p", "MemorySystem.Controller.pim_strict=true"],
        "gpu": ["-p", "MemorySystem.Controller.class_priority=1,3,0,2"],
        "fair": ["-p", "MemorySystem.Controller.class_priority=0,1,3,2",
                 "-p", f"MemorySystem.Controller.class_min_run={a.min_run}"],
    }

    def run(tag, trace, streams, policy="pim"):
        out = os.path.join(a.out, tag + ".out")
        if os.path.exists(out) and "memory_system_cycles" in open(out).read():
            return tag, parse(out)
        cmd = base + ["-t", trace] + POL[policy]
        if streams:
            sp = os.path.join(a.out, tag + ".streams")
            write_streams(sp, streams)
            cmd += ["-p", f"MemorySystem.host_streams={sp}"]
        sh(cmd, out)
        return tag, parse(out)

    # 1) SPU alone -> T_spu; GPU calibration runs (alone) to get achieved BW
    res = dict([run("spu_only", spu_trace, [])])
    T = res["spu_only"]["pim_done_cycles"]
    probe = 2_000_000   # bytes per channel for the standalone BW probe
    for kind, rd in (("sm", SM_RD), ("full", FULL_RD)):
        res.update([run(f"probe_{kind}", eoc, [(1, probe, f"{rd:.4f}", a.window, a.gap, a.run, GPU_ROW, 0)])])
    res.update([run("probe_dru", eoc, [(2, probe, "0.5", a.dru_window, 0, 4, DRU_ROW, 0)])])
    bw = {k: probe / res[f"probe_{k}"]["stream0_class{}_finish_mean".format(2 if k == "dru" else 1)]
          for k in ("sm", "full", "dru")}

    jobs = []
    for r in [float(x) for x in a.r.split(",")]:
        gpu_full = r * T * bw["full"]                      # bytes/ch so GPU-full alone takes r*T
        gpu_sm = gpu_full * 144 / 240                      # same poly-muls, SM lane only
        dru = gpu_sm * 96 / 144
        s_full = chunks(1, gpu_full, FULL_RD, a.window, a.gap, a.run, GPU_ROW, T)
        s_sm = chunks(1, gpu_sm, SM_RD, a.window, a.gap, a.run, GPU_ROW, T)
        s_dru = chunks(2, dru, DRU_RD, a.dru_window, 0, 4, DRU_ROW, T)
        rt = f"r{r:g}"
        jobs += [(f"{rt}_a_gpufull", eoc, s_full, "pim"),
                 (f"{rt}_ref_gpusm", eoc, s_sm, "pim"),
                 (f"{rt}_ref_dru", eoc, s_dru, "pim"),
                 (f"{rt}_ref_smdru", eoc, s_sm + s_dru, "pim")]
        for pol in a.policies.split(","):
            jobs += [(f"{rt}_b_{pol}", spu_trace, s_full, pol),
                     (f"{rt}_c_{pol}", spu_trace, s_sm + s_dru, pol)]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for tag, d in ex.map(lambda j: run(*j), jobs):
            res[tag] = d
    json.dump(dict(org=a.org, C=C, T_spu=T, bw_B_per_ck=bw, results=res),
              open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(f"org={a.org} C={C} T_spu={T:.0f} CK  standalone BW (B/CK/ch): "
          + " ".join(f"{k}={v:.2f} ({v / PEAK_B_PER_CK * 100:.0f}%)" for k, v in bw.items()))
    print("wrote", os.path.join(a.out, "results.json"))


if __name__ == "__main__":
    main()
