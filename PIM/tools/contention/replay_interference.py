#!/usr/bin/env python3
"""Replay the silicon interference experiment (GPU_baseline/fused4/interfere_bench.cu)
in Ramulator2: a victim host stream shaped like the fused four-step NTT, plus a
class-3 aggressor stream whose STANDALONE bandwidth matches the silicon
aggressor's standalone bandwidth. Reports the victim slowdown per point, to be
compared with the silicon slowdown net of the SM-occupancy control.

Usage: replay_interference.py --sim BIN --out DIR [--org l40s] [--targets 0.15,0.30,0.61]
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIM = os.path.dirname(os.path.dirname(HERE))
YAML = {"l40s": os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml"),
        "b200": os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml")}
VICTIM = {"f4_full": 136 / 240, "f4_sm": 88 / 144}


def run(a, tag, streams):
    out = os.path.join(a.out, tag + ".out")
    sp = os.path.join(a.out, tag + ".streams")
    with open(sp, "w") as f:
        for l in streams:
            f.write(" ".join(map(str, l)) + "\n")
    eoc = os.path.join(a.out, "eoc.trace")
    open(eoc, "w").write("AiM EOC\n")
    cmd = [a.sim, "-f", YAML[a.org], "-t", eoc, "-p", "MemorySystem.DRAM.org.channel=4",
           "-p", "Frontend.issue_width=4", "-p", f"MemorySystem.host_streams={sp}",
           "-p", "MemorySystem.Controller.class_priority=1,3,0,2"]
    with open(out, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    d = {m.group(1): float(m.group(2)) for m in
         re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(out).read(), re.M)}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--org", default="l40s")
    ap.add_argument("--targets", default="0.15,0.30,0.61")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    AGG_BYTES = 4_000_000
    # calibrate the aggressor gap to each standalone bandwidth target
    cal = {}
    for g in (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20):
        d = run(a, f"agg_alone_g{g}", [(3, 1_000_000, 0.5, 32, g, 8, 40000, 0)])
        cal[g] = 1_000_000 / d["stream0_class3_finish_mean"] / 16.0
    print("aggressor standalone BW fraction by gap:", {g: round(v, 3) for g, v in cal.items()})
    for vname, rd in VICTIM.items():
        vb = 1_000_000
        base = run(a, f"{vname}_alone", [(1, vb, f"{rd:.4f}", 32, 0, 8, 16384, 0)])
        tb = base["stream0_class1_finish_mean"]
        print(f"{vname}: alone {tb:.0f} CK ({vb / tb / 16 * 100:.0f}% of peak)")
        for t in [float(x) for x in a.targets.split(",")]:
            g = min(cal, key=lambda k: abs(cal[k] - t))
            d = run(a, f"{vname}_agg{t}", [(1, vb, f"{rd:.4f}", 32, 0, 8, 16384, 0),
                                           (3, AGG_BYTES, 0.5, 32, g, 8, 40000, 0)])
            tv = d["stream0_class1_finish_mean"]
            print(f"  aggressor standalone {cal[g] * 100:.0f}% (gap {g}): victim {tv / tb:.3f}x")


if __name__ == "__main__":
    main()
