#!/usr/bin/env python3
"""Fig. 8-style headline from the end-to-end co-simulation (e2e_cosim.py output):
per (c,t) geomean over logN of GPU baseline ms, PCG^2 ms and speedup; 'up to' = max over
the 5 (c,t) configs, geomean-of-5 as in the paper. L40S = no DRU, B200 = with DRU.
Two baselines: SOTA (GPU DPF + GPU-NTT merge) and the paper's (GPU DPF + square four-step).
Usage: fig8_headline.py DIR_DRAMCLK DIR_1GHZ"""
import io, contextlib, json, os, sys
from math import exp, log
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_
def gm(v):
    v = list(v); return exp(sum(map(log, v)) / len(v))
CFG = [(8, 4), (4, 16), (8, 8), (2, 64), (2, 128)]
PAPER = {"fast": {"L40S": [2.43, 2.62, 3.19, 2.21, 2.10], "B200": [5.01, 9.50, 6.86, 7.92, 7.78]},
         "slow": {"L40S": [2.49, 2.98, 3.38, 2.38, 1.69], "B200": [5.26, 4.90, 5.93, 2.11, 1.60]}}
LN = {"L40S": L_("L40S"), "B200": L_("B200")}
runs = {"DRAM clock": sys.argv[1], "1 GHz": sys.argv[2]}
for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
    for M, org, des in (("L40S", "l40s", "merge"), ("B200", "b200", "f4dru")):
        print(f"\n{M} ({'no DRU' if M == 'L40S' else 'with DRU'}), {lab}: speedup per (c,t)  [paper | SOTA base: DRAM clk, 1 GHz | paper base: DRAM clk, 1 GHz]")
        res = {}
        for k, d in runs.items():
            R = json.load(open(os.path.join(d, "e2e.json")))
            for i, (c, t) in enumerate(CFG):
                rr = [r for r in R if r["org"] == org and r["design"] == des and r["tier"] == tier and r["c"] == c and r["t"] == t]
                pcg = [max(r["both_ms"], r["nic_ms"]) for r in rr]
                sq = [LN[M][(c, t, r["logN"])]["gpu_dpf_g"] + LN[M][(c, t, r["logN"])]["ntt_dev"] + r["nic_ms"] for r in rr]
                res[(k, "sota", i)] = gm(r["base_ms"] / p for r, p in zip(rr, pcg))
                res[(k, "paper", i)] = gm(b / p for b, p in zip(sq, pcg))
        for i, (c, t) in enumerate(CFG):
            print(f"  ({c},{t:3d}) {PAPER[tier][M][i]:5.2f} | {res[('DRAM clock','sota',i)]:5.2f} {res[('1 GHz','sota',i)]:5.2f} | {res[('DRAM clock','paper',i)]:5.2f} {res[('1 GHz','paper',i)]:5.2f}")
        f = lambda k, b: (max(res[(k, b, i)] for i in range(5)), gm(res[(k, b, i)] for i in range(5)))
        print(f"  up to / geomean-of-5: paper {max(PAPER[tier][M]):.2f}/{gm(PAPER[tier][M]):.2f} | SOTA {f('DRAM clock','sota')[0]:.2f}/{f('DRAM clock','sota')[1]:.2f}, "
              f"{f('1 GHz','sota')[0]:.2f}/{f('1 GHz','sota')[1]:.2f} | paper base {f('DRAM clock','paper')[0]:.2f}/{f('DRAM clock','paper')[1]:.2f}, {f('1 GHz','paper')[0]:.2f}/{f('1 GHz','paper')[1]:.2f}")
