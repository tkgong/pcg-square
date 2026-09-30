#!/usr/bin/env python3
"""Fig. 8 headline with the STRENGTHENED GPU baseline on L40S: per cell, GPU DPF = min over arms
(serial/batched) of expand+convert with the single-hash conversion (measured here, network-free
since alpha=0 in this sweep), NTT = merge (measured), network per the paper. PCG^2 lanes from the
co-simulation window (overlap_simlane). Compares with the paper's baseline arm (serial, two-pass H')."""
import re, json, sys, os
from math import exp, log
sys.path.insert(0, "/home/tkgong/pcg-square/PIM/tools/contention/e2e")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, LOGN, alpha_bw, gm
    from fig8 import BETA
LN = L_("L40S")
new = {}
for l in open(sys.argv[1]):
    m = re.match(r"c=(\d+) t=(\d+) logN=(\d+) arm=(\w+) : expand=([\d.]+) convert=([\d.]+) beaver=([\d.]+)", l)
    if m:
        c, t, lg, arm = int(m[1]), int(m[2]), int(m[3]), m[4]
        v = float(m[5]) + float(m[6]) + float(m[7])
        k = (c, t, lg); new[k] = min(new.get(k, 1e18), v)
C = "/tmp/claude-1009/-home-tkgong-pcg-square/a7a02934-9915-481c-9898-61199d636b0a/scratchpad/cont"
for clk, d in (("2.25 GHz", "winabs_nom"), ("1 GHz", "winabs_1.0")):
    R = json.load(open(f"{C}/{d}/e2e.json"))
    for tier in ("fast", "slow"):
        per_old, per_new, cells_old, cells_new = [], [], [], []
        for (c, t) in CFG:
            so, sn = [], []
            for r in R:
                if not (r["org"] == "l40s" and r["design"] == "merge" and r["tier"] == tier and r["c"] == c and r["t"] == t): continue
                lg = r["logN"]; L = LN[(c, t, lg)]
                ntt = r["gpu_ms"] - L["gpu_dpf_g"]                       # merge NTT lane in the baseline
                base_old = r["base_serial_ms"]
                if (c, t, lg) not in new: continue
                base_new = new[(c, t, lg)] * 1.026 + ntt + r["nic_ms"]     # same batch_cost factor as lanes.py for the serial arm
                so.append(base_old / r["pcg_ms"]); sn.append(base_new / r["pcg_ms"])
            if so: per_old.append(gm(so)); per_new.append(gm(sn)); cells_old += so; cells_new += sn
        print(f"L40S SPU {clk:9s} {'40 Gbps ' if tier=='fast' else '400 Mbps'}: paper-arm baseline {max(per_old):.2f}/{gm(cells_old):.2f}  ->  stronger baseline (best arm, single H') {max(per_new):.2f}/{gm(cells_new):.2f}   per (c,t): " + " ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(per_old, per_new)))
