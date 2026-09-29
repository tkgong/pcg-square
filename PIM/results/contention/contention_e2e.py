#!/usr/bin/env python3
"""Fig. 8 speedups with a GPU-PIM contention factor s on the two DRAM lanes that
run concurrently in PCG^2 (SPU and SM/NTT): PCG^2 = max(s*spu, s*sm, nic).
The GPU-only baseline has no PIM, so it is unchanged. Variant P = paper lanes
(square + DRU); M = GPU-NTT merge on both sides (merge_e2e.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reproduce import L_, CFG, LOGN, alpha_bw, gm
from fig8 import BETA
import merge_e2e as me

def run(mach, tier, variant, s):
    LN = L_(mach); per, cells = [], []
    for (c, t) in CFG:
        a = alpha_bw(c, t, BETA[tier]); S = []
        for lg in LOGN:
            L = LN.get((c, t, lg))
            if not L: continue
            nic = (L["n"] + 2) * a
            if variant == "P": ng, sm = L["ntt_dev"], L["sm"]
            else: ng = sm = me.MERGE[mach][(lg, c * c)] * 2 * c * c
            g = L["gpu_dpf_g"] + ng + nic
            p = max(s * L["spu"], s * sm, nic)
            S.append(g / p)
        per.append(gm(S)); cells += S
    return per, gm(cells)

SS = (1.00, 1.05, 1.10, 1.15, 1.20)
for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
    print(f"==== {lab}")
    for mach in ("L40S", "B200"):
        for v in ("P", "M"):
            row = []
            for s in SS:
                per, g = run(mach, tier, v, s)
                row.append(f"s={s:.2f}: max {max(per):5.2f} geo {g:5.2f}")
            print(f"  {mach} {v}  " + " | ".join(row))
