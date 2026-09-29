#!/usr/bin/env python3
"""Re-derive Fig. 8 (plot_bw_bars.py D) from raw CSVs + sim anchors."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from lanes import *

# plot_bw_bars.py:42-55, copied for comparison only
D = {
    "fast": {
        "L40S": [(998.6, 410.3, 2.43), (572.4, 218.4, 2.62), (1415.0, 443.8, 3.19),
                 (341.0, 154.4, 2.21), (458.8, 218.3, 2.10)],
        "B200": [(410.5, 81.9, 5.01), (224.9, 23.7, 9.50), (562.2, 81.9, 6.86),
                 (187.6, 23.7, 7.92), (368.5, 47.4, 7.78)],
    },
    "slow": {
        "L40S": [(1020.8, 410.3, 2.49), (650.0, 218.4, 2.98), (1498.9, 443.8, 3.38),
                 (566.9, 238.0, 2.38), (1202.8, 712.2, 1.69)],
        "B200": [(430.5, 81.9, 5.26), (294.7, 60.1, 4.90), (637.6, 107.5, 5.93),
                 (418.7, 198.7, 2.11), (1203.4, 752.6, 1.60)],
    },
}
GEO = {"fast": {"L40S": 2.53, "B200": 7.26}, "slow": {"L40S": 2.61, "B200": 3.49}}
BETA = {"fast": 40e9, "slow": 400e6}


def model(L, alpha, mode):
    """Return (gpu, pim) ms for one cell under a named model."""
    n = L["n"]
    nic = (n + 2) * alpha
    gpu = L["gpu_dpf_g"] + L["ntt_dev"] + nic
    if mode == "doc":                     # e2e_coschedule.md: max of 3 lanes
        pim = max(L["spu"], L["sm"], nic)
    elif mode == "gang_serial_chain":     # gang, no overlap of NIC with SPU;
        pim = max(L["spu"] + nic, L["sm"])   # SM still pipelined vs SPU
    elif mode == "all_serial":
        pim = L["spu"] + L["sm"] + nic
    return gpu, pim


def run(mode="doc", verbose=True, **kw):
    res = {}
    for mach in ("L40S", "B200"):
        LN = lanes(mach, **kw)
        for tier in ("fast", "slow"):
            cells = []
            for i, (c, t) in enumerate(CFG):
                a = alpha_bw(c, t, BETA[tier])
                g, p, s = [], [], []
                for lg in LOGN:
                    L = LN.get((c, t, lg))
                    if L is None:
                        continue
                    gg, pp = model(L, a, mode)
                    g.append(gg); p.append(pp); s.append(gg / pp)
                cells.append((gm(g), gm(p), gm(s), len(s)))
            res[(tier, mach)] = cells
    if verbose:
        for tier in ("fast", "slow"):
            print(f"=== tier {tier} ({BETA[tier]/1e9:g} Gbps) mode={mode} {kw}")
            for mach in ("L40S", "B200"):
                ss = []
                for i, (c, t) in enumerate(CFG):
                    g, p, s, nc = res[(tier, mach)][i]
                    dg, dp, ds = D[tier][mach][i]
                    ss.append(s)
                    print(f"  {mach} ({c},{t:>3}) a={alpha_bw(c,t,BETA[tier])*1e3:8.1f}us"
                          f" n={nc} | GPU {g:7.1f} vs {dg:7.1f} ({g/dg-1:+.2%})"
                          f" | PIM {p:6.1f} vs {dp:6.1f} ({p/dp-1:+.2%})"
                          f" | S {s:5.2f} vs {ds:5.2f} ({s-ds:+.2f})")
                print(f"  {mach} geomean-of-5 speedup {gm(ss):.2f} vs GEO {GEO[tier][mach]}")
    return res


if __name__ == "__main__":
    run("doc")
