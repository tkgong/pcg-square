#!/usr/bin/env python3
"""Independent re-derivation of PCG^2 Fig. 8, Table IV, Fig. 10 and the
plot_dru_pipe.py E2E table from the raw repo CSVs + ramulator2 anchors.

Read-only on the repo. Run:  python3 reproduce.py > reproduce_out.txt

Method that reproduces the hard-coded tables (found by iteration):
  GPU  = 1.026 * [expand + convert + beaver - (exchanges + 2c^2)*alpha_meas]
             (serial arm, smallest measured alpha: L40S 50 us, B200 5 us;
              for (4,16) the lambda=128 row, i.e. dict "last row wins")
         + NTT_dev                       (ms_per_mul(logN, batch=c^2) * 2c^2)
         + (n+2) * alpha
  SPU  = c^2*2N*t leaves * pim_ns * max(1, N_SPU/I)
         L40S pim_ns = 718660*0.444/(768*4096)  (+ ramulator per-level ladder)
         B200 pim_ns = L40S * 0.1069, times an unexplained 1.0178
  SM   = NTT_dev  (B200: * (1 - DRU share))
  PIM  = max(SPU, SM, (n+2)*alpha)
  Fig. 8   : NTT = standalone-brev four-step; DRU share = (transpose+brev)/sum
             alpha = 2*c^2*16*t^2 B / beta, beta = 40 Gbps / 400 Mbps
  Table IV : NTT = per-cell faster of {standalone, fold}; DRU share =
             transpose(+brev) share of the chosen variant; alpha = 500 us;
             "+SPU" = SPU + SM + NIC, "+co-schedule" = max(SPU, SM, NIC)
  Fig. 10  : SM lane ignored; serial = SPU + NIC, co-sched = max(SPU, NIC);
             alpha* = geomean_cells(SPU/(n+2)); utilisation = geomean of
             lane/wall; throughput = geomean(N / wall)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanes import *
from fig8 import D, GEO, BETA

B200_PIM_FUDGE = 1.0178
PAPER_FIG8 = {  # labels printed in the paper's Fig. 8 (p.10)
    "fast": {"L40S": [2.4, 2.6, 3.2, 2.2, 2.1], "B200": [5.0, 9.5, 6.9, 7.9, 7.8]},
    "slow": {"L40S": [2.5, 3.0, 3.4, 2.4, 1.7], "B200": [5.3, 4.9, 5.9, 2.1, 1.6]},
}
PAPER_T4 = {"L40S": (701.8, 399.1, 399.1, 270.7, 270.7),
            "B200": (332.3, 86.8, 81.4, 50.6, 46.6)}
DRUPIPE = {  # plot_dru_pipe.py:49-58
    "L40S": [(414.0, 414.0, 278.0, 278.0), (459.3, 459.3, 278.0, 278.0),
             (671.9, 671.9, 357.5, 357.5), (1139.2, 1139.2, 691.5, 691.5)],
    "B200": [(90.1, 80.9, 53.4, 46.2), (130.3, 120.2, 73.9, 66.9),
             (311.5, 298.8, 215.5, 210.7), (731.8, 716.8, 614.0, 614.0)],
}
SMBOUND = {"L40S": [8, 8, 6, 4], "B200": [10, 6, 2, 0]}


def L_(mach, **kw):
    kw.setdefault("spu_model", "ladder")
    kw.setdefault("spu_scale", B200_PIM_FUDGE if mach == "B200" else 1.0)
    LN = lanes(mach, **kw)
    if mach == "L40S":
        for L in LN.values():
            L["sm"] = L["ntt_dev"]
    return LN


def suite(LN):
    return [LN[(c, t, lg)] for (c, t) in CFG for lg in LOGN if (c, t, lg) in LN]


def fig8():
    print("=" * 100)
    print("FIG. 8  (plot_bw_bars.py D) -- reproduced vs hard-coded D vs paper label")
    for tier in ("fast", "slow"):
        for mach in ("L40S", "B200"):
            LN = L_(mach)
            allc = []
            for i, (c, t) in enumerate(CFG):
                a = alpha_bw(c, t, BETA[tier])
                G, P, S = [], [], []
                for lg in LOGN:
                    L = LN.get((c, t, lg))
                    if not L:
                        continue
                    nic = (L["n"] + 2) * a
                    g = L["gpu_dpf_g"] + L["ntt_dev"] + nic
                    p = max(L["spu"], L["sm"], nic)
                    G.append(g); P.append(p); S.append(g / p)
                allc += S
                dg, dp, ds = D[tier][mach][i]
                print(f"  {tier:4s} {mach} ({c},{t:>3}) alpha={a*1e3:8.1f}us cells={len(S)}"
                      f" | GPU {gm(G):7.1f} D {dg:7.1f} ({gm(G)/dg-1:+.2%})"
                      f" | PIM {gm(P):6.1f} D {dp:6.1f} ({gm(P)/dp-1:+.2%})"
                      f" | x {gm(S):5.2f} D {ds:5.2f} paper {PAPER_FIG8[tier][mach][i]}"
                      f" | min cell x {min(S):.2f}")
            print(f"  {tier:4s} {mach} all-cell geomean {gm(allc):.3f} (GEO in plot_bw_bars: {GEO[tier][mach]})")


def table4_row(LN, a):
    cells = suite(LN)
    r = {k: [] for k in ("gpu", "spu", "spu_dru", "cos", "full")}
    for L in cells:
        nic = (L["n"] + 2) * a
        sm0 = L["ntt_dev"]
        smd = L["ntt_dev"] * (1 - L["dru_share"])
        r["gpu"].append(L["gpu_dpf_g"] + sm0 + nic)
        r["spu"].append(L["spu"] + sm0 + nic)
        r["spu_dru"].append(L["spu"] + smd + nic)
        r["cos"].append(max(L["spu"], sm0, nic))
        r["full"].append(max(L["spu"], smd, nic))
    return {k: gm(v) for k, v in r.items()}, cells


def table4():
    print("=" * 100)
    print("TABLE IV (alpha = 500 us) -- best-of{standalone,fold} NTT, 25-cell suite (L40S 22)")
    for mach in ("L40S", "B200"):
        LN = L_(mach, ntt_mode="best", dru=(mach == "B200"))
        if mach == "L40S":
            for L in LN.values():
                L["dru_share"] = 0.0
        g, cells = table4_row(LN, 0.5)
        names = ["GPU baseline", "+SPU", "+SPU+DRU", "+SPU+co-sched", "full"]
        for nm, k, p in zip(names, ("gpu", "spu", "spu_dru", "cos", "full"), PAPER_T4[mach]):
            print(f"  {mach} {nm:14s} {g[k]:7.1f} ms (paper {p:6.1f}, {g[k]/p-1:+.2%})"
                  f"  vs base {g['gpu']/g[k]:.3f}x (paper {PAPER_T4[mach][0]/p:.3f}x)")
        print(f"  {mach} incremental: SPU {g['gpu']/g['spu']:.3f} DRU|serial {g['spu']/g['spu_dru']:.3f}"
              f" cosched {g['spu']/g['cos']:.3f} DRU|cosched {g['cos']/g['full']:.3f}  (ncells={len(cells)})")
    print("  --- same model with the Fig. 8 NTT (standalone brev) = plot_dru_pipe.py E2E table")
    for mach in ("L40S", "B200"):
        LN = L_(mach)
        if mach == "L40S":
            for L in LN.values():
                L["dru_share"] = 0.0
        for i, a in enumerate((0.5, 2.0, 10.0, 30.0)):
            g, cells = table4_row(LN, a)
            v = (g["spu"], g["spu_dru"], g["cos"], g["full"])
            sb = sum(1 for L in cells if L["ntt_dev"] * (1 - L["dru_share"]) >
                     max(L["spu"], (L["n"] + 2) * a))
            print(f"  {mach} a={a*1e3:6.0f}us  " + " ".join(f"{x:7.1f}" for x in v) +
                  f"   plot_dru_pipe {DRUPIPE[mach][i]}  GPU {g['gpu']:.1f}"
                  f"  SM-bound {sb} (plot_dru_pipe {SMBOUND[mach][i]})")


def fig10():
    print("=" * 100)
    print("FIG. 10 (plot_spu_net_util.py) -- SM lane ignored; serial = SPU+NIC, co-sched = max(SPU,NIC)")
    ALPHA = {"L40S": [5.0, 10.0, 15.0, 20.0], "B200": [0.5, 1.5, 2.5, 3.5]}
    UTIL = {"L40S": [(64.6, 27.6, 92.7, 39.7), (49.1, 42.0, 74.2, 63.4), (39.9, 51.2, 59.6, 76.5), (33.8, 57.8, 50.1, 85.6)],
            "B200": [(71.8, 20.5, 96.7, 27.6), (48.3, 41.3, 71.0, 60.7), (37.1, 52.8, 54.4, 77.5), (30.2, 60.3, 43.3, 86.5)]}
    TPUT = {"L40S": [(9.65, 13.84), (7.33, 11.07), (5.96, 8.90), (5.04, 7.47)],
            "B200": [(83.93, 113.03), (56.49, 82.97), (43.31, 63.52), (35.32, 50.63)]}
    for mach in ("L40S", "B200"):
        C = suite(L_(mach))
        astar = gm(L["spu"] / (L["n"] + 2) for L in C)
        astar_sm = gm(max(L["spu"], L["sm"]) / (L["n"] + 2) for L in C)
        print(f"  {mach} alpha* = geomean(SPU/(n+2)) = {astar:.3f} ms (paper {11.69 if mach=='L40S' else 1.75});"
              f" incl. SM lane geomean(max(SPU,SM)/(n+2)) = {astar_sm:.3f} ms")
        for i, a in enumerate(ALPHA[mach]):
            us, un, uc, unc, ts, tc, tc3 = [], [], [], [], [], [], []
            for L in C:
                nic = (L["n"] + 2) * a
                ws, wc = L["spu"] + nic, max(L["spu"], nic)
                w3s, w3c = L["spu"] + L["sm"] + nic, max(L["spu"], L["sm"], nic)
                us.append(L["spu"] / ws); un.append(nic / ws)
                uc.append(L["spu"] / wc); unc.append(nic / wc)
                N = 1 << L["lg"]
                ts.append(N / ws / 1e3); tc.append(N / wc / 1e3); tc3.append(N / w3c / 1e3)
            u = tuple(round(100 * gm(v), 1) for v in (us, un, uc, unc))
            print(f"   a={a:4.1f}ms util {u} paper {UTIL[mach][i]} | tput {gm(ts):6.2f}->{gm(tc):6.2f}"
                  f" x{gm(tc)/gm(ts):.3f} paper {TPUT[mach][i]} x{TPUT[mach][i][1]/TPUT[mach][i][0]:.3f}"
                  f" | co-sched tput with SM lane in max(): {gm(tc3):6.2f}")


if __name__ == "__main__":
    fig8()
    table4()
    fig10()
