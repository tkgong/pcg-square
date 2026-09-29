#!/usr/bin/env python3
"""Fig. 8 vs SPU clock with the contention factor re-simulated at each clock.
SPU lane scaled by T_spu(f)/T_spu(nominal); f(r) = 'both done vs max()' of
config (c)/fair from that clock's sweep."""
import io, contextlib, json, os, sys
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, LOGN, alpha_bw, gm
    from fig8 import BETA
    from sim_e2e import interp
C = sys.argv[1] if len(sys.argv) > 1 else "../cont"
RUNS = {"L40S": [(2.25, "v4_l40s"), (1.5, "v5_l40s_f1.5"), (1.0, "v5_l40s_f1.0"), (0.75, "v5_l40s_f0.75"), (0.5, "v5_l40s_f0.5")],
        "B200": [(2.0, "v4_b200"), (1.5, "v5_b200_f1.5"), (1.0, "v5_b200_f1.0"), (0.75, "v5_b200_f0.75"), (0.5, "v5_b200_f0.5")]}
def load(d):
    S = json.load(open(os.path.join(C, d, "summary.json"))); T = json.load(open(os.path.join(C, d, "results.json")))["T_spu"]
    return T, [[float(r), float(S[r]["(c) GPU+SPU+DRU|fair"]["win_vs_max"])] for r in S]
def run(mach, tier, sc, pts):
    LN = L_(mach); per = []; cells = []
    for (c, t) in CFG:
        a = alpha_bw(c, t, BETA[tier]); S = []
        for lg in LOGN:
            L = LN.get((c, t, lg))
            if not L: continue
            nic = (L["n"] + 2) * a; g = L["gpu_dpf_g"] + L["ntt_dev"] + nic
            spu = L["spu"] * sc; hi, lo = max(spu, L["sm"]), min(spu, L["sm"])
            S.append(g / max(hi * (interp(pts, lo / hi) if pts else 1.0), nic))
        per.append(gm(S)); cells += S
    return max(per), gm(cells)
for tier in ("fast", "slow"):
    for mach, runs in RUNS.items():
        T0 = load(runs[0][1])[0]
        for f, d in runs:
            T, pts = load(d); sc = T / T0
            b0, g0 = run(mach, tier, sc, None); b, g = run(mach, tier, sc, pts)
            print(f"{tier} {mach} {f:4.2f}GHz spu x{sc:.2f} | no contention {b0:.2f}/{g0:.2f} | contention {b:.2f}/{g:.2f}")
