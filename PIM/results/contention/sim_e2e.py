#!/usr/bin/env python3
"""Fig. 8 with the co-simulated overlap penalty: PCG^2 = max(max(spu,sm)*f(r), nic),
r = min(spu,sm)/max(spu,sm), f = 'both done vs max()' from the contention sweep,
linearly interpolated in r (r < 0.12 -> f(0.12) scaled toward 1 at r=0)."""
import io, contextlib, sys, json
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, LOGN, alpha_bw, gm
    from fig8 import BETA
def interp(pts, r):
    pts = [(0.0, 1.0)] + sorted(pts)
    if r >= pts[-1][0]: return pts[-1][1]
    for (r0, f0), (r1, f1) in zip(pts, pts[1:]):
        if r0 <= r <= r1: return f0 + (f1 - f0) * (r - r0) / (r1 - r0)
def run(mach, tier, pts):
    LN = L_(mach); per = []; cells = []; rs = []
    for (c, t) in CFG:
        a = alpha_bw(c, t, BETA[tier]); S = []
        for lg in LOGN:
            L = LN.get((c, t, lg))
            if not L: continue
            nic = (L["n"] + 2) * a; g = L["gpu_dpf_g"] + L["ntt_dev"] + nic
            hi, lo = max(L["spu"], L["sm"]), min(L["spu"], L["sm"])
            f = interp(pts, lo / hi) if pts else 1.0; rs.append(lo / hi)
            S.append(g / max(hi * f, nic))
        per.append(gm(S)); cells += S
    return max(per), gm(cells), (min(rs), max(rs))
if __name__ == "__main__":
    mach = sys.argv[1]; pts = json.loads(sys.argv[2])
    for tier in ("fast", "slow"):
        b0, g0, rr = run(mach, tier, None); b, g, _ = run(mach, tier, pts)
        print(f"{mach} {tier}: r range {rr[0]:.2f}-{rr[1]:.2f} | max() {b0:.2f}/{g0:.2f} -> co-sim {b:.2f}/{g:.2f}")
