#!/usr/bin/env python3
"""Fig. 8 with the paper's own end-to-end method (no contention, paper SPU anchor), GPU
baseline NTT swapped from the square four-step to the GPU-NTT merge kernel (SOTA):
    baseline = GPU DPF + merge NTT + (n+2) alpha
    PCG^2    = max(SPU, SM, (n+2) alpha)
SM lane on the same GPU-NTT kernels: B200 = four-step with the transposes on the DRU
(f4g_sm, measured), L40S = merge (no DRU, as in the paper). Same plotting code as the paper.

    fig8_paper_sota.py OUT_DIR
"""
import io, contextlib, os, sys
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "e2e")); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_
    from lanes import alpha_bw
    from fig8 import BETA
from ntt_jobs import ntt_table
import plot_fig8_cosim as F
P = F.P


def gm(v):
    v = list(v); return exp(sum(map(log, v)) / len(v))


def cells(M, tier, base_ntt, pcg_sm):
    """per (c,t): lists of (baseline ms, PCG^2 ms) over logN"""
    LN, NT = L_(M), ntt_table(M.lower())
    out = []
    for (c, t) in P.CFG:
        bp = []
        for lg in P.LOGN:
            L = LN.get((c, t, lg))
            if not L or (lg, c * c) not in NT: continue
            nic = (L["n"] + 2) * alpha_bw(c, t, BETA[tier])
            mm, fs = NT[(lg, c * c)]
            base = L["gpu_dpf_g"] + (mm * 2 * c * c if base_ntt == "merge" else L["ntt_dev"]) + nic
            sm = L["sm"] if pcg_sm == "paper" else (fs if M == "B200" else mm) * 2 * c * c
            bp.append((base, max(L["spu"], sm, nic)))
        out.append(bp)
    return out


def data(base_ntt, pcg_sm):
    D, GEO = {}, {}
    for tier in ("fast", "slow"):
        D[tier], GEO[tier] = {}, {}
        for M in ("L40S", "B200"):
            cs = cells(M, tier, base_ntt, pcg_sm)
            D[tier][M] = [(round(gm(b for b, _ in bp), 1), round(gm(p for _, p in bp), 1), round(gm(b / p for b, p in bp), 2)) for bp in cs]
            GEO[tier][M] = gm(b / p for bp in cs for b, p in bp)
    return D, GEO


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    lines = []
    for lab, bn, sm in (("paper as published: square-NTT baseline, paper PCG^2", "square", "paper"),
                        ("SOTA baseline (GPU DPF + merge NTT), PCG^2 SM lane on GPU-NTT kernels (B200 + DRU)", "merge", "sota")):
        D, GEO = data(bn, sm)
        lines.append(lab)
        for tier, tl in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
            for M in ("L40S", "B200"):
                sp = [x[2] for x in D[tier][M]]
                lines.append(f"  {tl:8s} {M}: " + " ".join(f"{s:5.2f}" for s in sp) + f" | up to {max(sp):.2f}  geomean {GEO[tier][M]:.2f}")
    lines.append("(c,t) = " + " ".join(f"({c},{t})" for c, t in P.CFG) + "; up to = best (c,t) geomean over N, geomean = all cells")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    P.D, P.GEO = data("merge", "sota")
    F._patch_labels()
    for ext in ("pdf", "png"):
        P.main(os.path.join(out, f"fig8_papermethod_sotabase.{ext}"))
