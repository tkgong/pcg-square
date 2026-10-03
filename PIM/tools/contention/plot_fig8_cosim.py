#!/usr/bin/env python3
"""Fig. 8 of the paper, redrawn from the end-to-end co-simulation (e2e_window.py rows).
Same plotting code (PIM/tools/plot_bw_bars.py): bars = GPU baseline and PCG^2 runtime
(geomean over N = 2^20..2^24, hatched if fewer than five N), lines = speedup, title =
geomean speedup over all cells. B200 with the DRU, L40S with merge (runtime choice). GPU baseline = the
submission's DPF + merge NTT with the network co-scheduled: max(T_GPU, T_net).

    plot_fig8_cosim.py E2E_JSON OUT
"""
import io, contextlib, json, os, sys
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "e2e")); sys.path.insert(0, os.path.dirname(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_
import plot_bw_bars as P


def gm(v):
    v = list(v); return exp(sum(map(log, v)) / len(v))


def data(path):
    R = json.load(open(path))
    D, GEO = {}, {}
    for tier in ("fast", "slow"):
        D[tier], GEO[tier] = {}, {}
        for M, org, des in (("L40S", "l40s", "merge"), ("B200", "b200", "f4dru")):
            cells, allsp = [], []
            for (c, t) in P.CFG:
                rr = [r for r in R if r["org"] == org and r["design"] == des and r["tier"] == tier and r["c"] == c and r["t"] == t]
                pcg = [r["pcg_ms"] for r in rr]
                base = [max(r["gpu_ms"], r["nic_ms"]) for r in rr]
                sp = [b / p for b, p in zip(base, pcg)]
                cells.append((round(gm(base), 1), round(gm(pcg), 1), round(gm(sp), 2)))
                allsp += sp
            D[tier][M] = cells
            GEO[tier][M] = gm(allsp)
    return D, GEO


def _patch_labels():
    """Same figure as plot_bw_bars.main; only the speedup labels get a white backing so
    low L40S values (which sit over the tall baseline bars) stay readable."""
    import inspect
    src = inspect.getsource(P.main).replace(
        'textcoords="offset points",',
        'textcoords="offset points", bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85),')
    ns = dict(vars(P)); exec(src, ns); P.main = ns["main"]


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    P.D, P.GEO = data(path)
    _patch_labels()          # after D/GEO are set: the patched main reads them from its copied namespace
    for tier in ("fast", "slow"):
        for M in ("L40S", "B200"):
            print(tier, M, P.D[tier][M], f"geo {P.GEO[tier][M]:.2f}")
    P.main(out)
