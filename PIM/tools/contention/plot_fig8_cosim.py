#!/usr/bin/env python3
"""Fig. 8 of the paper, redrawn from the end-to-end co-simulation (e2e_cosim.py).
Same plotting code (PIM/tools/plot_bw_bars.py): bars = GPU baseline and PCG^2 runtime
(geomean over N = 2^20..2^24, hatched if fewer than five N), lines = speedup, title =
geomean speedup over all cells. L40S without DRU, B200 with DRU.

    plot_fig8_cosim.py RUN_DIR BASELINE OUT      BASELINE = sota | paper
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


def data(run_dir, baseline):
    R = json.load(open(os.path.join(run_dir, "e2e.json")))
    LN = {"L40S": L_("L40S"), "B200": L_("B200")}
    D, GEO = {}, {}
    for tier in ("fast", "slow"):
        D[tier], GEO[tier] = {}, {}
        for M, org, des in (("L40S", "l40s", "merge"), ("B200", "b200", "f4dru")):
            cells, allsp = [], []
            for (c, t) in P.CFG:
                rr = [r for r in R if r["org"] == org and r["design"] == des and r["tier"] == tier and r["c"] == c and r["t"] == t]
                # e2e_window.py rows carry pcg_ms / base_serial_ms; e2e_cosim.py rows both_ms / base_ms
                pcg = [r["pcg_ms"] if "pcg_ms" in r else max(r["both_ms"], r["nic_ms"]) for r in rr]
                if "pcg_ms" in rr[0]:
                    base = [r["base_paper_ms"] if baseline == "paper" else
                            (r["base_overlap_ms"] if baseline == "overlap" else r["base_serial_ms"]) for r in rr]
                elif baseline == "paper":
                    base = [LN[M][(c, t, r["logN"])]["gpu_dpf_g"] + LN[M][(c, t, r["logN"])]["ntt_dev"] + r["nic_ms"] for r in rr]
                else:
                    base = [r["base_ms"] for r in rr]
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
    run_dir, baseline, out = sys.argv[1], sys.argv[2], sys.argv[3]
    P.D, P.GEO = data(run_dir, baseline)
    _patch_labels()          # after D/GEO are set: the patched main reads them from its copied namespace
    for tier in ("fast", "slow"):
        for M in ("L40S", "B200"):
            print(tier, M, P.D[tier][M], f"geo {P.GEO[tier][M]:.2f}")
    P.main(out)
