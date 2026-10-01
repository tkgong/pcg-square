#!/usr/bin/env python3
"""Window-size invariance of the co-sim (e2e_window.py --win-inst/--win-depth): headline per run and
per-cell changes vs the reference window.  Usage: compare_windows.py REF_DIR RUN_DIR..."""
import json, sys
from math import exp, log
def gm(v): v = list(v); return exp(sum(map(log, v)) / len(v))
CFG = [(8, 4), (4, 16), (8, 8), (2, 64), (2, 128)]
ref = sys.argv[1]; runs = sys.argv[2:]
def load(d): return {(r["org"], r["design"], r["tier"], r["c"], r["t"], r["logN"]): r for r in json.load(open(f"{d}/e2e.json"))}
R0 = load(ref)
def headline(R, org, des, tier):
    per = []; allv = []
    for c, t in CFG:
        v = [r["base_serial_ms"] / r["pcg_ms"] for k, r in R.items() if k[:5] == (org, des, tier, c, t)]
        per.append(gm(v)); allv += v
    return max(per), gm(allv)
for d in [ref] + runs:
    R = load(d)
    s = []
    for org, des in (("l40s", "merge"), ("b200", "f4dru")):
        for tier in ("fast", "slow"):
            b, g = headline(R, org, des, tier); s.append(f"{org} {tier} {b:5.2f}/{g:5.2f}")
    print(f"{d:16s} " + " | ".join(s))
for d in runs:
    R = load(d)
    for org, des in (("l40s", "merge"), ("b200", "f4dru")):
        ks = [k for k in R0 if k[0] == org and k[1] == des and k[2] == "fast" and k in R]
        ds = [abs(R[k]["spu_slow"] / R0[k]["spu_slow"] - 1) for k in ks]
        dn = [abs(R[k]["ntt_slow"] / R0[k]["ntt_slow"] - 1) for k in ks]
        dl = [abs(R[k]["spu_ms"] / R0[k]["spu_ms"] - 1) for k in ks]
        dp = [abs(R[k]["pcg_ms"] / R0[k]["pcg_ms"] - 1) for k in ks]
        print(f"  {d} vs {ref} {org}: |dSPU lane| max {max(dl)*100:.2f}%  |dSPU slow| max {max(ds)*100:.2f}%  "
              f"|dNTT slow| max {max(dn)*100:.2f}% mean {sum(dn)/len(dn)*100:.2f}%  |dPCG2| max {max(dp)*100:.2f}% mean {sum(dp)/len(dp)*100:.2f}%  ({len(ks)} cells)")
