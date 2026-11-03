#!/usr/bin/env python3
"""Same-run comparison of the three g_ij input layouts of the PCG-output NTT (design ole = Cheddar's natural layout,
ole_t = column-contiguous as the SPUs write it, ole_r = the first column kernel's per-thread register order, written by
the SPUs and loaded with one vectorised move per thread).  Per (logN, batch): best valid config per layout, the ratio of
the bests, and the geometric mean over the configs all three share of ole/ole_t and ole/ole_r (same config -> pure layout
effect).  Usage: reg_layout_compare.py <csv> [<csv> ...]"""
import csv, math, sys
from collections import defaultdict
for path in sys.argv[1:]:
    R = [r for r in csv.DictReader(l for l in open(path) if not l.startswith("#")) if r["design"] in ("ole", "ole_t", "ole_r") and r["mismatches"] == "0"]
    T = defaultdict(dict)   # (logN, batch, cfg) -> design -> ms
    for r in R:
        T[(int(r["logN"]), int(r["batch"]), (r["P1"], r["P2"], r["M1"], r["B1"], r["M2"], r.get("maxrreg", "0")))][r["design"]] = float(r["ms_per_poly"])
    cells = sorted({k[:2] for k in T})
    print(f"{path}: {len(R)} valid rows, {len(T)} configs")
    print("logN batch | best ole | best ole_t | best ole_r | ole/ole_t | ole/ole_r | same-config geomean ole/ole_t | ole/ole_r | ole_t/ole_r | n configs")
    gt, gr, n = [], [], 0
    for lg, b in cells:
        best = {d: min(v[d] for k, v in T.items() if k[:2] == (lg, b) and d in v) for d in ("ole", "ole_t", "ole_r")}
        sh = [v for k, v in T.items() if k[:2] == (lg, b) and all(d in v for d in ("ole", "ole_t", "ole_r"))]
        gm = lambda a, c: math.exp(sum(math.log(v[a] / v[c]) for v in sh) / len(sh))
        gt += [math.log(v["ole"] / v["ole_t"]) for v in sh]; gr += [math.log(v["ole"] / v["ole_r"]) for v in sh]; n += len(sh)
        print(f"  {lg} {b:3d} | {best['ole']:.4f} | {best['ole_t']:.4f} | {best['ole_r']:.4f} | {best['ole']/best['ole_t']:.3f} | {best['ole']/best['ole_r']:.3f} | {gm('ole','ole_t'):.3f} | {gm('ole','ole_r'):.3f} | {gm('ole_t','ole_r'):.3f} | {len(sh)}")
    print(f"  all configs: geomean ole/ole_t {math.exp(sum(gt)/n):.3f}, ole/ole_r {math.exp(sum(gr)/n):.3f} over {n} configs\n")
