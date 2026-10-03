#!/usr/bin/env python3
"""Table IV for the final design: four-step NTT on GPU-NTT kernels with the DRU (f4dru) on both machines.
Rows (geomean ms over the suite): GPU baseline (single-hash DPF + merge NTT + network, serial) / +SPU (expansion
on the SPUs, merge NTT on the SMs, serial) / +SPU+DRU (four-step + DRU, serial) / +SPU+co-sched (max of the
contended lanes, merge) / full (max of the contended lanes, four-step + DRU). alpha = 500 us as in the paper,
plus the Fig. 8 tiers.   Usage: table4_final.py RUN_DIR(win22)"""
import csv, io, contextlib, json, os, re, sys
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
    from lanes import alpha_bw
    from fig8 import BETA
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
new_l = {}
for l in open(os.path.join(REPO, "GPU_baseline/results_l40s/e2e_single_hash_all_arms_L40S.txt")):
    m = re.match(r"c=(\d+) t=(\d+) logN=(\d+) arm=serial : expand=([\d.]+) convert=([\d.]+) beaver=([\d.]+)", l)
    if m: new_l[(int(m[1]), int(m[2]), int(m[3]))] = (float(m[4]) + float(m[5]) + float(m[6])) * 1.026
ab = {}
for r in csv.DictReader(open(os.path.join(REPO, "GPU_baseline/results_b200/single_hash_ab/e2e_single_hash_ab.csv"))):
    if r["status"] == "OK" and r["arm"] == "serial" and r["rtt_us"] == "5":
        ab.setdefault((int(r["c"]), int(r["t"]), int(r["logN"])), {})[r["mode"]] = float(r["expand_ms"]) + float(r["convert_ms"])
LN = {"L40S": L_("L40S"), "B200": L_("B200")}
def dpf(mach, k): return new_l.get(k) if mach == "L40S" else LN["B200"][k]["gpu_dpf_g"] * ab[k]["singlehash"] / ab[k]["twopass"]
W = {(r["org"], r["design"], r["tier"], r["c"], r["t"], r["logN"]): r for r in json.load(open(os.path.join(sys.argv[1], "e2e_nom.json")))}
for mach, org in (("L40S", "l40s"), ("B200", "b200")):
    for alab, amode in (("alpha = 500 us (Table IV)", 500), ("40 Gbps", "fast"), ("400 Mbps", "slow")):
        rows = {k: [] for k in ("GPU baseline", "+SPU", "+SPU+DRU", "+SPU+co-sched", "full")}
        for (c, t) in CFG:
            for lg in range(20, 25):
                k = (c, t, lg); key = (org, "merge", "fast", c, t, lg)
                if key not in W or k not in LN[mach] or dpf(mach, k) is None: continue
                L = LN[mach][k]; m = W[key]; f = W[(org, "f4dru", "fast", c, t, lg)]
                nic = (L["n"] + 2) * (0.5 if amode == 500 else alpha_bw(c, t, BETA[amode]))
                rows["GPU baseline"].append(dpf(mach, k) + (m["gpu_ms"] - L["gpu_dpf_g"]) + nic)
                rows["+SPU"].append(m["spu_ms"] + m["ntt_ms"] + nic)
                rows["+SPU+DRU"].append(f["spu_ms"] + f["ntt_ms"] + nic)
                rows["+SPU+co-sched"].append(max(m["spu_ms"] * m["spu_slow"], m["ntt_ms"] * m["ntt_slow"], nic))
                rows["full"].append(max(f["spu_ms"] * f["spu_slow"], f["ntt_ms"] * f["ntt_slow"], nic))
        g = {k: gm(v) for k, v in rows.items()}
        print(f"{mach}, {alab}, {len(rows['full'])} cells, geomean ms (speedup vs GPU baseline):")
        for k, v in g.items(): print(f"   {k:16s} {v:8.1f} ms   {g['GPU baseline']/v:6.2f}x")
        print(f"   increments: SPU {g['GPU baseline']/g['+SPU']:.2f}, DRU serial {g['+SPU']/g['+SPU+DRU']:.3f}, co-scheduling {g['+SPU']/g['+SPU+co-sched']:.2f}, DRU co-scheduled {g['+SPU+co-sched']/g['full']:.3f}")
