#!/usr/bin/env python3
"""Table IV from the co-simulation, DRU enabled on BOTH machines: GPU baseline / +SPU / +SPU+DRU / +SPU+co-sched
/ full, geomean ms over the suite, alpha = 500 us per exchange as in the paper (and the Fig. 8 tiers).
PCG^2's NTT = the submission's four-step (sq on the SMs; sqdru* with transposes+brev on the DRU); GPU baseline =
single-hash DPF + merge NTT (SOTA) [paper's square NTT in brackets]. DRU model: conservative (pays ACT/PRE) [ideal tap].
Usage: table4.py RUN_IDEAL RUN_CONSERVATIVE"""
import csv, io, contextlib, json, re, sys, os
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
def load(p): return {(r["org"], r["design"], r["tier"], r["c"], r["t"], r["logN"]): r for r in json.load(open(p))}
I, C = load(os.path.join(sys.argv[1], "e2e.json")), load(os.path.join(sys.argv[2], "e2e.json"))
for mach, org in (("L40S", "l40s"), ("B200", "b200")):
    for alab, amode in (("alpha = 500 us (Table IV)", 500), ("40 Gbps (Fig. 8 tier)", "fast"), ("400 Mbps", "slow")):
        rows = {k: [] for k in ("GPU (SOTA merge NTT)", "GPU (paper square NTT)", "+SPU", "+SPU+DRU", "+SPU+DRU ideal", "+SPU+co-sched", "full", "full ideal")}
        for (c, t) in CFG:
            for lg in range(20, 25):
                k = (c, t, lg); key = (org, "sq", "fast", c, t, lg)
                if key not in C or k not in LN[mach] or dpf(mach, k) is None: continue
                L = LN[mach][k]; a = C[key]; h = C[(org, "sqdruh", "fast", c, t, lg)]; i = I[(org, "sqdru", "fast", c, t, lg)]
                nic = (L["n"] + 2) * (0.5 if amode == 500 else alpha_bw(c, t, BETA[amode]))
                mrg = a["gpu_ms"] - L["gpu_dpf_g"]                       # merge NTT lane (measured)
                rows["GPU (SOTA merge NTT)"].append(dpf(mach, k) + mrg + nic)
                rows["GPU (paper square NTT)"].append(dpf(mach, k) + L["ntt_dev"] + nic)
                rows["+SPU"].append(a["spu_ms"] + a["ntt_ms"] + nic)
                rows["+SPU+DRU"].append(h["spu_ms"] + h["ntt_ms"] + nic); rows["+SPU+DRU ideal"].append(i["spu_ms"] + i["ntt_ms"] + nic)
                rows["+SPU+co-sched"].append(max(a["spu_ms"] * a["spu_slow"], a["ntt_ms"] * a["ntt_slow"], nic))
                rows["full"].append(max(h["spu_ms"] * h["spu_slow"], h["ntt_ms"] * h["ntt_slow"], nic))
                rows["full ideal"].append(max(i["spu_ms"] * i["spu_slow"], i["ntt_ms"] * i["ntt_slow"], nic))
        g = {k: gm(v) for k, v in rows.items()}; n = len(rows["+SPU"])
        print(f"{mach}, {alab}, {n} cells, geomean ms (speedup vs SOTA GPU baseline):")
        for k in ("GPU (SOTA merge NTT)", "GPU (paper square NTT)", "+SPU", "+SPU+DRU", "+SPU+co-sched", "full"):
            extra = f"   [ideal DRU {g['+SPU+DRU ideal']:.1f}]" if k == "+SPU+DRU" else (f"   [ideal DRU {g['full ideal']:.1f}]" if k == "full" else "")
            print(f"   {k:24s} {g[k]:8.1f} ms   {g['GPU (SOTA merge NTT)']/g[k]:6.2f}x{extra}")
        print(f"   DRU increment: serial {g['+SPU']/g['+SPU+DRU']:.3f} [ideal {g['+SPU']/g['+SPU+DRU ideal']:.3f}], co-scheduled {g['+SPU+co-sched']/g['full']:.3f} [ideal {g['+SPU+co-sched']/g['full ideal']:.3f}]; co-scheduling {g['+SPU+DRU']/g['full']:.3f}")
