#!/usr/bin/env python3
"""Fig. 8 headline with the GPU baseline's leaf conversion computing H' ONCE (CONV_V2) instead of
twice. PCG^2 from the co-simulation (overlap_simlane), baseline = GPU DPF + merge NTT + network.
  L40S: results_l40s/e2e_single_hash_all_arms_L40S.txt (serial arm, alpha 0), x1.026 as in lanes.py
  B200: results_b200/single_hash_ab/e2e_single_hash_ab.csv (job 1160453): per-cell ratio
        (expand+convert single-hash / two-pass, serial arm, rtt 5 us) applied to the paper's GPU DPF
        lane; the A/B's own two-pass numbers are printed against the paper's lane as a check.
Usage: headline_single_hash.py OUT_FILE"""
import csv, io, contextlib, json, os, re, sys
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
R = os.path.join(REPO, "PIM/results/contention/final_window/overlap_simlane")
GB = os.path.join(REPO, "GPU_baseline")
new_l = {}
for l in open(os.path.join(GB, "results_l40s/e2e_single_hash_all_arms_L40S.txt")):
    m = re.match(r"c=(\d+) t=(\d+) logN=(\d+) arm=serial : expand=([\d.]+) convert=([\d.]+) beaver=([\d.]+)", l)
    if m: new_l[(int(m[1]), int(m[2]), int(m[3]))] = (float(m[4]) + float(m[5]) + float(m[6])) * 1.026
ab = {}
for r in csv.DictReader(open(os.path.join(GB, "results_b200/single_hash_ab/e2e_single_hash_ab.csv"))):
    if r["status"] == "OK" and r["arm"] == "serial" and r["rtt_us"] == "5":
        ab.setdefault((int(r["c"]), int(r["t"]), int(r["logN"])), {})[r["mode"]] = float(r["expand_ms"]) + float(r["convert_ms"])
LN = {"L40S": L_("L40S"), "B200": L_("B200")}
out = []
chk = [ab[k]["twopass"] * 1.026 / LN["B200"][k]["gpu_dpf_g"] for k in ab if k in LN["B200"]]
out.append(f"B200 A/B two-pass (expand+convert, x1.026) / paper GPU DPF lane: geomean {gm(chk):.3f} (min {min(chk):.3f} max {max(chk):.3f})")
rat = [ab[k]["singlehash"] / ab[k]["twopass"] for k in ab]
out.append(f"B200 single-hash / two-pass GPU DPF: geomean {gm(rat):.3f} (min {min(rat):.3f} max {max(rat):.3f}); L40S: "
           f"{gm(new_l[k] / LN['L40S'][k]['gpu_dpf_g'] for k in new_l if k in LN['L40S']):.3f}")
for mach, org, des in (("L40S", "l40s", "merge"), ("B200", "b200", "f4dru")):
    for clk, tag in (("DRAM clk", "nom"), ("1 GHz", "1.0")):
        Rj = json.load(open(f"{R}/e2e_{tag}.json"))
        for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
            po, pn, pv, co, cn, cv = [], [], [], [], [], []
            for (c, t) in CFG:
                so, sn, sv = [], [], []
                for r in Rj:
                    if not (r["org"] == org and r["design"] == des and r["tier"] == tier and r["c"] == c and r["t"] == t): continue
                    k = (c, t, r["logN"]); L = LN[mach][k]; ntt = r["gpu_ms"] - L["gpu_dpf_g"]
                    if mach == "L40S":
                        if k not in new_l: continue
                        dpf = new_l[k]
                    else:
                        dpf = L["gpu_dpf_g"] * ab[k]["singlehash"] / ab[k]["twopass"]
                    so.append(r["base_serial_ms"] / r["pcg_ms"]); sn.append((dpf + ntt + r["nic_ms"]) / r["pcg_ms"])
                    sv.append(max(dpf + ntt, r["nic_ms"]) / r["pcg_ms"])          # GPU-only baseline overlapping the network
                po.append(gm(so)); pn.append(gm(sn)); pv.append(gm(sv)); co += so; cn += sn; cv += sv
            out.append(f"{mach} SPU {clk:8s} {lab:8s}: two-pass H' {max(po):.2f}/{gm(co):.2f} -> single-hash H' {max(pn):.2f}/{gm(cn):.2f}"
                       f" (overlap-net base {max(pv):.2f}/{gm(cv):.2f})   per (c,t): " + " ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(po, pn)))
open(sys.argv[1], "w").write("\n".join(out) + "\n"); print("\n".join(out))
