#!/usr/bin/env python3
"""System energy on L40S (reviewer D, Q14): measured GPU board power per phase (run_power_l40s.sh)
times the lane times of every Fig. 8 cell; PIM power from the paper's synthesis (192 SPUs = 2.87 W),
charged for the WHOLE PCG^2 run (conservative).
  baseline  E = P_dpf(c,t) T_dpf + P_ntt(logN, c^2) T_ntt + P_idle T_net        (network serial, Fig. 8 accounting)
  PCG^2     E = P_ntt T_ntt' + P_idle (T_pcg - T_ntt') + P_pim T_pcg           (the GPU runs only the NTT lane)
T_dpf = single-hash GPU DPF, T_ntt = merge NTT, T_ntt' = the NTT lane inside PCG^2 (incl. gather and its co-run
slowdown), T_pcg = co-simulated PCG^2 time (overlap_simlane, SPU at the DRAM clock, no DRU on L40S).
Usage: power_energy_l40s.py POWER_DIR OUT_FILE"""
import csv, io, contextlib, json, os, re, sys
from datetime import datetime
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
P_PIM = 2.87                                   # W, 192 SPUs x 14.95 mW (Sec. VI-C)
pdir, out = sys.argv[1], sys.argv[2]; RUN = sys.argv[3] if len(sys.argv) > 3 else "overlap_simlane"
ts = lambda s: datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
trace = [(ts(r[0]), float(r[1].replace(" W", ""))) for r in csv.reader(open(os.path.join(pdir, "power_trace.csv"))) if len(r) >= 2 and "W" in r[1]]
P = {}
for r in csv.DictReader(open(os.path.join(pdir, "phases.csv"))):
    s, e = float(r["start"]) + 1.0, float(r["end"]) - 0.5
    v = [p for t, p in trace if s <= t <= e]
    P[r["phase"]] = sum(v) / len(v)
lines = ["Measured L40S board power (W, mean over each phase):"] + [f"  {k:22s} {v:6.1f}" for k, v in P.items()]
P_idle = P["idle_context"]
def p_ntt(lg, b):   # nearest/interpolated measured logN
    lo, hi = max(x for x in (20, 22, 24) if x <= lg), min(x for x in (20, 22, 24) if x >= lg)
    a, c = P[f"ntt_merge_lg{lo}_b{b}"], P[f"ntt_merge_lg{hi}_b{b}"]
    return a if lo == hi else a + (c - a) * (lg - lo) / (hi - lo)
new_l = {}
for l in open(os.path.join(REPO, "GPU_baseline/results_l40s/e2e_single_hash_all_arms_L40S.txt")):
    m = re.match(r"c=(\d+) t=(\d+) logN=(\d+) arm=serial : expand=([\d.]+) convert=([\d.]+) beaver=([\d.]+)", l)
    if m: new_l[(int(m[1]), int(m[2]), int(m[3]))] = (float(m[4]) + float(m[5]) + float(m[6])) * 1.026
LN = L_("L40S")
for clk, tag in (("SPU = DRAM clock", "nom"), ("SPU 1 GHz", "1.0")):
    R = json.load(open(os.path.join(REPO, f"PIM/results/contention/final_window/{RUN}/e2e_{tag}.json")))
    for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
        per, cells, perv, cellsv, pw = [], [], [], [], []
        for (c, t) in CFG:
            v, vv = [], []
            for r in R:
                if not (r["org"] == "l40s" and r["design"] == "merge" and r["tier"] == tier and r["c"] == c and r["t"] == t): continue
                k = (c, t, r["logN"])
                if k not in new_l: continue
                T_dpf, T_ntt, T_net = new_l[k], r["gpu_ms"] - LN[k]["gpu_dpf_g"], r["nic_ms"]
                T_pcg, T_nttp = r["pcg_ms"], r["ntt_ms"] * r["ntt_slow"]
                Pn = p_ntt(r["logN"], c * c); Pd = P[f"dpf_c{c}t{t}"]
                E_base = Pd * T_dpf + Pn * T_ntt + P_idle * T_net
                E_basev = Pd * T_dpf + Pn * T_ntt + P_idle * max(0.0, T_net - T_dpf - T_ntt)   # GPU baseline overlapping the network
                E_pcg = Pn * T_nttp + P_idle * (T_pcg - T_nttp) + P_PIM * T_pcg
                v.append(E_base / E_pcg); vv.append(E_basev / E_pcg); pw.append(E_pcg / T_pcg)
            per.append(gm(v)); cells += v; perv.append(gm(vv)); cellsv += vv
        lines.append(f"{clk:16s} {lab:8s}: energy ratio baseline/PCG^2 per (c,t) " + " ".join(f"{x:.2f}" for x in per)
                     + f" | up to {max(per):.2f} geomean {gm(cells):.2f} | vs network-overlapping baseline {max(perv):.2f}/{gm(cellsv):.2f}"
                     + f" | PCG^2 mean system power {sum(pw)/len(pw):.0f} W")
open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
