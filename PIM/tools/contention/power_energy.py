#!/usr/bin/env python3
"""System energy (reviewer D): measured GPU board power per phase (run_power.sh) times the lane times of every
Fig. 8 cell, plus the synthesised PIM power (Table III per-unit power x count: L40S 192 SPUs + 24 DRUs = 5.91 W; B200 2,048 SPUs +
256 DRUs = 62.99 W) charged while the SPU lane runs, with the GPU's context-idle power charged to PCG^2 whenever its NTT lane is not running.
Same accounting as every other result: GPU baseline = the submission's DPF (two-pass H') + merge NTT with the
network co-scheduled (idle power only for the network time not hidden under compute).
  baseline  E = P_dpf(c,t) T_dpf + P_merge(logN, c^2) T_ntt + P_idle max(0, T_net - T_dpf - T_ntt)
  PCG^2     E = P_ntt' T_ntt' + P_idle (T_pcg - T_ntt') + P_pim T_spu'             (the GPU runs only its NTT lane)
One rule on both sides: every unit is charged its measured active power while it works and its idle power otherwise;
the GPU's idle power is the measured context-idle (both sides), the PIM's idle power is taken as zero (the synthesis
report gives no leakage split; charging the full PIM power over the whole run instead moves the ratios by <= 6%).
P_ntt' = merge on L40S (no DRU), the four-step SM lane f4g_sm on B200. Phase power = NVML energy counter / phase
time when the run recorded it, else the nvidia-smi mean over [start+1 s, end-0.5 s].
Usage: power_energy.py L40S|B200 POWER_DIR OUT_FILE [RUN=win22]"""
import csv, io, contextlib, json, os, re, sys
from datetime import datetime
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
mach, pdir, out = sys.argv[1], sys.argv[2], sys.argv[3]; RUN = sys.argv[4] if len(sys.argv) > 4 else "win22"
SPU_MW, DRU_MW = 30.574, 1.469                      # Table III: per-unit SPU and DRU power (mW)
NSPU, NDRU = {"L40S": 192, "B200": 2048}, {"L40S": 24, "B200": 256}   # 8 SPUs and 1 DRU per channel
org, des, pcg_ntt = {"L40S": ("l40s", "merge", "merge"), "B200": ("b200", "f4dru", "f4g_sm")}[mach]
P_PIM = (NSPU[mach] * SPU_MW + NDRU[mach] * DRU_MW) / 1e3
ts = lambda s: datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
trace = []
for r in csv.reader(open(os.path.join(pdir, "power_trace.csv"))):
    try: trace.append((ts(r[0]), float(r[1].replace(" W", ""))))
    except Exception: pass
P, src = {}, {}
for r in csv.DictReader(open(os.path.join(pdir, "phases.csv"))):
    s, e = float(r["start"]), float(r["end"])
    v = [p for t, p in trace if s + 1.0 <= t <= e - 0.5]
    smi = sum(v) / len(v) if v else float("nan")
    en = r.get("energy_mJ", "NA")
    if en not in ("NA", "", None): P[r["phase"]], src[r["phase"]] = float(en) / 1e3 / (e - s), "nvml"
    else: P[r["phase"]], src[r["phase"]] = smi, "smi"
    P[r["phase"] + "_smi"] = smi
lines = [f"{mach} board power per phase (W; nvml = energy counter / time, smi = nvidia-smi mean); PIM power {P_PIM:.2f} W:"]
lines += [f"  {k:22s} {v:6.1f}  ({src[k]}; smi {P[k + '_smi']:6.1f})" for k, v in P.items() if not k.endswith("_smi")]
P_idle = P["idle_context"]
def p_ntt(what, lg, b):
    lo, hi = max(x for x in (20, 22, 24) if x <= lg), min(x for x in (20, 22, 24) if x >= lg)
    a, c = P[f"ntt_{what}_lg{lo}_b{b}"], P[f"ntt_{what}_lg{hi}_b{b}"]
    return a if lo == hi else a + (c - a) * (lg - lo) / (hi - lo)
if not any(k.startswith("ntt_f4g_sm") for k in P): pcg_ntt = "merge"
LN = L_(mach)
dpf = lambda k: LN[k]["gpu_dpf_g"]
for clk, tag in (("SPU = DRAM clock", "nom"), ("SPU 1 GHz", "1.0")):
    R = json.load(open(os.path.join(REPO, f"PIM/results/contention/final_window/{RUN}/e2e_{tag}.json")))
    for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
        per, cells, perv, cellsv, pw, percell = [], [], [], [], [], []
        for (c, t) in CFG:
            v = []
            for r in R:
                if not (r["org"] == org and r["design"] == des and r["tier"] == tier and r["c"] == c and r["t"] == t): continue
                k = (c, t, r["logN"]); d = dpf(k)
                if d is None: continue
                T_ntt, T_net, T_pcg, T_nttp = r["gpu_ms"] - LN[k]["gpu_dpf_g"], r["nic_ms"], r["pcg_ms"], r["ntt_ms"] * r["ntt_slow"]
                Pm, Pp, Pd = p_ntt("merge", r["logN"], c * c), p_ntt(pcg_ntt, r["logN"], c * c), P[f"dpf_c{c}t{t}"]
                E_base = Pd * d + Pm * T_ntt + P_idle * max(0.0, T_net - d - T_ntt)
                T_spup = r["spu_ms"] * r["spu_slow"]                       # SPU lane with contention
                E_pcg = Pp * T_nttp + P_idle * (T_pcg - T_nttp) + P_PIM * T_spup
                v.append(E_base / E_pcg); pw.append(E_pcg / T_pcg); percell.append((c, t, r["logN"], E_base / E_pcg, E_base, E_pcg))
            per.append(gm(v)); cells += v
        lines.append(f"{clk:16s} {lab:8s}: energy ratio baseline/PCG^2 per (c,t) " + " ".join(f"{x:.2f}" for x in per)
                     + f" | up to {max(per):.2f} geomean {gm(cells):.2f} | PCG^2 mean system power {sum(pw)/len(pw):.0f} W")
        for (c, t) in CFG:
            pc = [x for x in percell if x[0] == c and x[1] == t]
            if pc: lines.append(f"    ({c},{t}): " + "  ".join(f"2^{lg} {ratio:.2f} ({eb/1e3:.1f}/{ep/1e3:.2f} J)" for _, _, lg, ratio, eb, ep in pc) + f"  | geomean {gm(x[3] for x in pc):.2f}")
open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
