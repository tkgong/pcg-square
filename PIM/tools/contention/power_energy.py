#!/usr/bin/env python3
"""System energy (reviewer D): measured GPU board power per phase (run_power.sh) times the lane times of every
Fig. 8 cell, plus the PIM power (Table III per-unit power x count: SPUs and reorder DRUs charged while the SPU lane runs;
the butterfly DRU's units, where the card's final NTT is f4b, charged while the NTT lane runs - its column passes run
beside the SM kernels, so this is an upper bound on their active time), with the GPU's context-idle power charged to PCG^2
whenever its NTT lane is not running. Same accounting as every other result (final_design.py): GPU baseline = the
submission's DPF (two-pass H') + Cheddar's NTT with the network co-scheduled (idle power only for the network time not
hidden under compute).
  baseline  E = P_dpf(c,t) T_dpf + P_cheddar(logN, c^2) T_ntt + P_idle max(0, T_net - T_dpf - T_ntt)
  PCG^2     E = P_ntt' T_ntt' + P_idle (T_pcg - T_ntt') + P_spu+rdru T_spu' + P_bf T_ntt'   (the GPU runs only its NTT lane)
One rule on both sides: every unit is charged its measured active power while it works and its idle power otherwise;
the GPU's idle power is the measured context-idle (both sides), the PIM's idle power is taken as zero (the synthesis
report gives no leakage split). P_ntt' = the measured board power of the final NTT's SM lane (phase ntt_<design>_lg*_b*:
cheddar / f4b_sm / f4c_sm / merge). Phase power = NVML energy counter / phase time when the run recorded it, else the
nvidia-smi mean over [start+1 s, end-0.5 s].
Writes OUT_FILE and, next to it, energy_<mach>.json (per-(c,t) geomean ratios per tier, read by the energy plots).
Usage: power_energy.py L40S|B200 POWER_DIR OUT_FILE [RUN_DIR=final_design.RUN_DIR]"""
import csv, io, contextlib, json, os, re, sys
from datetime import datetime
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e")); sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
import final_design as FD
mach, pdir, out = sys.argv[1], sys.argv[2], sys.argv[3]; RUN = sys.argv[4] if len(sys.argv) > 4 else FD.RUN_DIR
org = FD.ORG[mach]; des = FD.des(org, RUN)
PHASE = {"cheddar": "cheddar", "f4b": "f4b_sm", "f4c": "f4c_sm", "merge": "merge"}   # design -> power phase prefix (its SM lane)
P_SPU_RDRU = (FD.NSPU[org] * FD.spu_mw(org) + FD.NDRU[org] * FD.DRU_REORDER_MW * FD.CLK[org] / 2.25) / 1e3     # W, while the SPU lane runs
P_BF = (FD.NDRU[org] * (FD.dru_mw(org, RUN) - FD.DRU_REORDER_MW * FD.CLK[org] / 2.25)) / 1e3 if des == "f4b" else 0.0   # W, butterfly units while the NTT lane runs
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
lines = [f"{mach} board power per phase (W; nvml = energy counter / time, smi = nvidia-smi mean); PIM power while the SPU lane runs {P_SPU_RDRU:.2f} W"
         + (f", butterfly-DRU units while the NTT lane runs {P_BF:.2f} W ({FD.ESTIMATE_NOTE})" if P_BF else "") + f"; PCG^2 NTT = {des}, GPU baseline NTT = cheddar:"]
lines += [f"  {k:22s} {v:6.1f}  ({src[k]}; smi {P[k + '_smi']:6.1f})" for k, v in P.items() if not k.endswith("_smi")]
P_idle = P["idle_context"]
have = lambda what: any(k.startswith(f"ntt_{what}_") for k in P)
def p_ntt(what, lg, b):
    lo, hi = max(x for x in (20, 22, 24) if x <= lg), min(x for x in (20, 22, 24) if x >= lg)
    a, c = P[f"ntt_{what}_lg{lo}_b{b}"], P[f"ntt_{what}_lg{hi}_b{b}"]
    return a if lo == hi else a + (c - a) * (lg - lo) / (hi - lo)
base_ntt = "cheddar" if have("cheddar") else "merge"
pcg_ntt = PHASE[des] if have(PHASE[des]) else base_ntt
if base_ntt != "cheddar" or pcg_ntt != PHASE[des]:
    lines.append(f"  NOTE: power phases missing for {'cheddar' if base_ntt != 'cheddar' else ''} {PHASE[des] if pcg_ntt != PHASE[des] else ''}: using {base_ntt} / {pcg_ntt} phases instead (rerun GPU_baseline/run_power.sh after the Cheddar big-N sweep)")
LN = L_(mach)
dpf = lambda k: LN[k]["gpu_dpf_g"]
EJ = {}
for clk, tag in (("SPU = DRAM clock", "nom"), ("SPU 1 GHz", "1.0")):
    p = os.path.join(RUN, f"e2e_{tag}.json")
    if not os.path.exists(p): lines.append(f"{clk}: {p} not run"); continue
    R = json.load(open(p))
    for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
        per, cells, pw, percell = [], [], [], []
        for (c, t) in CFG:
            v = []
            for r in R:
                if not (r["org"] == org and r["design"] == des and r["tier"] == tier and r["c"] == c and r["t"] == t): continue
                k = (c, t, r["logN"]); d = dpf(k)
                if d is None: continue
                T_ntt, T_net, T_pcg, T_nttp = r[FD.GPU_KEY] - d, r["nic_ms"], r["pcg_ms"], r["ntt_ms"] * r["ntt_slow"]
                Pm, Pp, Pd = p_ntt(base_ntt, r["logN"], c * c), p_ntt(pcg_ntt, r["logN"], c * c), P[f"dpf_c{c}t{t}"]
                E_base = Pd * d + Pm * T_ntt + P_idle * max(0.0, T_net - d - T_ntt)
                T_spup = r["spu_ms"] * r["spu_slow"]                       # SPU lane with contention
                E_pcg = Pp * T_nttp + P_idle * (T_pcg - T_nttp) + P_SPU_RDRU * T_spup + P_BF * T_nttp
                v.append(E_base / E_pcg); pw.append(E_pcg / T_pcg); percell.append((c, t, r["logN"], E_base / E_pcg, E_base, E_pcg))
            if v: per.append(gm(v)); cells += v
        lines.append(f"{clk:16s} {lab:8s}: energy ratio baseline/PCG^2 per (c,t) " + " ".join(f"{x:.2f}" for x in per)
                     + f" | up to {max(per):.2f} geomean {gm(cells):.2f} max cell {max(cells):.2f} | PCG^2 mean system power {sum(pw)/len(pw):.0f} W")
        if tag == "nom": EJ[tier] = per
        for (c, t) in CFG:
            pc = [x for x in percell if x[0] == c and x[1] == t]
            if pc: lines.append(f"    ({c},{t}): " + "  ".join(f"2^{lg} {ratio:.2f} ({eb/1e3:.1f}/{ep/1e3:.2f} J)" for _, _, lg, ratio, eb, ep in pc) + f"  | geomean {gm(x[3] for x in pc):.2f}")
open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
json.dump(EJ, open(os.path.join(os.path.dirname(os.path.abspath(out)), f"energy_{mach.lower()}.json"), "w"), indent=1)
