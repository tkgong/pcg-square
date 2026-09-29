#!/usr/bin/env python3
"""PCG^2 with the DRU (fused four-step on GPU-NTT kernels, transposes on the
DRU) vs PCG^2 without it (GPU-NTT merge on the SMs), both co-running with the
SPU, over the SOTA all-GPU baseline (GPU DPF + merge NTT).
Per-mul NTT times: measured (L40S: fused4 CSV; B200: AICR job, fused4-dru a392780).
Contention: f(r) = both-done / max(alone) from the dm_* co-simulations
(config c for the DRU design, config m for merge)."""
import io, contextlib, csv, json, os, re, sys
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, LOGN, alpha_bw, gm
    from fig8 import BETA
    from sim_e2e import interp
C = sys.argv[1] if len(sys.argv) > 1 else "../cont"
B200 = """20 4 .1291 .1251|20 16 .1251 .1213|20 64 .1234 .1203|21 4 .2650 .2625|21 16 .2594 .2567|21 64 .2578 .2556|
22 4 .5463 .5369|22 16 .5390 .5311|22 64 .5368 .5291|23 4 1.1297 1.1074|23 16 1.1232 1.1023|23 64 1.1210 1.1002|
24 4 2.4100 2.2850|24 16 2.4015 2.2766|24 64 2.4008 2.2769"""
NTT = {"B200": {}, "L40S": {}}
for e in B200.replace("\n", "").split("|"):
    lg, b, m, s = e.split(); NTT["B200"][(int(lg), int(b))] = (float(m), float(s))
_lines = open("l40s_f4g.csv").read().splitlines()
for r in csv.DictReader([_lines[0]] + [l for l in _lines if not l.startswith("gpu,") and not l.startswith("#")]):
    k = (int(r["logN"]), int(r["batch"])); m, s = float(r["merge_ms_mul"]), float(r["f4g_sm_ms_mul"])
    if s <= 0: continue
    o = NTT["L40S"].get(k); NTT["L40S"][k] = (min(m, o[0]), min(s, o[1])) if o else (m, s)
def fcurves(d):
    R = json.load(open(os.path.join(C, d, "results.json"))); T = R["T_spu"]; R = R["results"]
    fin = lambda x: max(v for k, v in x.items() if re.fullmatch(r"stream\d+_class\d_finish_max", k))
    def busy(tag):   # host lane alone: sum over chunks of (finish - start), SM stream only (class 1)
        st = [(int(l.split()[0]), int(l.split()[7])) for l in open(os.path.join(C, d, tag + ".streams")) if not l.startswith("#")]
        return sum(R[tag][f"stream{k}_class{c}_finish_mean"] - s0 for k, (c, s0) in enumerate(st) if c == 1)
    fc, fm = [], []
    for k in R:
        m = re.match(r"r([0-9.]+)_c_fair$", k)
        if m:
            rv = m.group(1); x = R[k]; a = fin(R[f"r{rv}_ref_smdru"]); b = busy(f"r{rv}_ref_gpusm")
            fc.append([min(b, T) / max(b, T), max(x["pim_done_cycles"], fin(x)) / max(a, T)])
            x = R[f"r{rv}_m_fair"]; a = fin(R[f"r{rv}_ref_merge"]); b = busy(f"r{rv}_ref_merge")
            fm.append([min(b, T) / max(b, T), max(x["pim_done_cycles"], fin(x)) / max(a, T)])
    return T, sorted(fc), sorted(fm)
def run(mach, tier, sc, fc, fm):
    LN = L_(mach); out = {"dru": [], "merge": []}; per = {"dru": [], "merge": []}
    for (c, t) in CFG:
        a = alpha_bw(c, t, BETA[tier]); S = {"dru": [], "merge": []}
        for lg in LOGN:
            L = LN.get((c, t, lg))
            if not L: continue
            nic = (L["n"] + 2) * a; mm, ss = NTT[mach][(lg, c * c)]
            g = L["gpu_dpf_g"] + mm * 2 * c * c + nic            # SOTA baseline: merge NTT
            spu = L["spu"] * sc
            for name, sm, f in (("dru", ss * 2 * c * c, fc), ("merge", mm * 2 * c * c, fm)):
                hi, lo = max(spu, sm), min(spu, sm)
                S[name].append(g / max(hi * (interp(f, lo / hi) if f else 1.0), nic))
        for k in S: per[k].append(gm(S[k])); out[k] += S[k]
    return {k: (max(per[k]), gm(out[k])) for k in per}
if __name__ == "__main__":
    for mach, org in (("L40S", "l40s"), ("B200", "b200")):
        Tn = fcurves(f"dm_{org}_nom")[0]
        for clk in ("nom", "1.0"):
            T, fc, fm = fcurves(f"dm_{org}_{clk}")
            print(f"{mach} clk={clk}: f_DRU(r)={[[round(x,2) for x in p] for p in fc]}  f_merge(r)={[[round(x,2) for x in p] for p in fm]}")
            for tier in ("fast", "slow"):
                i = run(mach, tier, T / Tn, None, None); c = run(mach, tier, T / Tn, fc, fm)
                print(f"   {tier}: no contention  DRU {i['dru'][0]:.2f}/{i['dru'][1]:.2f}  merge {i['merge'][0]:.2f}/{i['merge'][1]:.2f}"
                      f"  | contention  DRU {c['dru'][0]:.2f}/{c['dru'][1]:.2f}  merge {c['merge'][0]:.2f}/{c['merge'][1]:.2f}")
