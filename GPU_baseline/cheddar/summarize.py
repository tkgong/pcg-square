#!/usr/bin/env python3
"""Cheddar vs GPU-NTT merge (HEonGPU's NTT) on one card: forward NTT per size, whole multiply at PCG's (logN, batch), the
four-step NTT with transposes on the SMs from the e2e model's table, and the L40S end-to-end effect of using Cheddar's NTT on
both sides (GPU baseline and PCG^2's SM NTT lane, the latter scaled by the per-(logN, batch) multiply ratio).
Usage (from the repo root): python3 GPU_baseline/cheddar/summarize.py [TAG=l40s]"""
import csv, io, contextlib, json, os, sys
from math import exp, log
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = sys.argv[1] if len(sys.argv) > 1 else "l40s"; RES = os.path.join(ROOT, f"GPU_baseline/results_{TAG}")
gm = lambda v: (lambda l: exp(sum(map(log, l)) / len(l)))(list(v))
rd = lambda p: list(csv.DictReader(l for l in open(p) if not l.startswith("#")))
out = []
S, M = rd(f"{RES}/cheddar_ntt_sweep_{TAG}.csv"), {int(r["logN"]): r for r in rd(f"{RES}/merge_scale_16_24_{TAG}.csv")}
bad = [r for r in S if r["ok"] != "1"]
out.append(f"Forward NTT, 1 GB of 64-bit polynomials, prime 2^62-3*2^25+1; {len(bad)} of {len(S)} Cheddar configs fail the CPU reference check"
           + (" (all with first-kernel shape " + ", ".join(sorted({f"M1 = {r['M1']} at {r['threads1']} threads" for r in bad}))
              + ": these exceed the per-block register file and never launch; the bench now skips such configs; the tables use ok = 1 rows only)" if bad else ""))
out.append("logN | GPU-NTT merge ms (kernels) | Cheddar, its own 2^16 rule (9 row stages) | Cheddar, best two-kernel config (col+row stages, smem KB) | merge/best")
for L in range(16, 25):
    rr = [r for r in S if int(r["logN"]) == L and r["ok"] == "1"]; b = min(rr, key=lambda r: float(r["ms_per_transform"]))
    own = [r for r in rr if (r["P2"], r["M1"], r["B1"]) == ("9", "4", "4")]; m = float(M[L]["merge_ms_per_transform"])
    out.append(f"  {L} | {m:.4f} ({M[L]['merge_kernels']}) | {float(own[0]['ms_per_transform']):.4f}" if own else f"  {L} | {m:.4f} ({M[L]['merge_kernels']}) | infeasible")
    out[-1] += f" | {float(b['ms_per_transform']):.4f} ({b['P1']}+{b['P2']}, {b['smem1_KB']}/{b['smem2_KB']}) | {m / float(b['ms_per_transform']):.2f}x"
P = rd(f"{RES}/cheddar_polymul_{TAG}.csv"); best = {}
for r in P:
    k = (int(r["logN"]), int(r["batch"])); t = float(r["cheddar_ms_per_poly"])
    if k not in best or t < best[k][1]: best[k] = (float(r["merge_ms_per_poly"]), t, f"{r['cheddar_P1']}+{r['P2']}")
out += ["", f"Negacyclic multiply per polynomial (2 forward + pointwise + 1 inverse), ms; {sum(int(r['mismatches']) for r in P)} mismatching elements vs the merge product over {len(P)} runs"]
T = {}
if TAG == "l40s":
    for r in rd(os.path.join(ROOT, "PIM/tools/contention/ntt_l40s_fused4.csv")):
        if r["gpu"] == "gpu": continue
        k = (int(r["logN"]), int(r["batch"])); T[k] = min(T.get(k, 1e9), float(r["f4g_full_ms_mul"]))
out.append("logN batch | merge | Cheddar best (col+row) | merge/Cheddar" + (" | four-step, transposes on SMs (e2e table) | that/Cheddar" if T else ""))
for k in sorted(best):
    m, c, cfg = best[k]
    out.append(f"  {k[0]} {k[1]:3d} | {m:.4f} | {c:.4f} ({cfg}) | {m / c:.2f}x" + (f" | {T[k]:.4f} | {T[k] / c:.2f}x" if T else ""))
out.append(f"  geomean merge/Cheddar over the {len(best)} (logN, batch): {gm(m / c for m, c, _ in best.values()):.2f}x")
if TAG == "l40s":
    sys.path.insert(0, os.path.join(ROOT, "PIM/tools/contention/e2e"))
    with contextlib.redirect_stdout(io.StringIO()):
        from reproduce import L_
    R = json.load(open(os.path.join(ROOT, "PIM/results/contention/final_window/win22/e2e_nom.json"))); LN = L_("L40S")
    out += ["", "L40S end-to-end (final accounting; PCG^2 uses its SM NTT = merge on L40S), with Cheddar's NTT on both sides:"]
    for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
        s0, s1, b0, b1 = [], [], [], []
        for r in R:
            if not (r["org"] == "l40s" and r["design"] == "merge" and r["tier"] == tier): continue
            c, lg = r["c"], r["logN"]; m, ch, _ = best[(lg, c * c)]; f = ch / m; L = LN[(c, r["t"], lg)]
            base0 = r["base_overlap_ms"]; base1 = max(L["gpu_dpf_g"] + (r["gpu_ms"] - L["gpu_dpf_g"]) * f, r["nic_ms"])
            pcg0 = r["pcg_ms"]; pcg1 = max(r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"] * f, r["nic_ms"])
            s0.append(base0 / pcg0); s1.append(base1 / pcg1); b0.append(base0); b1.append(base1)
        out.append(f"  {lab}: GPU baseline {gm(b0):.1f} -> {gm(b1):.1f} ms; PCG^2 speedup geomean {gm(s0):.2f} -> {gm(s1):.2f} (max cell {max(s0):.2f} -> {max(s1):.2f})")

F = f"{RES}/cheddar_f4dru_{TAG}.csv"
if os.path.exists(F):
    Q = [r for r in rd(F) if len(r) > 3 and r["design"] != "OOM"]
    out += ["", f"Four-step NTT on Cheddar's kernels, transposes on the DRU (SM lane = SM kernels + pointwise, DRU fully overlapped), per polynomial multiply, ms; "
            f"{sum(int(r['mismatches']) for r in Q if r['design'] in ('f4c_full', 'cheddar'))} mismatching elements vs the merge product"]
    out.append("logN batch | merge | Cheddar best | four-step SM lane best (col+row) | Cheddar/SM lane | merge/SM lane | four-step with transposes on the SMs | DRAM bound with DRU traffic (216N B at peak) / Cheddar")
    PEAK = {"l40s": 864e9, "b200": 8.0e12}[TAG]; rc, rm = [], []
    for L in range(20, 25):
        for b in (4, 16, 64):
            rr = [r for r in Q if int(r["logN"]) == L and int(r["batch"]) == b]
            if not rr: continue
            g = lambda d: min(rr, key=lambda r: float(r["ms_per_poly"]) if r["design"] == d else 1e9)
            m = float(g("merge")["ms_per_poly"]); c = float(g("cheddar")["ms_per_poly"]); s_ = g("f4c_sm"); sv = float(s_["ms_per_poly"]); fv = float(g("f4c_full")["ms_per_poly"])
            rc.append(c / sv); rm.append(m / sv); lb = 216 * (1 << L) / PEAK * 1e3
            out.append(f"  {L} {b:3d} | {m:.4f} | {c:.4f} | {sv:.4f} ({s_['P1']}+{s_['P2']}) | {c / sv:.2f}x | {m / sv:.2f}x | {fv:.4f} | {lb / c:.2f}")
    out.append(f"  geomean Cheddar/SM lane {gm(rc):.2f}x (range {min(rc):.2f}-{max(rc):.2f}); merge/SM lane {gm(rm):.2f}x")
s = "\n".join(out); print(s); open(f"{RES}/cheddar_vs_heongpu_{TAG}.txt", "w").write(s + "\n")
