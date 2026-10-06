#!/usr/bin/env python3
"""Reviewer A, Q1: sensitivity of the headline to the PIM/GPU compute ratio, memory system fixed. Only the SPU's
functional-unit cycles (ChaCha8 expansion, fused conversion, mod-mul, accumulation) are scaled: capability x0.5 / x0.75 /
x1 / x1.25 = arithmetic latency x2 / x1.33 / x1 / x0.8. DRAM timing, ACT16/ABRD/ABWR, the GPU request streams and the
network are unchanged; contention is re-co-simulated for every point (e2e_window.py --alu-mult).
Usage: compute_ratio_sweep.py OUT_TXT label=dir [label=dir ...]   (each dir holds e2e.json; several dirs per label may be
given as label=dir1+dir2, e.g. the L40S and B200 halves of one run)"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
OUT = sys.argv[1]; RUNS = [a.split("=") for a in sys.argv[2:]]
DES = {"L40S": ("l40s", "merge"), "B200": ("b200", "f4dru")}
LN = {m: L_(m) for m in DES}
L = ["Headline vs the PIM/GPU compute ratio (memory system, GPU kernels and network fixed; SPU FPU cycles scaled; contention re-co-simulated)", ""]
for mach, (org, des) in DES.items():
    L.append(f"{mach}: {'capability':>10s} {'SPU lat':>8s} | {'SPU lane ms':>11s} {'SPU slow':>8s} {'NTT slow':>8s} {'DPF ratio':>9s} | {'40 Gbps up to / geomean':>24s} | {'400 Mbps up to / geomean':>24s} | SPU-bound (40G)")
    for lab, dirs in RUNS:
        R = []
        for d in dirs.split("+"):
            if os.path.exists(os.path.join(d, "e2e.json")): R += json.load(open(os.path.join(d, "e2e.json")))
        rr = [r for r in R if r["org"] == org and r["design"] == des]
        if not rr: L.append(f"      {lab:>10s}  (no rows)"); continue
        cap = float(lab.lstrip("x")); fast = [r for r in rr if r["tier"] == "fast"]
        spu = gm(r["spu_ms"] for r in fast); ss = gm(r["spu_slow"] for r in fast); ns = gm(r["ntt_slow"] for r in fast)
        dpf = gm(LN[mach][(r["c"], r["t"], r["logN"])]["gpu_dpf_g"] / r["spu_ms"] for r in fast)
        nb = sum(1 for r in fast if r["spu_ms"] * r["spu_slow"] >= max(r["ntt_ms"] * r["ntt_slow"], r["nic_ms"]))
        cols = []
        for tier in ("fast", "slow"):
            per = {}
            for r in rr:
                if r["tier"] != tier: continue
                per.setdefault((r["c"], r["t"]), []).append(max(r["gpu_ms"], r["nic_ms"]) / r["pcg_ms"])
            cols.append((max(gm(v) for v in per.values()), gm(x for v in per.values() for x in v)))
        L.append(f"      {lab:>10s} {1/cap:7.2f}x | {spu:11.1f} {ss:8.3f} {ns:8.3f} {dpf:9.2f} | {cols[0][0]:10.2f} / {cols[0][1]:<10.2f} | {cols[1][0]:10.2f} / {cols[1][1]:<10.2f} | {nb}/{len(fast)}")
    L.append("")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
