#!/usr/bin/env python3
"""Reviewer A, Q1: the SPU's assumed frequency and throughput, the aggregate capability of the SPU array against the
GPU's SMs (raw 32-bit integer op rate, and the DPF leaf throughput each side actually reaches), and the sensitivity of
the headline speedup to that ratio: the SPU lane is slowed by k (= the SPU/GPU capability ratio divided by k), the
NTT lane, the network and the co-simulated slowdowns are kept.    Usage: spu_capability_sweep.py WIN22_DIR OUT_TXT"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
W, OUT = sys.argv[1], sys.argv[2]
# SPU datapath as simulated: 8 lanes x (VADD + VXORL) = 16 32-bit ops per cycle at the DRAM clock; ChaCha8 block per
# lane = 155 cycles (expand), 465 cycles for the fused last level (expand + H' + mod-p + partial sum), 4 cycles per
# 8-leaf group for the fold (pass 2); 8 SPUs per channel.
SPU = {"L40S": dict(org="l40s", f=2.25e9, n=192, des="merge"), "B200": dict(org="b200", f=2.0e9, n=2048, des="f4dru")}
GPU = {"L40S": dict(sms=142, lanes=64, f=2.52e9), "B200": dict(sms=148, lanes=128, f=1.965e9)}   # INT32 rate as in the paper's roofline: L40S 64 INT32 lanes/SM = 22.9 Tops, B200 128/SM/clk = 37.2 Tops
OPS_PER_CYCLE = 16
cyc_per_leaf = (155 + 465) / 16 + 4 / 8                   # tree levels + fused last level + fold, per leaf
L = []
for mach, S in SPU.items():
    G = GPU[mach]; LN = L_(mach)
    R = [r for r in json.load(open(os.path.join(W, "e2e_nom.json"))) if r["org"] == S["org"] and r["design"] == S["des"]]
    spu_ops = S["n"] * OPS_PER_CYCLE * S["f"]; gpu_ops = G["sms"] * G["lanes"] * G["f"]
    spu_leaf_peak = S["n"] * S["f"] / cyc_per_leaf
    cells = {(r["c"], r["t"], r["logN"]): r for r in R if r["tier"] == "fast"}
    leaves = {k: k[0] ** 2 * 2 * (1 << k[2]) * k[1] for k in cells}
    spu_tp = gm(leaves[k] / (cells[k]["spu_ms"] / 1e3) for k in cells); gpu_tp = gm(leaves[k] / (LN[k]["gpu_dpf_g"] / 1e3) for k in cells)
    L.append(f"{mach}: SPU = {S['n']} units x 8 lanes x 2 ALUs at the DRAM clock {S['f']/1e9:.2f} GHz = {spu_ops/1e12:.1f} T int32 ops/s; "
             f"GPU SMs = {G['sms']} x {G['lanes']} lanes x {G['f']/1e9:.2f} GHz = {gpu_ops/1e12:.1f} T ops/s -> raw ratio SPU/GPU {spu_ops/gpu_ops:.2f}")
    L.append(f"      per SPU: ChaCha8 block per lane in 155 cycles, fused last level 465, fold 4 per 8 leaves -> {cyc_per_leaf:.2f} cycles/leaf = {S['f']/cyc_per_leaf/1e6:.0f} M leaves/s per SPU, "
             f"{spu_leaf_peak/1e9:.0f} G leaves/s for the array (simulated lane reaches {spu_tp/1e9:.0f} G leaves/s, geomean over cells)")
    L.append(f"      GPU DPF kernel reaches {gpu_tp/1e9:.1f} G leaves/s (measured, two-pass H') -> achieved DPF throughput ratio SPU/GPU {spu_tp/gpu_tp:.2f}; "
             f"GPU ALU utilisation in the DPF ~ {gpu_tp * 3 * 256 / gpu_ops:.0%} (3 ChaCha8 blocks/leaf), SPU ~ {spu_tp * 2 * 256 / spu_ops:.0%} (2 blocks/leaf)")
    L.append(f"      {'k':>4s} {'raw ratio':>9s} {'DPF ratio':>9s} | 40 Gbps up to / geomean | 400 Mbps up to / geomean | SPU-bound cells 40 Gbps")
    for k in ((1, 1.25, 1.5, 1.76, 2, 3, 4, 6, 8, 12, 16) if mach == "B200" else (1, 1.25, 1.5, 2, 2.5, 3, 4, 6, 8, 12, 16)):
        row = []
        for tier in ("fast", "slow"):
            rr = [r for r in R if r["tier"] == tier]
            sp = {}
            for r in rr:
                base = max(r["gpu_ms"], r["nic_ms"]); p = max(k * r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"], r["nic_ms"])
                sp.setdefault((r["c"], r["t"]), []).append(base / p)
            per = {ct: gm(v) for ct, v in sp.items()}; allv = [x for v in sp.values() for x in v]
            row.append((max(per.values()), gm(allv)))
            if tier == "fast": nb = sum(1 for r in rr if k * r["spu_ms"] * r["spu_slow"] >= max(r["ntt_ms"] * r["ntt_slow"], r["nic_ms"]))
        L.append(f"      {k:4.2f} {spu_ops/gpu_ops/k:9.2f} {spu_tp/gpu_tp/k:9.2f} | {row[0][0]:7.2f} / {row[0][1]:7.2f}      | {row[1][0]:7.2f} / {row[1][1]:7.2f}       | {nb}/{len(cells)}")
    # k at which the geomean 40 Gbps speedup reaches 1
    lo, hi = 1.0, 200.0
    for _ in range(60):
        k = (lo + hi) / 2
        g = gm(max(r["gpu_ms"], r["nic_ms"]) / max(k * r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"], r["nic_ms"]) for r in R if r["tier"] == "fast")
        lo, hi = (k, hi) if g > 1 else (lo, k)
    L.append(f"      geomean speedup reaches 1.0 at k = {lo:.1f} (raw ratio {spu_ops/gpu_ops/lo:.3f}, DPF ratio {spu_tp/gpu_tp/lo:.2f}); the existing re-simulated points: 1.75/1.5/1.0 GHz = k 1.14/1.33/2.0, single ALU = k 1.8")
    L.append("")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
