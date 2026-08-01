#!/usr/bin/env python3
"""
Phase-stitched end-to-end prediction, v1 ("shortest path").

Composes three measurement layers into an e2e time per (N, t) cell:
  host residual   : measured on the real 2-PC run (docs/gpu_end_to_end.md,
                    RTX 5000 Ada GPU path, Xeon w5-3435X host) =
                    total x (1 - named_local%) = F_DeltaShiftShare +
                    unattributed sync (the doc's timers do not close to 100%).
  GPU compute     : Accel-Sim SM80_A100 replay constants (endtoend/results.md):
                    DPF+leaf_convert 0.912 ns/leaf (kernel-only, device-resident
                    equivalent), poly-mul(65536) = 56.1 us + merge-table shape.
  PIM             : Ramulator2-calibrated 0.45 ns/leaf / (PU/64).

Variants per cell:
  (a) real Ada e2e            (doc reference)
  (b) stitched all-GPU        residual + simDPF+lc + simNTT   [device-resident]
  (c) PIM design, serial      residual + T_pim + simNTT
  (d1) PIM design, pipelined  max(residual, T_pim + simNTT)   [deltashift
       overlapped across independent PCG batches]
  (d2) PIM design, full охerlap max(residual, T_pim, simNTT)  [also PIM || SM]

NOTE (b) is not a reproduction of (a): (a)'s named phases include per-level
D2H/H2D and host packing; (b) uses kernel-only device-resident numbers -- it is
the stitched *best all-GPU implementation* on the simulated machine.
Composition validity anchor: the doc's e2e-derived DPF rate (2.4 ns/leaf, Ada)
matches our standalone asymptote (repro_e2e_gpu_baseline.md).
"""

# ---- host tables (docs/gpu_end_to_end.md, measured 2026-06-26) -------------
# t=16 sweep with per-phase splits: N_log2 -> (total_ms, deltashift%, dpf%,
# lc%, step4%, named_local_total%)
T16_SWEEP = {
    10: (163,  22.2, 0.5, 0.2, 0.2, 0.8),
    12: (216,  21.1, 1.2, 0.4, 0.3, 1.8),
    14: (268,  19.3, 2.7, 1.0, 0.5, 4.3),
    16: (628,   9.6, 3.4, 2.7, 1.1, 7.1),
    18: (1117,  4.9, 7.2, 6.7, 2.7, 16.6),
}
# unified sweep representative rows: (t, N_log2) -> (gpu_total_ms, named_local%)
UNIFIED = {
    (4, 10): (11.7, 6.6),  (4, 14): (26.1, 16.5), (4, 16): (48.8, 27.6),  (4, 20): (985, 41.5),
    (8, 10): (43.9, 2.4),  (8, 16): (110, 16.6),  (8, 20): (1648, 30.6),
    (16, 10): (165, 0.9),  (16, 16): (614, 7.1),  (16, 20): (3129, 24.0),
    (32, 10): (958, 0.3),  (32, 16): (1932, 4.0), (32, 18): (2861, 11.4),
}

# ---- simulated-machine constants (endtoend/results.md) ----------------------
A100SIM_DPF_NS_LEAF = 0.912         # DPF+leaf_convert, kernel-only, fastest shape
A100SIM_POLYMUL_65536_MS = 0.0561   # merge full poly-mul, replayed
PIM_NS_LEAF_64PU = 0.45             # Ramulator2, 64 PU
GPU_MERGE_POLYMUL_MS = {            # RTX5000Ada merge table, SHAPE only
    4096: 0.0255, 8192: 0.0323, 16384: 0.0504, 32768: 0.0816,
    65536: 0.1486, 131072: 0.3052, 262144: 0.7108, 524288: 2.098,
    1048576: 5.456,
}

def a100sim_polymul_ms(N):
    """Anchor at the replayed N=65536 point; other N by merge-table shape.
    Below the table (N<4096) small NTTs are launch-bound: clamp to the 4096
    entry (a floor, favorable to the GPU baseline)."""
    import math
    ref = GPU_MERGE_POLYMUL_MS[65536]
    keys = sorted(GPU_MERGE_POLYMUL_MS)
    if N <= keys[0]:
        shape = GPU_MERGE_POLYMUL_MS[keys[0]] / ref
    elif N in GPU_MERGE_POLYMUL_MS:
        shape = GPU_MERGE_POLYMUL_MS[N] / ref
    elif N > keys[-1]:
        shape = GPU_MERGE_POLYMUL_MS[keys[-1]] / ref * (N / keys[-1])
    else:
        lo = max(k for k in keys if k <= N); hi = min(k for k in keys if k >= N)
        f = (math.log(N) - math.log(lo)) / (math.log(hi) - math.log(lo))
        shape = (GPU_MERGE_POLYMUL_MS[lo] *
                 (GPU_MERGE_POLYMUL_MS[hi] / GPU_MERGE_POLYMUL_MS[lo]) ** f) / ref
    return A100SIM_POLYMUL_65536_MS * shape

def cell(t, nlog, total_ms, named_pct, PUs=(64, 256)):
    N = 1 << nlog
    leaves = 8 * N * t                        # c^2 * t^2 * (2N/t), c=2
    residual = total_ms * (1 - named_pct / 100.0)
    dpf_lc = leaves * A100SIM_DPF_NS_LEAF / 1e6
    ntt = 8 * a100sim_polymul_ms(N)           # 8 full poly-muls at degree N
    b = residual + dpf_lc + ntt
    out = dict(N=N, t=t, residual=residual, dpf_lc=dpf_lc, ntt=ntt, a=total_ms, b=b)
    for PU in PUs:
        pim = leaves * PIM_NS_LEAF_64PU / (PU / 64) / 1e6
        out[f"c{PU}"] = residual + pim + ntt
        out[f"d1_{PU}"] = max(residual, pim + ntt)
        out[f"d2_{PU}"] = max(residual, pim, ntt)
        out[f"pim{PU}"] = pim
    return out

def main():
    print("# Phase-stitched e2e v1. host=Xeon/Ada measured (docs/gpu_end_to_end.md);")
    print("# GPU=Accel-Sim SM80_A100 replay constants; PIM=Ramulator2 0.45ns/leaf.")
    print("# residual = total x (1-named%) = deltashift + unattributed sync.")
    print("# (a) real-Ada e2e  (b) stitched all-GPU  (c) +PIM serial")
    print("# (d1) deltashift overlapped  (d2) + PIM||SM overlap.  PU in {64,256}.")
    hdr = (f"# {'t':>3} {'N':>8} | {'resid':>7} {'dpf+lc':>7} {'ntt':>6} | "
           f"{'a:Ada':>7} {'b:allG':>7} {'c:64':>7} {'c:256':>7} "
           f"{'d1:256':>7} {'d2:256':>7} | {'b/d1':>5} {'a/d1':>5}")
    print("\n== unified sweep cells (t x N grid, named-local% only) ==")
    print(hdr)
    for (t, nlog), (tot, named) in sorted(UNIFIED.items()):
        r = cell(t, nlog, tot, named)
        print(f"  {t:>3} 2^{nlog:<6} | {r['residual']:7.1f} {r['dpf_lc']:7.2f} "
              f"{r['ntt']:6.2f} | {r['a']:7.1f} {r['b']:7.1f} {r['c64']:7.1f} "
              f"{r['c256']:7.1f} {r['d1_256']:7.1f} {r['d2_256']:7.1f} | "
              f"{r['b']/r['d1_256']:5.2f} {r['a']/r['d1_256']:5.2f}")
    print("\n== t=16 sweep (per-phase splits available) + cross-check ==")
    print(hdr)
    for nlog, (tot, ds, dpf, lc, s4, named) in sorted(T16_SWEEP.items()):
        r = cell(16, nlog, tot, named)
        print(f"  {16:>3} 2^{nlog:<6} | {r['residual']:7.1f} {r['dpf_lc']:7.2f} "
              f"{r['ntt']:6.2f} | {r['a']:7.1f} {r['b']:7.1f} {r['c64']:7.1f} "
              f"{r['c256']:7.1f} {r['d1_256']:7.1f} {r['d2_256']:7.1f} | "
              f"{r['b']/r['d1_256']:5.2f} {r['a']/r['d1_256']:5.2f}")
    print("\n== EXPAND-ONLY view (deltashift & residual EXCLUDED; PCG Gen/Expand "
          "split -- the silent local phase) ==")
    print(f"# {'t':>3} {'N':>8} | {'a\\':>8} {'b:allG':>7} {'c:64':>7} {'c:256':>7} "
          f"{'d2:256':>7} | {'a\\/c256':>7} {'b/c256':>6} {'b/d2':>5}")
    print("#  a\\ = Ada measured named-local (dpf_eval+leaf_conv+step4, incl. "
          "copies/packing); b,c,d2 = kernel-only sim")
    for (t, nlog), (tot, named) in sorted(UNIFIED.items()):
        r = cell(t, nlog, tot, named)
        a_named = tot * named / 100.0
        b_x = r['dpf_lc'] + r['ntt']
        c64 = r['pim64'] + r['ntt']
        c256 = r['pim256'] + r['ntt']
        d2 = max(r['pim256'], r['ntt'])
        print(f"  {t:>3} 2^{nlog:<6} | {a_named:8.2f} {b_x:7.2f} {c64:7.2f} "
              f"{c256:7.2f} {d2:7.2f} | {a_named/c256:7.1f} {b_x/c256:6.2f} "
              f"{b_x/d2:5.2f}")

    print("\n# cross-check (t=16): doc measured (dpf%+lc%)*total vs sim dpf_lc "
          "(kernel-only; ratio = copies/sync overhead x device):")
    for nlog, (tot, ds, dpf, lc, s4, named) in sorted(T16_SWEEP.items()):
        doc_ms = (dpf + lc) / 100 * tot
        r = cell(16, nlog, tot, named)
        print(f"#   N=2^{nlog}: doc {doc_ms:7.2f} ms vs sim {r['dpf_lc']:7.2f} ms "
              f"-> {doc_ms/max(r['dpf_lc'],1e-9):5.1f}x")

if __name__ == "__main__":
    main()
