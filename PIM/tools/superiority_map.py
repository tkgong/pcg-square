#!/usr/bin/env python3
"""
PIM superiority map v1 — full (N, t, PU) plane, Expand phase (deltashift
excluded; PCG Gen/Expand split), design vs the strongest L40S all-GPU baseline.

Lanes and provenance:
  baseline (L40S all-GPU, kernel-only = strongest):
      T = 2Nt x DPF_ns_leaf_L40S  +  2 x polymul_L40S(N)      [per pair, serial]
  design (two-tier PIM + SM NTT, PIM||SM overlapped):
      T = max( 2Nt x 0.401ns / (PU/64),  NTT_term(N) )        [per pair]
      with optional MAU stage-1 offload shaving the NTT term.
  contention = 0 under the BANK-PARTITION PLACEMENT PREMISE (PIM regions are a
  physical-address carve-out; NTT working set lives in PIM-free banks; the only
  shared traffic is the g handoff, N x 8B per pair ~ negligible). Without the
  premise, apply the measured [3%,10%] bounds (tree_sweeps_smstream.txt).

L40S numbers (v1.1): polymul is MEASURED silicon at every anchor 2^16..2^24
(nsys kernel-only, GPU3 2026-07-08, pim/results/l40s_largeN.txt; odd exponents
log-log interpolated between silicon anchors). The L2(96MB)->DRAM regime break
is visible at 2^23 (slope 2.2x -> 3.2x per octave). Note: the old Ada merge
table's large-N entries include copy/staging and overestimate kernel-only NTT
~6x at 2^20 -- superseded for all model uses. DPF_ns_leaf_L40S = 0.744
(kernel-only, best shape, measured).

Capacity wall: cells where the double-buffered leaf region exceeds one bank
are flagged W: they require the block-streaming schedule, which step-4
MEASURED at 0.3963 ns/leaf -- faster than the flat 0.401 used here, so W-cell
speedups are conservative (tree_sweeps_blocked.txt).
"""
import math

# RETIRED (2026-07-11): DPF_NS_LEAF_L40S = 0.744 embedded the fake single-bin
# leaf_convert (caliber finding D1) and overstated the real all-GPU DPF+lc by
# 2.5-6.8x. Replaced by the measured three-regime lookup below (baseline_v2
# PHASE 1 battery, production kernels, kernel-only; total leaves = 2Nt picks
# the regime; gpu-ntt-pivot:pim/results/l40s_realkernel_dpf.txt).
def dpf_ns_leaf_l40s(total_leaves):
    """Measured L40S DPF+lc regime curve (kernel-only ns/leaf), log-log
    interpolated between battery anchors; flat 0.195 plateau >= 134M leaves."""
    import math as _m
    pts = [(0.52e6, 0.188), (1.05e6, 0.134), (2.1e6, 0.108), (4.2e6, 0.118),
           (8.4e6, 0.163), (16.8e6, 0.174), (33.6e6, 0.186), (67.1e6, 0.189),
           (134.2e6, 0.197), (1.074e9, 0.195)]
    if total_leaves <= pts[0][0]:
        return pts[0][1]
    if total_leaves >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= total_leaves <= x1:
            f = (_m.log(total_leaves) - _m.log(x0)) / (_m.log(x1) - _m.log(x0))
            return y0 * (y1 / y0) ** f
    return pts[-1][1]

PIM_NS_LEAF = 0.401               # measured, two-tier MAU, 64 PU (HBM3 org)
POLYMUL_L40S_US_MEASURED = {   # nsys kernel-only silicon, GPU3 2026-07-08
    65536: 20.2, 262144: 42.3, 1048576: 126.0, 2097152: 246.3,
    4194304: 551.2, 8388608: 1791.0, 16777216: 4513.1}
ADA_MERGE_MS = {65536: 0.1486, 131072: 0.3052, 262144: 0.7108,
                524288: 2.098, 1048576: 5.456}
BANK_BYTES = 128 << 20            # 2^17 rows x 1 KiB (modeled org)

def polymul_l40s_us(N):
    """Silicon at 2^16; elsewhere Ada shape x device factor anchored at 2^16.
    Above the Ada table, scale ~N (DRAM-bound regime tail)."""
    import math as _m
    if N in POLYMUL_L40S_US_MEASURED:
        return POLYMUL_L40S_US_MEASURED[N], ""
    keys = sorted(POLYMUL_L40S_US_MEASURED)
    if keys[0] < N < keys[-1]:               # log-log interpolate silicon
        lo = max(k for k in keys if k <= N); hi = min(k for k in keys if k >= N)
        f = (_m.log(N)-_m.log(lo))/(_m.log(hi)-_m.log(lo))
        v = POLYMUL_L40S_US_MEASURED[lo] * (POLYMUL_L40S_US_MEASURED[hi]/POLYMUL_L40S_US_MEASURED[lo])**f
        return v, ""
    dev = POLYMUL_L40S_US_MEASURED[65536] / (ADA_MERGE_MS[65536] * 1e3)
    keys = sorted(ADA_MERGE_MS)
    if N <= keys[-1]:
        lo = max(k for k in keys if k <= N); hi = min(k for k in keys if k >= N)
        if lo == hi:
            ada = ADA_MERGE_MS[lo]
        else:
            f = (math.log(N) - math.log(lo)) / (math.log(hi) - math.log(lo))
            ada = ADA_MERGE_MS[lo] * (ADA_MERGE_MS[hi] / ADA_MERGE_MS[lo]) ** f
    else:
        ada = ADA_MERGE_MS[keys[-1]] * (N / keys[-1])
    return ada * 1e3 * dev, "p"

def ntt_term_us(N, PU, t_pim_us):
    """Per-pair NTT time on the design side: 2 full poly-muls, minus optimal
    MAU stage-1 offload (only helps when NTT is the pole). MAU stage-1 capacity:
    64 ch x 0.5 bf/CK (2-bank CCDL)."""
    full, flag = polymul_l40s_us(N)
    t_ntt = 2 * full
    n = int(math.log2(N))
    N1 = min(1 << 14, N >> 6)
    f1 = math.log2(N1) / n if N1 > 1 else 0
    bf_total = 2 * 3 * (N / 2) * n            # 2 muls x 3 transforms
    w1_mau = bf_total * f1 / (64 * 0.5) * 0.625e-3   # us
    w1_gpu, w2_gpu = t_ntt * f1, t_ntt * (1 - f1)
    best = max(t_pim_us, t_ntt)
    for i in range(101):
        x = i / 100
        T = max(t_pim_us + x * w1_mau, (1 - x) * w1_gpu + w2_gpu)
        best = min(best, T)
    return t_ntt, best, flag

def main():
    print(__doc__.split("\n\n")[0])
    print(f"# {'N':>8} {'t':>3} | {'base us':>9} | "
          f"{'PU=64':>7} {'x':>6} {'PU=256':>7} {'x':>6} {'PU=1024':>8} {'x':>6} | flags")
    for lgN in (16, 18, 20, 22, 24):
        N = 1 << lgN
        for t in (4, 8, 16, 32):
            leaves = 2 * N * t                       # per pair
            base_dpf = leaves * dpf_ns_leaf_l40s(leaves) / 1e3   # us
            full, flag = polymul_l40s_us(N)
            base = base_dpf + 2 * full
            # leaf region is double-buffered (round parity) -> 2x one instance
            wall = "W" if 2 * (2 * N // t) * 16 >= BANK_BYTES else ""
            cells = []
            for PU in (64, 256, 1024):
                t_pim = leaves * PIM_NS_LEAF / (PU / 64) / 1e3
                _, T, _ = ntt_term_us(N, PU, t_pim)
                cells += [T, base / T]
            print(f"  {'2^%d' % lgN:>8} {t:>3} | {base:9.0f} | "
                  f"{cells[0]:7.0f} {cells[1]:5.1f}x {cells[2]:7.0f} {cells[3]:5.1f}x "
                  f"{cells[4]:8.0f} {cells[5]:5.1f}x | {flag}{wall}")
        print()
    print("# flags: p = NTT point pending L40S silicon (Ada-shape x device factor,")
    print("#            watcher endtoend/run_largeN_bench.sh will land the real number)")
    print("#        W = capacity wall: double-buffered leaf region 2 x (2N/t x 16B)")
    print("#            >= one bank (128MB) -> REQUIRES the block-streaming schedule,")
    print("#            which is MEASURED FASTER than flat (0.3963 vs 0.4143 ns/leaf,")
    print("#            tree_sweeps_blocked.txt) -> W-cell speedups are conservative")

if __name__ == "__main__":
    main()
