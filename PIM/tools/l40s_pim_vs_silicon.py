#!/usr/bin/env python3
"""
L40S GDDR6 full-card PIM (192 PU) vs L40S silicon, SAME (N,t) protocol grid.

GPU lane  : MEASURED silicon, kernel-only DPF+lc ns/leaf per protocol cell
            (baseline_v2 PHASE 1 battery, endtoend/dpf_real_bench via run.sh,
            GPU3 exclusive 2026-07-11, gpu-ntt-pivot:pim/results/
            l40s_realkernel_dpf.txt). Highest-evidence baseline: no lookup, no
            interpolation -- the exact cell that the PIM design competes with.
PIM lane  : Ramulator2 GDDR6_L40S org (24ch x 16bank, 2 banks/PU = 192 PU,
            GDDR6_18Gbps timing, gated), full-load instances traces
            (pim/tools/gen_dpf_tree_trace.py -C 24 -P 8), two-tier MAU
            (die-level on GDDR: 1 MAU / 2 channels, cl=0 column streams).
            ns/leaf constants below are MEASURED sweep outputs
            (pim/results/tree_sweeps_gddr6.txt):
              single-ALU PU (cl=272, seed-parallel ChaCha8 8r8w)
              dual-ALU PU   (cl=155, XADD/VXORL anti-phase QR pairing;
                             requires in-row packing, see sweep notes)
Comparison: DPF+lc phase only (the PIM offload target). NTT is common to both
            sides and excluded here; co-execution effects live in
            coexec_model.py --gddr.
"""
import math

# ---- GPU lane: measured silicon grid, kernel-only TOT ns/leaf --------------
# (lgN, t) -> ns/leaf. Each bench cell ran B=t^2 instances of 2^n=2N/t leaves
# = 2Nt total leaves = exactly one pair's expand volume, so the per-cell rate
# is the per-pair rate.
GPU_NS_LEAF = {  # baseline_v2 PHASE 1, tier=kernel-only
    (16, 4): 0.188, (16, 8): 0.134, (16, 16): 0.110, (16, 32): 0.127,
    (18, 4): 0.107, (18, 8): 0.110, (18, 16): 0.161, (18, 32): 0.174,
    (20, 4): 0.165, (20, 8): 0.175, (20, 16): 0.180, (20, 32): 0.182,
    (22, 4): 0.192, (22, 8): 0.193, (22, 16): 0.195, (22, 32): 0.195,
    (24, 4): 0.203, (24, 8): 0.195, (24, 16): 0.195, (24, 32): 0.195,
}

# ---- PIM lane: measured GDDR6 full-load sweep constants (ns/leaf) ----------
# Filled from pim/results/tree_sweeps_gddr6.txt (n=12, I=768, 24ch x 8PU,
# gated, reduce=mau, prepend+round-robin discipline).
PIM_NS_LEAF_GDDR6 = {
    "single": None,   # cl=272 measured
    "dual":   None,   # cl=155 measured (dual-ALU PU)
}
PIM_FLOOR_NS_LEAF = None      # cl=0 memory floor (channel-bus bound)
SM_CONTENTION = {"single": None, "dual": None}   # (mau+sm - mau)/mau upper bound

BANK_BYTES = 128 << 20        # 2^17 rows x 1 KiB (GDDR6_L40S org, same as HBM cfg)


def bottleneck(alu):
    """Attribution from the sweep decomposition: compute-bound if the measured
    point sits on the cl-scaled compute line, bus-bound if it sits on the
    cl=0 floor."""
    ns = PIM_NS_LEAF_GDDR6[alu]
    if ns is None:
        return "?"
    if PIM_FLOOR_NS_LEAF and ns <= PIM_FLOOR_NS_LEAF * 1.10:
        return "bus"
    return "compute"


def main():
    print(__doc__.split("\n\n")[0])
    s, d = PIM_NS_LEAF_GDDR6["single"], PIM_NS_LEAF_GDDR6["dual"]
    if s is None or d is None:
        print("!! PIM sweep constants not yet filled (tree_sweeps_gddr6.txt pending)")
        return
    print(f"# PIM 192PU measured: single-ALU {s:.4f} ns/leaf ({bottleneck('single')}-bound), "
          f"dual-ALU {d:.4f} ({bottleneck('dual')}-bound), cl=0 floor {PIM_FLOOR_NS_LEAF:.4f}")
    print(f"# {'N':>6} {'t':>3} {'leaves':>11} | {'GPU ms':>8} {'ns/lf':>6} | "
          f"{'PIM-1alu ms':>11} {'x':>5} | {'PIM-2alu ms':>11} {'x':>5} | flags")
    for lgN in (16, 18, 20, 22, 24):
        N = 1 << lgN
        for t in (4, 8, 16, 32):
            leaves = 2 * N * t
            g_ns = GPU_NS_LEAF[(lgN, t)]
            gpu_ms = leaves * g_ns / 1e6
            pim1_ms = leaves * s / 1e6
            pim2_ms = leaves * d / 1e6
            wall = "W" if 2 * (2 * N // t) * 16 >= BANK_BYTES else ""
            print(f"  {'2^%d' % lgN:>6} {t:>3} {leaves:>11} | {gpu_ms:8.2f} {g_ns:6.3f} | "
                  f"{pim1_ms:11.2f} {gpu_ms/pim1_ms:4.1f}x | "
                  f"{pim2_ms:11.2f} {gpu_ms/pim2_ms:4.1f}x | {wall}")
        print()
    print("# W = capacity wall: double-buffered leaf region 2 x (2N/t x 16B) >= one")
    print("#     bank (128MB) -> needs the block-streaming schedule (measured FASTER")
    print("#     than flat on HBM, tree_sweeps_blocked.txt; conservative here).")
    print("# GPU column is silicon (kernel-only). PIM columns are Ramulator2 GDDR6")
    print("# full-load measurements; SM-stream contention upper bounds: "
          f"single {SM_CONTENTION['single']}, dual {SM_CONTENTION['dual']}.")


if __name__ == "__main__":
    main()
