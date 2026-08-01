#!/usr/bin/env python3
"""
PIM || GPU co-execution timeline model for Ring-LPN PCG-OLE.

ARCHITECTURE (corrected): the PIM is near-bank compute INSIDE the GPU's own HBM3
stacks (HBM-PIM), and the CUDA SMs are on the SAME die sharing the SAME HBM.
There is NO PCIe hand-off: g is written by near-bank PIM into HBM cells and read
by the SMs from HBM. Two lanes share one HBM, but use DIFFERENT bandwidth domains:

  near-bank PIM lane : DPF expand + leaf_convert -> uses BANK-INTERNAL bandwidth;
                       the 2c^2*N*t leaf blocks never touch the HBM->SM bus.
  SM-NTT lane        : forward NTT(g) + pointwise-MAC + one INTT -> uses the
                       HBM->SM external bus + SM compute.
  g hand-off         : HBM-resident. Cost = leaf_convert's g-write (bus) + NTT's
                       g-read (already inside T_GPU). No transfer.

Steady-state (pipelined over c^2 pairs and many OLE instances):
  T_pair = max(T_PIM, T_GPU)  [+ HBM-bus contention penalty if the two lanes'
                               bus traffic together exceed HBM bandwidth]
  serial baseline A->B = sum(T_PIM) + sum(T_GPU).
Bottleneck = argmax lane. Balance: PU count s.t. T_PIM <= T_GPU.

All timings are analytical, driven by MEASURED tables (cited). GPU NTT table is
from RTX 5000 Ada (docs/merge_vs_4step.md); target is a hypothetical HBM3 GPU
(H100/H200 class) -> an SM/NTT scale factor lets you re-point it. PIM ns/leaf is
the 64-PU HBM3 model (pim/results/tree_sweeps_leafconv32.txt).
"""
import argparse

# ---- MEASURED INPUTS -------------------------------------------------------

# PIM full DPF (expand + leaf_convert), 64 PU, ChaCha8, HBM3 tCK=625ps, 1.6GHz.
# Calibrated to tree_sweeps_leafconv32.txt: n=18 full = 0.450 ns/leaf; near-asymptote.
PIM_NS_PER_LEAF_64PU = 0.450
# Two-tier variant (--mau): leaf_convert on the per-channel MAU (base-die fused
# butterfly pipe, sparse-prime reduction) instead of the bank PU; bank PU does
# expansion only, MAU column streams overlap with the next instance's expansion.
# Measured: pim/results/tree_sweeps_mau.txt (instances mode, cl=272 8r8w).
PIM_NS_PER_LEAF_MAU = 0.401  # measured: instances mode n=12, +5% bank contention
                             # over the 0.388 expansion-only floor (2-bank org)
PIM_PU_BASE = 64
# Cross-channel scaling is linear (measured 65.6x for 64ch); within-channel bank
# scaling is capped (~4.3x/8 measured). We scale PU by channels (linear) up to the
# stack's channel count, then note the within-channel cap if PU/ch > 1.
PIM_WITHIN_CH_EFF = 4.3 / 8.0   # measured efficiency when packing banks in one ch

# GPU merge-NTT FULL poly-mul (fwd+pointwise+inv), device-time, RTX 5000 Ada.
# docs/merge_vs_4step.md:145-155. ms per single poly-mul of ring degree N.
GPU_MERGE_POLYMUL_MS = {
    4096: 0.0255, 8192: 0.0323, 16384: 0.0504, 32768: 0.0816,
    65536: 0.1486, 131072: 0.3052, 262144: 0.7108, 524288: 2.098,
    1048576: 5.456,
}
# Move2 needs only the FORWARD NTT (a*a precomputed resident). A full negacyclic
# poly-mul = 2 fwd + 1 pointwise + 1 inv ~= 3 transforms; fwd ~= full/3 (estimate;
# refine with PCG_4STEP_STAGE_MS=1). pointwise is O(N), tiny vs transform.
FWD_FRACTION = 1.0 / 3.0

# HBM3 external bus bandwidth (bytes/s), the HBM->SM domain. Parameterized.
# L40S = GDDR6 (for --gddr mode: near-bank PIM in the GDDR dies, AiM-style).
HBM3_BUS_BW = {"H100": 3.35e12, "H200": 4.8e12, "A100": 2.0e12, "L40S": 0.864e12}

# --gddr mode: L40S full-card PIM, 24ch x 8PU = 192 PU (GDDR6_L40S org),
# die-level MAU (1 per DRAM die = per 2 channels; needs 2/die below ~1GHz MAU
# clock at dual-ALU rates -- see gen_dpf_tree_trace.py [mau-die] check).
# ns/leaf MEASURED by the Ramulator2 full-load sweeps
# (pim/results/tree_sweeps_gddr6.txt): single = cl272 PU, dual = cl155
# (dual-ALU XADD/VXORL PU, requires in-row packing).
PIM_NS_LEAF_GDDR192 = {"single": None, "dual": None}   # filled from sweeps

# All-GPU baseline: measured L40S DPF expand + leaf_convert asymptote (AES PRG,
# incl. D2H; pim/results/gpu_dpf_leafconv.txt). GPU DPF is MEMORY-bound (AES vs
# ChaCha <3%), so we scale it to the target GPU by memory-bandwidth ratio.
GPU_DPF_NS_LEAF_L40S = 2.9
L40S_BW = 0.864e12

# --- sim-grounded baseline (--baseline simgrounded): BOTH lanes measured on the
# SAME simulated machine (Accel-Sim, SM80_A100 config, 1.41 GHz) by replaying
# L40S-captured SASS traces (endtoend/results.md). Kernel-only, device-resident
# equivalent (memcpys excluded). This is the iso-device baseline: the only
# difference between baseline and design is WHERE the DPF executes.
# DPF ns/leaf: replayed grid (same 2^18 leaves, three tree shapes) gave
# 0.912 (n=12,B=64) / 1.399 (n=14,B=16) / 1.472 (n=16,B=4) ns/leaf; the spread
# is mostly the bench's single-bin atomicAdd in leaf_convert (contention grows
# with D; the real protocol scatters to distinct positions) plus shallow-level
# underutilization at large n. We take the FASTEST shape -- most favorable to
# the all-GPU baseline, i.e. conservative for our speedup claims.
A100SIM_DPF_NS_LEAF = 0.912
A100SIM_POLYMUL_65536_MS = 0.0561  # 79,108 cyc @ 1.41 GHz (merge, N=65536)
A100SIM_FWD_FRACTION = 0.304   # measured in the same replay (vs 1/3 assumed)

WORD = 8  # bytes per Z_p coefficient (uint64)
LEAF = 16 # bytes per 128b DPF leaf block

# ---- NTT stage-1 offload (4-step) on PIM ----------------------------------
# butterfly = 1 modmul + modadd + modsub on 62b values. With the established
# assumption modmul = 32 CK per 8-lane batch (~4 CK/modmul effective), plus
# add/sub and a per-batch twiddle VLD (twiddles are NOT broadcast constants:
# stage s has 2^s distinct values -> per-bank resident table, 1 VLD/batch):
BF_CK_PER_PU = 6.0      # cycles per butterfly per PU (incl. twiddle load)
TCK_NS = 0.625          # 1.6 GHz PU clock
N1_CAP = 1 << 14        # max bank-local column-NTT size (fits bank rows easily)


def gpu_fwd_ntt_ms(N, sm_scale):
    """Forward NTT(N) ms on the target GPU. Interpolate the measured full poly-mul
    table (log-log), take fwd fraction, apply SM scale (target/RTX throughput)."""
    keys = sorted(GPU_MERGE_POLYMUL_MS)
    if N <= keys[0]:
        full = GPU_MERGE_POLYMUL_MS[keys[0]] * (N / keys[0])
    elif N >= keys[-1]:
        full = GPU_MERGE_POLYMUL_MS[keys[-1]] * (N / keys[-1])  # ~linear-ish tail
    else:
        lo = max(k for k in keys if k <= N)
        hi = min(k for k in keys if k >= N)
        if lo == hi:
            full = GPU_MERGE_POLYMUL_MS[lo]
        else:
            import math
            f = (math.log(N) - math.log(lo)) / (math.log(hi) - math.log(lo))
            full = GPU_MERGE_POLYMUL_MS[lo] * (GPU_MERGE_POLYMUL_MS[hi] / GPU_MERGE_POLYMUL_MS[lo]) ** f
    return full * FWD_FRACTION / sm_scale


def pim_pair_ms(N, t, c, PU):
    """Near-bank PIM time for ONE (i,j) pair: expand + leaf_convert all B=t^2
    instances of domain D=2N/t -> leaves/pair = B*D = 2*N*t (per pair, per c^2)."""
    leaves = 2 * N * t                      # = t^2 * (2N/t)
    # PU scaling: linear across channels; base 64 PU. If PU is achieved by packing
    # banks within a channel beyond 1 PU/ch, apply the measured within-ch efficiency.
    eff_pu = PU
    ns = leaves * PIM_NS_PER_LEAF_64PU / (eff_pu / PIM_PU_BASE)
    return ns / 1e6                         # ns -> ms


def gpu_pair_ms(N, c, sm_scale):
    """SM-NTT time for ONE pair after Move2: 2 forward NTT(N) per pair (a*a resident,
    g needs 1 fwd; the pair's 2 muls share). Conservatively 2 fwd/pair; +pointwise
    negligible. The single INTT/instance is amortized (added once per instance below)."""
    return 2 * gpu_fwd_ntt_ms(N, sm_scale)


def bus_traffic_pair_bytes(N, c):
    """HBM->SM bus bytes touched per pair: leaf_convert writes g (N words, folded),
    NTT reads g + a*a spectra + writes z_hat accumulation. ~ (g_write + 3 reads) N words."""
    return (1 + 3) * N * WORD   # g write + (g, aa, z_hat) reads, order-of-magnitude


def ntt_split(N):
    """4-step split N = N1 x N2. N1 = bank-local column-NTT size (PIM does the
    N2 column NTTs of size N1 = stage-1); keep N2 >= 64 columns for PU parallelism."""
    import math
    n = int(math.log2(N))
    N1 = min(N1_CAP, N >> 6)          # N2 = N/N1 >= 64
    f1 = math.log2(N1) / n            # stage-1 fraction of butterflies (~GPU-time fraction)
    return N1, f1


def pim_ntt_stage1_ms(N, PU):
    """PIM time to do stage-1 (all column NTTs) of the pair's 2 forward NTTs."""
    N1, _ = ntt_split(N)
    import math
    bf = 2 * (N / 2) * math.log2(N1)  # 2 fwd NTTs x (N/2)*log2(N1) butterflies
    return bf * BF_CK_PER_PU * TCK_NS / PU / 1e6


def a100sim_polymul_ms(N):
    """Full poly-mul on the simulated A100: anchored at the replayed N=65536
    point, scaled to other N by the measured merge table's shape (labeled)."""
    ref = GPU_MERGE_POLYMUL_MS[65536]
    keys = sorted(GPU_MERGE_POLYMUL_MS)
    if N in GPU_MERGE_POLYMUL_MS:
        shape = GPU_MERGE_POLYMUL_MS[N] / ref
    elif N > keys[-1]:
        shape = GPU_MERGE_POLYMUL_MS[keys[-1]] / ref * (N / keys[-1])
    else:
        import math
        lo = max(k for k in keys if k <= N); hi = min(k for k in keys if k >= N)
        f = (math.log(N) - math.log(lo)) / (math.log(hi) - math.log(lo))
        shape = (GPU_MERGE_POLYMUL_MS[lo] *
                 (GPU_MERGE_POLYMUL_MS[hi] / GPU_MERGE_POLYMUL_MS[lo]) ** f) / ref
    return A100SIM_POLYMUL_65536_MS * shape


def model_point(N, t, c, PU, gpu, sm_scale, baseline="bwscale"):
    pairs = c * c
    t_pim = pim_pair_ms(N, t, c, PU)          # per pair
    if baseline == "simgrounded":
        # one machine everywhere: the design's SM-NTT lane uses the same
        # A100-sim numbers as the baseline's NTT term.
        t_gpu = 2 * a100sim_polymul_ms(N) * A100SIM_FWD_FRACTION
    elif baseline == "l40s":
        # iso-device GDDR mode: L40S silicon NTT (kernel-only battery table).
        from superiority_map import polymul_l40s_us
        t_gpu = 2 * polymul_l40s_us(N)[0] / 1e3 * FWD_FRACTION
    else:
        t_gpu = gpu_pair_ms(N, c, sm_scale)   # per pair
    bw = HBM3_BUS_BW[gpu]
    t_bus = bus_traffic_pair_bytes(N, c) / bw * 1e3   # ms per pair (contention lane)
    # steady state per pair = max of the resource lanes it occupies:
    #   PIM lane (bank-internal) and GPU lane (SM) overlap; the SHARED bus carries
    #   only t_bus, which overlaps with SM compute but competes with itself.
    t_lane = max(t_pim, t_gpu, t_bus)
    intt_ms = (a100sim_polymul_ms(N) * A100SIM_FWD_FRACTION
               if baseline == "simgrounded" else
               gpu_fwd_ntt_ms(N, sm_scale))   # one INTT per instance (~1 fwd), amortized
    total_overlap = pairs * t_lane + intt_ms  # +fill/drain of one lane (~t_lane) omitted (small vs pairs*)
    total_serial = pairs * (t_pim + t_gpu) + intt_ms
    speedup = total_serial / total_overlap
    # bottleneck lane
    lane = max((t_pim, "PIM"), (t_gpu, "GPU-NTT"), (t_bus, "HBM-bus"))[1]
    # balance: PU needed so T_PIM <= T_GPU
    pu_balance = PU * (t_pim / t_gpu) if t_gpu > 0 else float('inf')

    # --- NTT stage-1 offload (4-step): PIM takes fraction x of stage-1
    # butterflies with its spare cycles; GPU keeps (1-x) stage-1 + all stage-2.
    # Choose x in [0,1] minimizing the per-pair max. psi-twist fusion into
    # leaf_convert is assumed (saves the GPU an elementwise pass, ~free).
    _, f1 = ntt_split(N)
    w1_gpu = t_gpu * f1               # GPU-time of stage-1 if GPU does it
    w2_gpu = t_gpu * (1 - f1)         # GPU-time of stage-2 (incl. transpose read)
    w1_pim = pim_ntt_stage1_ms(N, PU) # PIM-time of stage-1 if PIM does all of it
    best_x, best_T = 0.0, t_lane
    for i in range(101):
        x = i / 100.0
        T = max(t_pim + x * w1_pim, (1 - x) * w1_gpu + w2_gpu, t_bus)
        if T < best_T:
            best_x, best_T = x, T
    off_gain = t_lane / best_T
    off_lane = max((t_pim + best_x * w1_pim, "PIM"),
                   ((1 - best_x) * w1_gpu + w2_gpu, "GPU-NTT"),
                   (t_bus, "HBM-bus"))[1]
    # --- all-GPU baseline: same target GPU does DPF+leaf_convert AND the NTT,
    # serially on the same SM pool (no second resource to overlap with).
    if baseline == "simgrounded":
        # iso-device: DPF ns/leaf and NTT both from Accel-Sim A100-config
        # replays of L40S SASS traces (endtoend/results.md), kernel-only.
        ns_gpu_dpf = A100SIM_DPF_NS_LEAF
        full_ms = a100sim_polymul_ms(N)
        t_gpu_base = 2 * full_ms * A100SIM_FWD_FRACTION   # NTT lane, same machine
        t_allgpu = (2 * N * t) * ns_gpu_dpf / 1e6 + t_gpu_base
    elif baseline == "l40s":
        # iso-device GDDR: silicon DPF+lc regime curve (baseline_v2 battery).
        from superiority_map import dpf_ns_leaf_l40s
        ns_gpu_dpf = dpf_ns_leaf_l40s(2 * N * t)
        t_allgpu = (2 * N * t) * ns_gpu_dpf / 1e6 + t_gpu
    else:
        # bw-scale mode (legacy): L40S measurement scaled by bandwidth ratio.
        ns_gpu_dpf = GPU_DPF_NS_LEAF_L40S * (L40S_BW / bw)
        t_allgpu = (2 * N * t) * ns_gpu_dpf / 1e6 + t_gpu   # per pair, ms
    vs_gpu = t_allgpu / best_T                          # co-exec(+offload) speedup
    return dict(t_pim=t_pim, t_gpu=t_gpu, t_bus=t_bus, lane=lane,
                overlap_ms=total_overlap, serial_ms=total_serial,
                speedup=speedup, pu_balance=pu_balance,
                off_x=best_x, off_T=best_T, off_gain=off_gain, off_lane=off_lane,
                t_allgpu=t_allgpu, vs_gpu=vs_gpu)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", choices=list(HBM3_BUS_BW), default="H100")
    ap.add_argument("--sm-scale", type=float, default=3.0,
                    help="target-GPU SM/NTT throughput vs the RTX5000-Ada table (H100~3x)")
    ap.add_argument("--c", type=int, default=2)
    ap.add_argument("--baseline", choices=["bwscale", "simgrounded"], default="bwscale",
                    help="all-GPU baseline source: L40S x BW-ratio (legacy) or "
                         "Accel-Sim A100-config replays (iso-device)")
    ap.add_argument("--mau", action="store_true",
                    help="two-tier PIM: leaf_convert on the per-channel MAU; "
                         "bank PU does expansion only (0.401 ns/leaf measured)")
    ap.add_argument("--gddr", choices=["single", "dual"], default=None,
                    help="L40S GDDR6 full-card mode: 192 PU (24ch x 8PU), "
                         "measured GDDR sweep ns/leaf (single=cl272 PU, "
                         "dual=cl155 dual-ALU PU), L40S silicon NTT + DPF "
                         "baseline, 864 GB/s bus")
    ap.add_argument("--contention", type=float, default=0.0,
                    help="bank-level PIM-lane slowdown from co-resident SM "
                         "traffic. Measured Level-1 bounds (tree_sweeps_"
                         "smstream.txt): 0.03 lower (bank-slot share) .. 0.10 "
                         "upper (frontend-serialized); true value needs "
                         "phase-2 out-of-band SM injection")
    a = ap.parse_args()
    global PIM_NS_PER_LEAF_64PU, PIM_PU_BASE
    if a.gddr:
        ns = PIM_NS_LEAF_GDDR192[a.gddr]
        if ns is None:
            ap.error("PIM_NS_LEAF_GDDR192[%r] not yet filled from "
                     "tree_sweeps_gddr6.txt" % a.gddr)
        PIM_NS_PER_LEAF_64PU = ns       # measured at 192 PU ->
        PIM_PU_BASE = 192               # PU/BASE division = 1 at PU=192
        a.gpu, a.baseline, a.sm_scale = "L40S", "l40s", 1.0
    elif a.mau:
        PIM_NS_PER_LEAF_64PU = PIM_NS_PER_LEAF_MAU
    PIM_NS_PER_LEAF_64PU *= (1.0 + a.contention)

    Ns = [1 << 16, 1 << 20, 1 << 24]
    ts = [4, 8, 16, 32]
    PUs = [192] if a.gddr else [64, 256, 512, 1024]
    print(f"# PIM||GPU co-exec model (shared HBM, no PCIe). gpu={a.gpu} "
          f"bus={HBM3_BUS_BW[a.gpu]/1e12}TB/s sm_scale={a.sm_scale} c={a.c}")
    print(f"# PIM=64PU 0.45ns/leaf (measured); NTT=merge RTX5000Ada/{a.sm_scale} (fwd=full/3)")
    print(f"# offload: PIM takes x of NTT stage-1 (4-step, N1<=2^14, "
          f"{BF_CK_PER_PU:.0f}CK/butterfly/PU)")
    if a.baseline == "simgrounded":
        print(f"# all-GPU baseline: SIM-GROUNDED (Accel-Sim SM80_A100 replays of L40S "
              f"SASS): DPF {A100SIM_DPF_NS_LEAF}ns/leaf, poly-mul(65536)="
              f"{A100SIM_POLYMUL_65536_MS}ms, fwd={A100SIM_FWD_FRACTION}")
    else:
        print(f"# all-GPU baseline: same GPU does DPF+lc serially before NTT; DPF "
              f"scaled from L40S {GPU_DPF_NS_LEAF_L40S}ns/leaf by BW ratio (mem-bound)")
    print(f"# {'N':>8} {'t':>3} {'PU':>5} | {'T_PIM ms':>9} {'T_GPU ms':>9} {'T_bus ms':>9} "
          f"| {'lane':>8} {'PU@bal':>7} || {'x*':>4} {'T_off ms':>9} {'off-lane':>8} {'gain':>5} "
          f"|| {'allGPU ms':>9} {'vs GPU':>6}")
    for N in Ns:
        for t in ts:
            for PU in PUs:
                r = model_point(N, t, a.c, PU, a.gpu, a.sm_scale, a.baseline)
                print(f"  {N:>8} {t:>3} {PU:>5} | {r['t_pim']:9.4f} {r['t_gpu']:9.4f} "
                      f"{r['t_bus']:9.4f} | {r['lane']:>8} {r['pu_balance']:7.0f} "
                      f"|| {r['off_x']:4.2f} {r['off_T']:9.4f} {r['off_lane']:>8} "
                      f"{r['off_gain']:4.2f}x || {r['t_allgpu']:9.4f} {r['vs_gpu']:5.1f}x")
        print()


if __name__ == "__main__":
    main()
