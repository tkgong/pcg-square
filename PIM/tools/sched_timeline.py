#!/usr/bin/env python3
"""Discrete-event timeline for the five software-scheduler optimizations
coupling the PIM DPF lane with the SM NTT lane (new Beaver algorithm,
no MAU / no multiplier; NTT entirely on the SM).

Schedules (cumulative ablation):
  S0  serial            : all-blocks PIM, then all-blocks NTT (no overlap)
  S1  block pipeline    : block k's NTT runs while block k+1 expands (PIM/NET/SM
                          three-lane pipeline, block granularity)
  S2  + Move2 streaming : per-block NTT = fwdNTT(g_ij) + pointwise-MAC into an
                          NTT-domain accumulator; ONE INTT at the very end
                          (a-hat = NTT(a_i*a_j) precomputed, a public)
  S3  + per-round Beaver: the 2 Beaver opens fire per ROUND (as each round's
                          sums complete) instead of per block -> network hides
                          inside the block instead of gating its end
  S4  + doorbell rounds : SM consumes round r-1's C region while PIM writes
                          round r -> SM lane starts mid-block; PIM pays the
                          measured opposite-parity contention factor f_opp
  S5  + parity discipline: S4 scheduled deliberately (opp parity) vs the bad
                          case (same parity, f_same) -- reported as S4(bad)
                          vs S5(good); the delta is opt-5's value

Calibration inputs (all measured):
  PIM   : Ramulator2 L40S chacha cycles (v5 battery; per-round = total/rounds,
          round op-mix uniformity verified) + contention factors f_opp/f_same
  SM    : L40S silicon merge-backend stage split (PCG_MERGE_STAGE_MS=1,
          batch=16): FWD16/PW16/INTT16 per-mul us, logN 14..24
  NET   : RTT parameter (Beaver opens); separate lane per the coexec-doc
          convention (never folded into compute lanes)

Cross-device: PIM per-round scales by (none-anchor cycles x tCK / channels),
NTT tables scale by the NTT16 device ratios (est for non-L40S).
"""
import math, os, re, sys

# ---- L40S silicon: merge-backend stage split (batch=16, per-mul us) ---------
# Measured this campaign (PCG_MERGE_STAGE_MS=1, ntt_batch_bench --backend merge
# --batch 16 --iters 5). fwd = avg(fwd_a, fwd_b)/16; sums match device_ms +-1%.
FWD16 = {14: 0.81, 16: 2.48, 18: 9.10, 20: 75.4, 22: 354.0, 24: 1453.0}
PW16 = {14: 0.26, 16: 0.46, 18: 1.34, 20: 37.9, 22: 150.0, 24: 606.0}
INTT16 = {14: 0.77, 16: 2.36, 18: 9.15, 20: 78.1, 22: 358.0, 24: 1470.0}
NTT16 = {14: 2.00, 16: 6.18, 18: 27.1, 20: 266.4, 22: 1197.6, 24: 4952.6}

# ---- GPU DPF device factors + 4step tables (same numbers as e2e_secparams) --
GPU_DEV_FACTOR = {"bc": 1.0, "h200h": 0.55, "b200": 0.49,
                  "a100": 0.74, "ada5k": 1.30, "gddr7": 0.79 * 864/1792 + 0.21*0.9}
HBM_DEVS = {"a100", "h200h", "b200"}
# L40S 4step ms/mul + transpose ms (b=16, MEASURED 2^20..2^24 only);
# B200 measured (ms, transp frac). No extrapolation below 2^20.
FOURSTEP_BC16 = {20: (0.4980, 0.163), 22: (2.2254, 0.702), 24: (9.7661, 3.061)}
FOURSTEP_B20016 = {20: (0.1730, .245), 22: (0.7225, .231), 24: (3.0532, .217)}


def fourstep16(dev, lg, nttr, dru):
    """4step full-mul ms (b=16) on device; dru removes transpose (HBM only).
    Returns None where unmeasured (lg < 20)."""
    if lg not in FOURSTEP_BC16:
        return None
    if dev == "b200":
        ms, tp = FOURSTEP_B20016[lg]
        tms = ms * tp
    else:
        ms, tms = FOURSTEP_BC16[lg]
        ms, tms = ms * nttr, tms * nttr
    return (ms - tms) if (dru and dev in HBM_DEVS) else ms

# ---- device map + NTT device ratios (vs L40S; measured h200/b200, est rest) -
DEV = {
    "bc":    ("L40S",       0.444,  24, 1, 1.00, False),
    "ada5k": ("RTX5000Ada", 0.444,  16, 1, 1.35, True),
    "gddr7": ("RTXPRO6000", 0.571,  64, 1, 0.70, True),
    "a100":  ("A100-80",    0.625,  80, 1, 0.73, True),   # geomean ratio
    "h200h": ("H200",       0.625,  96, 2, 0.53, True),   # NTT16 h200/bc @2^20
    "b200":  ("B200",       0.500, 128, 2, 0.49, True),
}
LEAVES_PER_CH = 8 * 4 * 4096          # n=12, 4 rounds, P=8
ROUNDS_SIM = 4                        # the sim battery's rounds
BANK_BYTES = 128 << 20


def cyc(path):
    try:
        for line in open(path):
            m = re.search(r"memory_system_cycles:\s*(\d+)", line)
            if m:
                return int(m.group(1))
    except OSError:
        pass
    return None


def schedule(nblk, t_pim_blk, t_round, t_sm_blk, t_intt, rtt, mode, f=0.0):
    """Event-driven pipeline over nblk blocks.
    t_pim_blk: PIM time to produce ONE block's leaves (= T_pim_total/nblk;
               contention-inflated by f in the 'round' modes where the SM
               consume stream overlaps the expansion).
    t_round  : time between doorbells (T_pim_total / total_rounds). One round
               may span several blocks (t small) or one block several rounds
               (t large) -- only the doorbell CADENCE matters here.
    t_sm_blk : SM time per block (2 full muls, or fwd+pw for Move2).
    t_intt   : one INTT at the very end (Move2) else 0.
    rtt      : one network round-trip; 2 Beaver opens gate each block's
               consumability in 'block' mode; in 'round' mode the opens ride
               the round cadence and only the last round's can stick out.
    mode     : 'serial' | 'block' | 'round'.
    """
    if mode == "round":
        t_pim_blk = t_pim_blk * (1 + f)
    if mode == "serial":
        return nblk * (t_pim_blk + 2 * rtt + t_sm_blk) + t_intt
    pim_free = 0.0
    sm_free = 0.0
    done = 0.0
    for b in range(nblk):
        pim_free += t_pim_blk
        if mode == "block":
            ready = pim_free + 2 * rtt
        else:
            ready = pim_free + max(0.0, 2 * rtt - t_round)
        start = max(ready, sm_free)
        sm_free = start + t_sm_blk
        done = sm_free
    return done + t_intt


def main():
    S = sys.argv[1] if len(sys.argv) > 1 else "."
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    base = cyc(os.path.join(S, "v5_ch465.out"))
    opp = cyc(os.path.join(S, "v5b_opp.out"))
    same = cyc(os.path.join(S, "v5b_same.out"))
    if not all((base, opp, same)):
        sys.exit("missing v5 sims (need v5_ch465/_opp/_same .out)")
    f_opp = opp / base - 1
    f_same = same / base - 1

    rtts = [0.0, 0.05, 2.0]      # ms: same-host, same-DC, WAN

    out = os.path.join(repo, "pim", "results", "sched_timeline.txt")
    with open(out, "w") as f:
        f.write("== FULL sw-hw-network co-design scheduling pipeline\n"
                "== SOFTWARE ladder: S0 serial | S1 block-pipe | S2 +Move2 |\n"
                "==   S3 +round-Beaver | S4 +doorbell(opp) | S5bad = same parity\n"
                "== HARDWARE columns: S1_4s = block-pipe on 4step backend;\n"
                "==   S1_4sD = + DRU transpose offload (HBM only; measured\n"
                "==   stage split; '--' below 2^20 = unmeasured, no extrap)\n"
                "== NETWORK: rtt in {0, 0.05, 2} ms (same-host/same-DC/WAN);\n"
                "==   netexp = S4(rtt) - S4(rtt=0) = EXPOSED network time;\n"
                "==   hidden whenever 2*RTT <= t_round (round cadence).\n"
                f"== MEASURED contention: f_opp={f_opp*100:.2f}%  f_same={f_same*100:.2f}%"
                f"  (L40S sim, C-consume stream)\n"
                "== SM stages: L40S merge split (fwd/pw/intt, b=16); Move2 block\n"
                "== = 2 x (FWD16+PW16) [2 muls/block, same convention as e2e],\n"
                "== one INTT at end. GPU baseline DPF part device-scaled by\n"
                "== GPU_DEV_FACTOR (fix: was unscaled). c^2=16 blocks.\n==\n")
        # GPU new-algo curve (measured, L40S) for the baseline
        GPU_BC = [(0.52e6, .720), (1.05e6, .470), (4.2e6, .247), (16.8e6, .216),
                  (67.1e6, .216), (268e6, .218), (1.07e9, .220)]

        def interp(curve, x):
            if x <= curve[0][0]:
                return curve[0][1]
            if x >= curve[-1][0]:
                return curve[-1][1]
            for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
                if x0 <= x <= x1:
                    t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
                    return y0 * (y1 / y0) ** t
            return curve[-1][1]

        for dev, (lab, tck, C, card, nttr, est) in DEV.items():
            none_c = cyc(os.path.join(S, f"v2_{dev}_cl155_none.out"))
            if none_c is None:
                continue
            ratio = base / cyc(os.path.join(S, "v4_bc_n12_none.out"))
            pim_ns = none_c * ratio * tck / (C * LEAVES_PER_CH * card)
            f.write(f"-- {lab}{' est' if est else ''}: PIM DPF+convert "
                    f"{pim_ns:.4f} ns/leaf, NTT ratio {nttr:.2f}\n")
            f.write(f"   {'N':>5} {'t':>4} {'rtt':>5} | {'GPU':>8} |"
                    f" {'S0':>7} {'S1':>7} {'S2':>7} {'S3':>7} {'S4':>7} |"
                    f" {'S5bad':>7} | {'S1_4s':>7} {'S1_4sD':>7} |"
                    f" {'netexp':>7} | x(S4)\n")
            for lgN in (16, 18, 20, 22, 24):
                N = 1 << lgN
                for t in (4, 16, 128):
                    # per-OLE-cell accounting: c^2 = 16 blocks, one block =
                    # t^2 instances = 2Nt leaves. BOTH lanes cover all 16.
                    leaves_blk = 2 * N * t
                    leaves_all = 16 * leaves_blk
                    t_pim_blk = leaves_blk * pim_ns / 1e6        # ms
                    t_pim_total = 16 * t_pim_blk
                    total_rounds = max(1, 16 * t * t // (C * 8))
                    t_round = t_pim_total / total_rounds
                    # GPU baseline: as-implemented 2 full muls per block,
                    # DPF part device-scaled (F3 fix).
                    gf = GPU_DEV_FACTOR[dev]
                    ntt_full = 2 * NTT16[lgN] * nttr / 1e3       # ms per block
                    # Move2 streamed: a-hat precomputed; per block TWO
                    # fwd+pointwise-MAC (2 muls/block, F2 fix); one INTT at end.
                    ntt_strm = 2 * (FWD16[lgN] + PW16[lgN]) * nttr / 1e3
                    intt = INTT16[lgN] * nttr / 1e3
                    gpu = leaves_all * interp(GPU_BC, leaves_all) * gf / 1e6 \
                        + 16 * ntt_full
                    # hardware dimension: 4step backend block cost +- DRU
                    fs = fourstep16(dev, lgN, nttr, dru=False)
                    fsd = fourstep16(dev, lgN, nttr, dru=True)
                    # network-exposure reference: S4 at rtt=0
                    s4_0 = schedule(16, t_pim_blk, t_round, ntt_strm, intt,
                                    0.0, "round", f=f_opp)
                    for rtt in rtts:
                        s0 = schedule(16, t_pim_blk, t_round, ntt_full, 0, rtt, "serial")
                        s1 = schedule(16, t_pim_blk, t_round, ntt_full, 0, rtt, "block")
                        s2 = schedule(16, t_pim_blk, t_round, ntt_strm, intt, rtt, "block")
                        s3 = schedule(16, t_pim_blk, t_round, ntt_strm, intt, rtt, "round")
                        s4 = schedule(16, t_pim_blk, t_round, ntt_strm, intt, rtt,
                                      "round", f=f_opp)
                        s5b = schedule(16, t_pim_blk, t_round, ntt_strm, intt, rtt,
                                       "round", f=f_same)
                        if fs is not None:
                            s1_4s = schedule(16, t_pim_blk, t_round, 2*fs, 0, rtt, "block")
                            s1_4sd = schedule(16, t_pim_blk, t_round, 2*fsd, 0, rtt, "block")
                            c4s, c4sd = f"{s1_4s:7.2f}", f"{s1_4sd:7.2f}"
                        else:
                            c4s = c4sd = f"{'--':>7}"
                        netexp = s4 - s4_0
                        wall = "W" if 2 * (2 * N // t) * 16 >= BANK_BYTES else " "
                        f.write(f"   2^{lgN:<3} {t:>4} {rtt:5.2f} | {gpu:8.2f} |"
                                f" {s0:7.2f} {s1:7.2f} {s2:7.2f} {s3:7.2f}"
                                f" {s4:7.2f} | {s5b:7.2f} | {c4s} {c4sd} |"
                                f" {netexp:7.3f} | {gpu/s4:5.1f}x {wall}\n")
            f.write("\n")
    print(f"wrote {out}\nf_opp={f_opp*100:.2f}%  f_same={f_same*100:.2f}%")


if __name__ == "__main__":
    main()
