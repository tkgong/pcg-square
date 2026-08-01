#!/usr/bin/env python3
"""New-algorithm PIM campaign collector (Beaver-corrected output layer).

Single PIM configuration -- NOT the old A/B/C MAU ablation:
  near-bank ARX PU + multiplier-free sparse-prime mod-ALU; the LAST tree level
  fuses expand + per-child ChaCha8 out-hash H' + mod-P convert (REDP) +
  partial-sum (ADDP) + 8B write; NTT entirely on the SM. No MAU, no per-PU
  multiplier, no DRU.

Inputs (scratchpad):
  v4_bc_n12_none.out / v4_bc_n12_chacha{310,465,620}.out  (L40S full-load sim)
  v2_{dev}_cl155_none.out                                 (6-device none anchors)
GPU baseline: new DPF+convert regime curve, MEASURED on L40S silicon
  (endtoend/dpf_real_bench.cu, new dpf_out_sums/scatter_g); 5 other devices
  scaled by the per-device old-silicon ratio (flagged est). NTT unchanged.

Outputs:
  pim/results/newalgo_pim.txt       (per-device PIM DPF+convert ns/leaf)
  pim/results/newalgo_speedup.txt   (6 dev x 20 (N,t) speedup vs all-GPU)
"""
import math, os, re, sys

# dev -> (label, tck_ns, channels, PUs_sim, card_factor)
DEV = {
    "bc":    ("L40S",         0.444,  24,  192, 1),
    "ada5k": ("RTX5000Ada",   0.444,  16,  128, 1),
    "gddr7": ("RTXPRO6000",   0.571,  64,  512, 1),
    "a100":  ("A100-80",      0.625,  80,  640, 1),
    "h200h": ("H200",         0.625,  96,  768, 2),
    "b200":  ("B200",         0.500, 128, 1024, 2),
}
LEAVES_PER_CH = 8 * 4 * 4096      # n=12, 4 rounds, P=8 (matches v2 basis)

# GPU baseline: NEW DPF+convert (ChaCha8 out-hash) regime curve, ns/leaf vs
# total leaves. MEASURED on L40S (endtoend/dpf_real_bench.cu, this campaign).
GPU_BC = [(0.52e6, .720), (1.05e6, .470), (4.2e6, .247), (16.8e6, .216),
          (67.1e6, .216), (268e6, .218), (1.07e9, .220)]
# Old-silicon per-device ratio (measured h200/b200; est a100/ada5k/gddr7) applied
# to the NEW L40S absolute -> transfers inter-device positioning. All non-bc est
# (new-algo silicon only on L40S).
OLD_BC   = [.188, .134, .108, .118, .163, .174, .186, .189, .197, .195]
OLD_H200 = [.205, .143, .112, .102, .078, .077, .070, .069, .065, .064]
OLD_B200 = [.203, .127, .092, .073, .065, .062, .055, .054, .050, .049]
def _ratio(dev_old, x):   # geomean-ish avg ratio dev/bc across the old curve
    return sum(d / b for d, b in zip(dev_old, OLD_BC)) / len(OLD_BC)
DEV_GPU_FACTOR = {
    "bc": 1.0,
    "h200h": _ratio(OLD_H200, 0), "b200": _ratio(OLD_B200, 0),
    "a100": math.sqrt(_ratio(OLD_H200, 0) * 1.0),          # geomean(L40S,H200)
    "ada5k": 1.30, "gddr7": 0.79 * 864 / 1792 + 0.21 * 0.9,
}
def gpu_curve(dev):
    f = DEV_GPU_FACTOR[dev]
    return [(x, y * f) for x, y in GPU_BC]
EST_GPU = {"ada5k", "gddr7", "a100", "h200h", "b200"}   # new silicon only on bc

NTT16 = {   # unchanged (NTT not touched by the new algo); us/mul batch-16
    "bc":   {14: 2.00, 16: 6.18, 18: 27.1, 20: 266.4, 22: 1197.6, 24: 4952.6},
    "h200h": {14: 2.54, 16: 8.06, 18: 33.7, 20: 141.4, 22: 605.7, 24: 2633.6},
    "b200": {14: 2.49, 16: 7.51, 18: 31.0, 20: 129.4, 22: 549.6, 24: 2449.8},
}
NTT16["a100"] = {k: math.sqrt(NTT16["bc"][k] * NTT16["h200h"][k]) for k in NTT16["bc"]}
NTT16["ada5k"] = {k: v * 1.35 for k, v in NTT16["bc"].items()}
NTT16["gddr7"] = {k: v * 0.70 for k, v in NTT16["bc"].items()}

# L40S silicon merge-backend stage split (PCG_MERGE_STAGE_MS=1, batch=16,
# per-mul us). Move2 streamed NTT lane: per block = fwd(g_ij) + pointwise-MAC;
# ONE INTT at the end (amortized /16 blocks). Other devices scale by the same
# NTT16 device ratio (stage mix assumed constant; est).
FWD16 = {14: 0.81, 16: 2.48, 18: 9.10, 20: 75.4, 22: 354.0, 24: 1453.0}
PW16 = {14: 0.26, 16: 0.46, 18: 1.34, 20: 37.9, 22: 150.0, 24: 606.0}
INTT16 = {14: 0.77, 16: 2.36, 18: 9.15, 20: 78.1, 22: 358.0, 24: 1470.0}


def ntt_lane_us(dev, lgN, sched):
    r = NTT16[dev][lgN] / NTT16["bc"][lgN]
    if sched == "streamed":     # Move2: a-hat precomputed -> ONE fwd(g)+pw per
        return (FWD16[lgN] + PW16[lgN] + INTT16[lgN] / 16) * r  # block, INTT/16
    return 2 * NTT16[dev][lgN] / NTT16["bc"][lgN] * NTT16["bc"][lgN]  # 2 full muls

BANK_BYTES = 128 << 20


def interp(curve, x):
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 * (y1 / y0) ** f
    return curve[-1][1]


def cyc(path):
    try:
        for line in open(path):
            m = re.search(r"memory_system_cycles:\s*(\d+)", line)
            if m:
                return int(m.group(1))
    except OSError:
        pass
    return None


def main():
    S = sys.argv[1] if len(sys.argv) > 1 else "."
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CCL = int(sys.argv[2]) if len(sys.argv) > 2 else 465   # primary convert-cl
    SCHED = sys.argv[3] if len(sys.argv) > 3 else "batch"  # batch | streamed

    # measured L40S ratios (full load n=12)
    bc_none = cyc(os.path.join(S, "v4_bc_n12_none.out")) or 389192
    ratios = {}
    for ccl in (310, 465, 620):
        c = cyc(os.path.join(S, f"v4_bc_n12_chacha{ccl}.out"))
        if c:
            ratios[ccl] = c / bc_none
    if CCL not in ratios:
        sys.exit(f"missing L40S chacha{CCL} sim (have {sorted(ratios)})")
    ratio = ratios[CCL]

    # per-device PIM DPF+convert ns/leaf (chacha/none ratio is device-independent
    # -- same bank-PU op structure; validated +-4% for none/fused/mau across orgs).
    pim_ns = {}
    for dev, (lab, tck, C, PU, card) in DEV.items():
        none_c = cyc(os.path.join(S, f"v2_{dev}_cl155_none.out"))
        if none_c is None:
            continue
        chacha_c = none_c * ratio
        pim_ns[dev] = chacha_c * tck / (C * LEAVES_PER_CH * card)

    out1 = os.path.join(repo, "pim", "results", "newalgo_pim.txt")
    with open(out1, "w") as f:
        f.write("== NEW-ALGORITHM PIM (Beaver output layer, no MAU, no multiplier)\n"
                "== near-bank ARX PU + sparse-prime mod-ALU (REDP/ADDP/CADDP/NEGP);\n"
                "== last level = fused CONVERT (expand + ChaCha8 H' + mod-P + 8B wr);\n"
                "== NTT entirely on SM. Full load n=12, 4 rounds, gated, broadcast.\n"
                f"== L40S MEASURED chacha{CCL}/none ratio = {ratio:.3f} "
                f"(sens: 310={ratios.get(310,0):.3f} 620={ratios.get(620,0):.3f}).\n"
                "== other devices: ratio x per-device none anchor x tck (channel\n"
                "== linearity +-4%). ns/leaf = card-level.\n==\n")
        f.write("dev         PU(card) | none    convert  conv/none | tck(ns) ch\n")
        for dev, (lab, tck, C, PU, card) in DEV.items():
            none_c = cyc(os.path.join(S, f"v2_{dev}_cl155_none.out"))
            if none_c is None or dev not in pim_ns:
                continue
            none_ns = none_c * tck / (C * LEAVES_PER_CH * card)
            f.write(f"{lab:<11} {PU*card:>4}   | {none_ns:.4f}  {pim_ns[dev]:.4f}   "
                    f"{pim_ns[dev]/none_ns:.3f}     | {tck:.3f}  {C}\n")

    out2 = os.path.join(repo, "pim", "results",
                        "newalgo_speedup_streamed.txt" if SCHED == "streamed"
                        else "newalgo_speedup.txt")
    with open(out2, "w") as f:
        f.write("== NEW-ALGORITHM speedup vs all-GPU, per (N,t), overlapped steady\n"
                "== T = max(T_pim_dpf+convert, T_ntt_sm). GPU serial = leaves x\n"
                "== newDPF+convert(leaves) + 2 x polymul16(N). Single PIM config\n"
                "== (no MAU/DRU/multiplier). GPU DPF+convert MEASURED on L40S;\n"
                f"== PIM NTT lane sched={SCHED}"
                " (streamed = Move2: fwd+pw per block, INTT/16).\n"
                f"== est = GPU baseline scaled (new silicon only on L40S). cl={CCL}.\n"
                "== W = capacity-wall cell (block-streaming, measured faster).\n==\n")
        for dev, (lab, tck, C, PU, card) in DEV.items():
            if dev not in pim_ns:
                f.write(f"-- {lab}: no sim anchor, skipped\n\n")
                continue
            dpf = pim_ns[dev]
            est = " est" if dev in EST_GPU else ""
            f.write(f"-- {lab} ({PU*card} PU){est}: PIM DPF+convert ns/leaf = {dpf:.4f}\n")
            f.write(f"   {'N':>6} {'t':>3} | {'GPU ms':>9} | {'x':>6} | lane flags\n")
            gc = gpu_curve(dev)
            for lgN in (16, 18, 20, 22, 24):
                N = 1 << lgN
                for t in (4, 8, 16, 32):
                    leaves = 2 * N * t
                    gdpf = leaves * interp(gc, leaves) / 1e6
                    gntt = 2 * NTT16[dev][lgN] / 1e3       # GPU baseline: 2 full muls
                    gpu = gdpf + gntt
                    lane_ntt = ntt_lane_us(dev, lgN, SCHED) / 1e3
                    tpim = leaves * dpf / 1e6
                    tov = max(tpim, lane_ntt)
                    lane = "DPF" if tpim >= lane_ntt else "NTT"
                    wall = "W" if 2 * (2 * N // t) * 16 >= BANK_BYTES else ""
                    f.write(f"   {'2^%d' % lgN:>6} {t:>3} | {gpu:9.2f} | "
                            f"{gpu/tov:5.1f}x | {lane:>4} {wall}\n")
            f.write("\n")
    print(f"wrote {out1}\nwrote {out2}\nL40S conv/none ratio (cl{CCL}) = {ratio:.3f}")


if __name__ == "__main__":
    main()
