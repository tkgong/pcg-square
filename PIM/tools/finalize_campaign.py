#!/usr/bin/env python3
"""
FINAL CAMPAIGN collector: 6 devices x 3 MAU cases x (N,t) grid.

Usage: finalize_campaign.py <scratch_dir_with_.out_files>

Inputs : Ramulator2 .out files (memory_system_cycles), naming
         <dev>_{cl272,cl155,cl0}_{none,mau,fused}[_sm|_ntt1|_ntt05|_ntt2] and
         <dev>_n14_cl155_mau, dev in {bc,ada5k,gddr7,a100,h200h,b200}.
Outputs: pim/results/final_campaign_pim.txt      (per-device ablation, sim tier)
         pim/results/final_campaign_speedup.txt  (3 cases x 6 devices x (N,t))

Cases: A = MAU does DPF lc + NTT stage-1 (SM keeps stage-2; DPF lane carries
           the stage-1 column contention, measured by the _ntt1 points)
       B = MAU does DPF lc only, NTT all on SM (DPF lane carries SM-traffic
           contention, measured by the _sm points)
       C = no MAU, modmul fused into the bank PU (fused points; SM contention
           delta borrowed from the mau _sm delta, flagged approx)

GPU baselines: measured silicon regime curves + batch-16 merge NTT tables for
L40S / H200 / B200; A100 / Ada5K / RTX PRO 6000 flagged EST (see notes inline).
"""
import math
import os
import re
import sys

# dev -> (label, tck_ns, channels, PUs_sim, card_factor)
DEV = {
    "bc":    ("L40S",         0.444,  24,  192, 1),
    "ada5k": ("RTX5000Ada",   0.444,  16,  128, 1),
    "gddr7": ("RTXPRO6000",   0.571,  64,  512, 1),
    "a100":  ("A100-80",      0.625,  80,  640, 1),
    "h200h": ("H200",         0.625,  96,  768, 2),   # half card simulated
    "b200":  ("B200",         0.500, 128, 1024, 2),   # one die-domain simulated
}

# ---- GPU silicon: DPF+lc regime curves (kernel-only ns/leaf vs total leaves) ----
# Sources: baseline_v2 batteries (l40s/h200/b200_realkernel_dpf.txt).
CURVES = {
    "bc":   [(0.52e6, .188), (1.05e6, .134), (2.1e6, .108), (4.2e6, .118),
             (8.4e6, .163), (16.8e6, .174), (33.6e6, .186), (67.1e6, .189),
             (134e6, .197), (1.07e9, .195)],
    "h200h": [(0.52e6, .205), (1.05e6, .143), (2.1e6, .112), (4.2e6, .102),
              (8.4e6, .078), (16.8e6, .077), (33.6e6, .070), (67.1e6, .069),
              (134e6, .065), (1.07e9, .064)],
    "b200": [(0.52e6, .203), (1.05e6, .127), (2.1e6, .092), (4.2e6, .073),
             (8.4e6, .065), (16.8e6, .062), (33.6e6, .055), (67.1e6, .054),
             (134e6, .050), (1.07e9, .049)],
}
# EST curves (flagged in output): A100 = geometric mean of L40S/H200 pointwise
# (HBM2e sits between GDDR6 and HBM3e in both BW and the compute mix; replace
# with the cluster battery when it lands). Ada5K = L40S x 1.35 in the DRAM
# regime (576 vs 864 GB/s on the 79%-memory-bound plateau), x1.05 launch-side.
# RTX PRO 6000 = L40S scaled by the memory-share model (79% mem x 864/1792 +
# 21% compute x ~0.9 SM ratio).
CURVES["a100"] = [(x, math.sqrt(a * b)) for (x, a), (_, b)
                  in zip(CURVES["bc"], CURVES["h200h"])]
CURVES["ada5k"] = [(x, v * (1.05 if x <= 2.1e6 else 1.35)) for x, v in CURVES["bc"]]
CURVES["gddr7"] = [(x, v * (0.79 * 864 / 1792 + 0.21 * 0.9)) for x, v in CURVES["bc"]]
EST_GPU = {"a100", "ada5k", "gddr7"}

# ---- GPU silicon: merge poly-mul per-mul us (batch=16 basis; protocol runs
# batched muls). Sources: *_ntt_batch.txt batteries.
NTT16 = {
    "bc":   {14: 2.00, 16: 6.18, 18: 27.1, 20: 266.4, 22: 1197.6, 24: 4952.6},
    "h200h": {14: 2.54, 16: 8.06, 18: 33.7, 20: 141.4, 22: 605.7, 24: 2633.6},
    "b200": {14: 2.49, 16: 7.51, 18: 31.0, 20: 129.4, 22: 549.6, 24: 2449.8},
}
NTT16["a100"] = {k: math.sqrt(NTT16["bc"][k] * NTT16["h200h"][k]) for k in NTT16["bc"]}
NTT16["ada5k"] = {k: v * 1.35 for k, v in NTT16["bc"].items()}
NTT16["gddr7"] = {k: v * 0.70 for k, v in NTT16["bc"].items()}

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


def cycles(path):
    try:
        for line in open(path):
            m = re.search(r"memory_system_cycles:\s*(\d+)", line)
            if m:
                return int(m.group(1))
    except OSError:
        pass
    return None


def main():
    scratch = sys.argv[1] if len(sys.argv) > 1 else "."
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ns = {}          # (dev, point) -> card-level ns/leaf
    src = {}         # (dev, point) -> "n12" | "n10" (fallback provenance)
    missing = []
    for dev, (label, tck, C, PU, card) in DEV.items():
        leaves12 = C * 8 * 4 * 4096
        leaves10 = C * 8 * 4 * 1024
        leaves14 = C * 8 * 2 * 16384
        for point in ("cl272_none", "cl272_mau", "cl272_fused", "cl155_none",
                      "cl155_mau", "cl155_fused", "cl0_none", "cl272_mau_sm",
                      "cl155_mau_sm", "cl272_mau_ntt1", "cl155_mau_ntt1",
                      "n14_cl155_mau", "cl155_mau_ntt05", "cl155_mau_ntt2"):
            # v2 battery: full-load n=12 4-round MEASURED points (fixed
            # simulator: MAU decoupled from bank row management, b942075).
            # These supersede the r1-slice ratio fill entirely.
            v2 = ("v2_%s_cl0_cl0.out" % dev if point == "cl0_none"
                  else "v2_%s_%s.out" % (dev, point))
            c = cycles(os.path.join(scratch, v2))
            lv = leaves14 if point.startswith("n14") else leaves12
            tag = "v2"
            if c is None:
                c = cycles(os.path.join(scratch, f"{dev}_{point}.out"))
                tag = "n12"
            if c is None:
                # n=10 backup battery (mau points; n-insensitivity 0.5-0.7%)
                c = cycles(os.path.join(scratch, f"{dev}_n10_{point}.out"))
                lv, tag = leaves10, "n10"
            if c is None:
                if not point.startswith("cl155_mau_ntt0") and \
                   not point.startswith("cl155_mau_ntt2"):
                    missing.append(f"{dev}_{point}")
                continue
            ns[(dev, point)] = c * tck / lv / card
            src[(dev, point)] = tag
        # RATIO fill for mau-family points: the full-load n=12 4-round sims hit
        # the simulator's quadratic wall-clock blowup (wall ~ O(total_requests^2)
        # from per-cycle scans; measured 1-round 65 s vs 2-round >540 s), so
        # the mau/none RATIO is measured on a cheap n=10 2-round channel-SLICE
        # (<dev>_s10_* points; channel linearity validated +-4%, the ratio is
        # channel-local and slice-invariant) and applied to the MEASURED n=12
        # none value. Exposed-tail amortization differs (1/2 vs 1/4 rounds), so
        # the slice ratio is CONSERVATIVE (overstates the mau cost by ~half the
        # tail term). Flagged 'ratio'.
        # ratio basis: prefer the 1-round slice (r1_*, tail fully exposed =
        # conservative), fall back to the 2-round slice (s10_*) none where the
        # 1-round none was not run. The mau/none RATIO is channel-local and
        # amortization-conservative; applied to the MEASURED n=12 none value.
        for cl in ("cl272", "cl155"):
            base = ns.get((dev, f"{cl}_none"))
            if base is None:
                continue
            s_none = (cycles(os.path.join(scratch, f"{dev}_r1_{cl}_none.out"))
                      or cycles(os.path.join(scratch, f"{dev}_s10_{cl}_none.out")))
            for m in ("mau", "mau_sm", "mau_ntt1"):
                point = f"{cl}_{m}"
                if (dev, point) in ns:
                    continue
                s_m = cycles(os.path.join(scratch, f"{dev}_r1_{cl}_{m}.out"))
                base_none = s_none
                if s_m is None:
                    s_m = cycles(os.path.join(scratch, f"{dev}_s10_{cl}_{m}.out"))
                    base_none = cycles(os.path.join(scratch, f"{dev}_s10_{cl}_none.out"))
                if s_m is None or not base_none:
                    continue
                ns[(dev, point)] = base * (s_m / base_none)
                src[(dev, point)] = "ratio"
                if f"{dev}_{point}" in missing:
                    missing.remove(f"{dev}_{point}")

    out1 = os.path.join(repo, "pim", "results", "final_campaign_pim.txt")
    with open(out1, "w") as f:
        f.write("== FINAL CAMPAIGN, PIM sim tier (Ramulator2, all-bank broadcast,\n"
                "== gated, 128b seeds 8r8w, full load 4 rounds, n=12; card-level\n"
                "== ns/leaf = sim/card_factor). Ablation cases:\n"
                "==   A: MAU = DPF lc + NTT stage-1 (_ntt1 = contention-loaded)\n"
                "==   B: MAU = DPF lc only          (_sm  = SM traffic loaded)\n"
                "==   C: no MAU, fused modmul on PU\n==\n")
        hdr = ("dev        PU(card) | floor  | 1ALU: none  mau   fused | "
               "2ALU: none  mau   fused | mau_sm(2A) ntt1(2A) | n14x | 2ALUx  mau/fused\n")
        f.write(hdr)
        for dev, (label, tck, C, PU, card) in DEV.items():
            g = lambda p: ns.get((dev, p))
            def s(p, w=6):
                v = g(p)
                return f"{v:.4f}" if v is not None else "  -   "
            gain = (g("cl272_mau") / g("cl155_mau")
                    if g("cl272_mau") and g("cl155_mau") else None)
            mvf = (g("cl155_fused") / g("cl155_mau")
                   if g("cl155_fused") and g("cl155_mau") else None)
            n14 = (ns.get((dev, "n14_cl155_mau"), 0) /
                   g("cl155_mau") - 1 if g("cl155_mau") and
                   (dev, "n14_cl155_mau") in ns else None)
            f.write(f"{label:<10} {PU*card:>4}   | {s('cl0_none')} | "
                    f"{s('cl272_none')} {s('cl272_mau')} {s('cl272_fused')} | "
                    f"{s('cl155_none')} {s('cl155_mau')} {s('cl155_fused')} | "
                    f"{s('cl155_mau_sm')}  {s('cl155_mau_ntt1')} | "
                    f"{('%+.1f%%' % (n14*100)) if n14 is not None else '  -  '} | "
                    f"{('%.2fx' % gain) if gain else ' -  '}  "
                    f"{('%.2fx' % mvf) if mvf else ' -  '}\n")
        v2n = sum(1 for t in src.values() if t == "v2")
        f.write(f"\n== v2 full-load measured points (n=12, 4 rounds, fixed sim): {v2n}\n")
        n10pts = sorted(f"{d}_{p}" for (d, p), t in src.items() if t == "n10")
        if n10pts:
            f.write("\n== n=10 backup battery used for (n-insensitivity 0.5-0.7%): "
                    + " ".join(n10pts) + "\n")
        ana = sorted(f"{d}_{p}" for (d, p), t in src.items() if t == "analytic")
        if ana:
            f.write("\n== ANALYTIC fallback (none x 1.05 hidden-stream model, "
                    "HBM-calibrated epsilon; sims pending): " + " ".join(ana) + "\n")
        if missing:
            f.write("\n== MISSING points (rerun needed): " + " ".join(missing) + "\n")
        ntt05, ntt2 = ns.get(("bc", "cl155_mau_ntt05")), ns.get(("bc", "cl155_mau_ntt2"))
        if ntt05 and ntt2:
            f.write(f"\n== mau-ntt volume sensitivity (L40S, 2ALU): 0.5x={ntt05:.4f} "
                    f"1x={ns.get(('bc','cl155_mau_ntt1'),0):.4f} 2x={ntt2:.4f}\n")

    out2 = os.path.join(repo, "pim", "results", "final_campaign_speedup.txt")
    with open(out2, "w") as f:
        f.write("== FINAL CAMPAIGN speedup vs all-GPU, per (N,t) pair, overlapped\n"
                "== steady state T = max(T_dpf_pim, T_ntt_sm). GPU serial =\n"
                "== leaves x regime(leaves) + 2 x polymul16(N). PIM lane uses the\n"
                "== CONTENTION-LOADED point for each case (A: _ntt1, B: _sm,\n"
                "== C: fused + (sm-mau) delta approx). 2ALU (cl=155) throughout.\n"
                "== est = GPU baseline estimated (no silicon battery yet).\n"
                "== W = capacity-wall cell (block-streaming schedule; measured\n"
                "== faster than flat on HBM -> conservative).\n==\n")
        for dev, (label, tck, C, PU, card) in DEV.items():
            base_mau = ns.get((dev, "cl155_mau"))
            dpf_A = ns.get((dev, "cl155_mau_ntt1"))
            dpf_B = ns.get((dev, "cl155_mau_sm"))
            dpf_C0 = ns.get((dev, "cl155_fused"))
            if not all((base_mau, dpf_A, dpf_B, dpf_C0)):
                f.write(f"-- {label}: incomplete sim points, skipped\n\n")
                continue
            dpf_C = dpf_C0 + (dpf_B - base_mau)          # borrow SM delta (approx)
            est = " est" if dev in EST_GPU else ""
            f.write(f"-- {label} ({PU*card} PU, 2ALU){est}: DPF ns/leaf "
                    f"A={dpf_A:.4f} B={dpf_B:.4f} C={dpf_C:.4f}\n")
            f.write(f"   {'N':>6} {'t':>3} | {'GPU ms':>9} | "
                    f"{'A x':>6} {'B x':>6} {'C x':>6} | lane(B) flags\n")
            for lgN in (16, 18, 20, 22, 24):
                N = 1 << lgN
                for t in (4, 8, 16, 32):
                    leaves = 2 * N * t
                    gdpf = leaves * interp(CURVES[dev], leaves) / 1e6
                    gntt = 2 * NTT16[dev][lgN] / 1e3
                    gpu = gdpf + gntt
                    n1 = min(1 << 14, N >> 6)
                    f1 = math.log2(n1) / lgN if n1 > 1 else 0
                    tA = max(leaves * dpf_A / 1e6, gntt * (1 - f1))
                    tB = max(leaves * dpf_B / 1e6, gntt)
                    tC = max(leaves * dpf_C / 1e6, gntt)
                    lane = "DPF" if leaves * dpf_B / 1e6 >= gntt else "NTT"
                    wall = "W" if 2 * (2 * N // t) * 16 >= BANK_BYTES else ""
                    f.write(f"   {'2^%d' % lgN:>6} {t:>3} | {gpu:9.2f} | "
                            f"{gpu/tA:5.1f}x {gpu/tB:5.1f}x {gpu/tC:5.1f}x | "
                            f"{lane:>4} {wall}\n")
            f.write("\n")
    print(f"wrote {out1}\nwrote {out2}")
    if missing:
        print("MISSING:", " ".join(missing))


if __name__ == "__main__":
    main()
