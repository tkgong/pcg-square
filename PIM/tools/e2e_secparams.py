#!/usr/bin/env python3
"""E2E PCG at the security-calibrated sizes (BCG+20 Table 1), double ablation.

Per security row (lambda, N, c, t), per device:
  GPU baselines (L40S measured; other devices est-scaled):
    (a) naive-NTT baseline   = newDPF(leaves) + 2c^2 x naive_ms(N, b=c^2)
    (b) 4step-NTT baseline   = newDPF(leaves) + 2c^2 x fourstep_ms(N, b=c^2)
  PIM e2e (overlapped steady state, net rounds hidden per convention):
    max( leaves x pim_ns(dev, +-LSU) , 2c^2 x [4step - DRU?xtranspose](N,b) )
  Ablation 1: DPF +-LSU  (sim ggm_compute_overlap true/false, measured ratio)
  Ablation 2: 4step +-DRU (transposes offloaded to shadow bandwidth; HBM only
              -- GDDR SMs are BW-bound, no shadow -> DRU gain recorded 0)

leaves_total = c^2 * 2Nt. NTT size = N, batch = c^2.
"""
import math, os, re, sys

SEC = [(80, 2, 64), (80, 4, 16), (80, 8, 4),
       (128, 2, 128), (128, 4, 16), (128, 8, 8)]
LGNS = [20, 21, 22, 23, 24]
BATCH = {2: 4, 4: 16, 8: 64}

# ---- L40S silicon: naive NTT ms/mul (naive_ntt_bench, BITEXACT vs merge) ----
NAIVE = {  # [lgN][batch]
    20: {4: 0.3769, 16: 1.6231, 64: 1.5927},
    21: {4: 2.5275, 16: 3.3760, 64: 3.3262},
    22: {4: 7.2411, 16: 7.0807, 64: 7.0472},
    23: {4: 15.7192, 16: 15.6092, 64: 15.5993},
    24: {4: 33.8085, 16: 33.5150, 64: 33.4347},
}
# ---- L40S silicon: 4step ms/mul + transpose stage per-mul (nttsec battery) --
FOURSTEP_BC = {
    20: {4: (0.2404, 0.069), 16: (0.4980, 0.163), 64: (0.4927, 0.158)},
    21: {4: (0.7338, 0.223), 16: (0.9982, 0.325), 64: (1.0056, 0.321)},
    22: {4: (2.2476, 0.699), 16: (2.2254, 0.702), 64: (2.2184, 0.703)},
    23: {4: (4.8517, 1.474), 16: (4.8250, 1.487), 64: (4.8231, 1.483)},
    24: {4: (9.7974, 3.066), 16: (9.7661, 3.061), 64: (9.7577, 3.049)},
}
# ---- B200 silicon: 4step ms/mul (crossover battery, tkgong) + transp% ------
FOURSTEP_B200 = {
    20: {4: (0.1781, .236), 16: (0.1730, .245), 64: (0.1685, .243)},
    21: {4: (0.3623, .237), 16: (0.3500, .239), 64: (0.3453, .238)},
    22: {4: (0.7411, .232), 16: (0.7225, .231), 64: (0.7185, .230)},
    23: {4: (1.5030, .225), 16: (1.4851, .223), 64: (1.4815, .223)},
    24: {4: (3.0707, .218), 16: (3.0532, .217), 64: (3.0479, .217)},
}

# ---- GPU new-algorithm DPF+convert curve (L40S measured, ns/leaf) ----------
GPU_BC = [(0.52e6, .720), (1.05e6, .470), (4.2e6, .247), (16.8e6, .216),
          (67.1e6, .216), (268e6, .218), (1.07e9, .220)]

# dev -> (label, mem gen, tck_ns, channels, card, ntt_ratio_vs_bc, hbm?, est?)
DEV = {
    "bc":    ("L40S",       "GDDR6",  0.444,  24, 1, 1.00, False, False),
    "ada5k": ("RTX5000Ada", "GDDR6",  0.444,  16, 1, 1.35, False, True),
    "gddr7": ("RTXPRO6000", "GDDR7",  0.571,  64, 1, 0.70, False, True),
    "a100":  ("A100-80",    "HBM2e",  0.625,  80, 1, 0.73, True,  True),
    "h200h": ("H200",       "HBM3",   0.625,  96, 2, 0.53, True,  True),
    "b200":  ("B200",       "HBM3e",  0.500, 128, 2, None, True,  True),
}
LEAVES_PER_CH = 8 * 4 * 4096
GPU_DEV_FACTOR = {"bc": 1.0, "h200h": 0.55, "b200": 0.49,
                  "a100": 0.74, "ada5k": 1.30, "gddr7": 0.79 * 864/1792 + 0.21*0.9}


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


def fourstep(dev, lg, b, dru):
    """(N,batch) 4step ms/mul on device; dru=True removes the transpose stage
    (hidden in shadow bandwidth) on HBM devices only."""
    hbm = DEV[dev][6]
    if dev == "b200":
        ms, tp = FOURSTEP_B200[lg][b]
        tms = ms * tp
    else:
        ms, tms = FOURSTEP_BC[lg][b]
        r = DEV[dev][5]
        ms, tms = ms * r, tms * r
    if dru and hbm:
        return ms - tms
    return ms


def read_cpu(path):
    """bench_result,tag,N,c,t,w,prg,backend,dpf,warmup,iters,mean_ms,std_ms,ok"""
    cpu = {}
    try:
        for line in open(path):
            if not line.startswith("bench_result,"):
                continue
            p = line.strip().split(",")
            if len(p) < 14 or p[13] != "1":
                continue
            cpu[(int(p[2]), int(p[3]), int(p[4]))] = float(p[11])
    except OSError:
        pass
    return cpu


def main():
    S = sys.argv[1] if len(sys.argv) > 1 else "."
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    bc_none = cyc(os.path.join(S, "v4_bc_n12_none.out"))
    if bc_none is None:
        sys.exit("missing anchor v4_bc_n12_none.out -- refusing silent fallback")
    ch_lsu = cyc(os.path.join(S, "v5_ch465.out"))
    ch_nolsu = cyc(os.path.join(S, "v6_noLSU.out"))
    ch_xpose = cyc(os.path.join(S, "v6_xpose.out"))
    if not all((ch_lsu, ch_nolsu, ch_xpose)):
        sys.exit(f"missing sims: lsu={ch_lsu} nolsu={ch_nolsu} xpose={ch_xpose}")
    # single-issue (one fused XOR/ADD ALU) points: cl=282 EXTEND + cl=846 CONVERT
    si_lsu = cyc(os.path.join(S, "v7_single.out"))
    si_nolsu = cyc(os.path.join(S, "v7_single_noLSU.out"))
    # node-major (naive) layout, same hardware: per-word strided access ->
    # 4x rd/wr columns per op (opsize/wrsize x4), cl unchanged
    nm_str = cyc(os.path.join(S, "v9_nm_strided.out"))
    cpu_ms = read_cpu(os.path.join(S, "cpu_secparams.csv"))
    r_lsu = ch_lsu / bc_none
    r_nolsu = ch_nolsu / bc_none
    f_dru = ch_xpose / ch_lsu - 1          # NTT-phase stream contention on PIM

    pim_ns = {}
    for dev, (lab, gen, tck, C, card, _, _, _) in DEV.items():
        none_c = cyc(os.path.join(S, f"v2_{dev}_cl155_none.out"))
        if none_c is None:
            continue
        base = none_c * tck / (C * LEAVES_PER_CH * card)
        pim_ns[dev] = {"lsu": base * r_lsu, "nolsu": base * r_nolsu}

    out = os.path.join(repo, "pim", "results", "e2e_secparams.txt")
    with open(out, "w") as f:
        f.write("== E2E PCG at SECURITY-CALIBRATED sizes (BCG+20 Table 1)\n"
                "== Algorithm: Beaver output layer (NO modmul in DPF+convert).\n"
                "== PIM: fused ChaCha CONVERT (cl465) on near-bank ARX+modALU;\n"
                "==      NTT on SM via 4-STEP (naive/4step world; merge unused).\n"
                "== GPU baselines (L40S silicon; others est-scaled): newDPF curve\n"
                "==   + (a) naive NTT   (b) 4step NTT,  2c^2 muls, batch=c^2.\n"
                "== PIM e2e = max(DPF lane, NTT lane); net rounds hidden (conv.).\n"
                f"== MEASURED: LSU ratio {r_lsu:.3f} vs noLSU {r_nolsu:.3f} "
                f"(LSU gain {r_nolsu/r_lsu:.3f}x); NTT-phase stream contention "
                f"f_dru = {f_dru*100:.2f}%\n"
                "== DRU rule: HBM devices hide the 4step transpose stage in idle\n"
                "== bandwidth (compute-bound butterflies); GDDR has no shadow ->\n"
                "== DRU gain = 0 there (honest lower bound).\n==\n")

        # ---- Table 1: main grid ----
        f.write("---- 1. main grid: ms per expansion + speedups ----\n")
        for dev, (lab, gen, tck, C, card, _, hbm, est) in DEV.items():
            if dev not in pim_ns:
                continue
            f.write(f"-- {lab} [{gen}]{' est' if est else ''}: PIM DPF "
                    f"{pim_ns[dev]['lsu']:.4f} ns/leaf (noLSU "
                    f"{pim_ns[dev]['nolsu']:.4f})\n")
            f.write(f"   {'lam':>4} {'N':>5} {'c':>2} {'t':>4} |"
                    f" {'GPUnaive':>9} {'GPU4step':>9} |"
                    f" {'PIM-L-D':>8} {'PIM+L-D':>8} {'PIM+L+D':>8} |"
                    f" {'x_naive':>7} {'x_4step':>7} | lane\n")
            for lam, c, t in SEC:
                b = BATCH[c]
                for lg in LGNS:
                    N = 1 << lg
                    leaves = c * c * 2 * N * t
                    gf = GPU_DEV_FACTOR[dev]
                    gdpf = leaves * interp(GPU_BC, leaves) * gf / 1e6
                    g_naive = gdpf + 2*c*c * NAIVE[lg][b] * (DEV[dev][5] or 0.49) / 1.0
                    g_4step = gdpf + 2*c*c * fourstep(dev, lg, b, dru=False)
                    dpf_l = leaves * pim_ns[dev]["lsu"] / 1e6
                    dpf_n = leaves * pim_ns[dev]["nolsu"] / 1e6
                    ntt_nd = 2*c*c * fourstep(dev, lg, b, dru=False)
                    ntt_d = 2*c*c * fourstep(dev, lg, b, dru=True)
                    pim_nl_nd = max(dpf_n, ntt_nd)
                    pim_l_nd = max(dpf_l, ntt_nd)
                    dpf_l_dru = dpf_l * (1 + (f_dru if hbm else 0.0))
                    pim_l_d = max(dpf_l_dru, ntt_d)
                    lane = "DPF" if dpf_l >= ntt_d else "NTT"
                    f.write(f"   {lam:>4} 2^{lg:<3} {c:>2} {t:>4} |"
                            f" {g_naive:9.2f} {g_4step:9.2f} |"
                            f" {pim_nl_nd:8.2f} {pim_l_nd:8.2f} {pim_l_d:8.2f} |"
                            f" {g_naive/pim_l_d:6.1f}x {g_4step/pim_l_d:6.1f}x |"
                            f" {lane}\n")
            f.write("\n")

        # ---- Table 2: LSU ablation ----
        f.write("---- 2. Ablation 1: DPF +-LSU (PIM DPF lane slowdown w/o LSU) ----\n")
        f.write(f"   measured on L40S sim: chacha465 noLSU/LSU = "
                f"{r_nolsu/r_lsu:.3f}x (I/O no longer hidden under compute).\n"
                f"   Device-independent (same op structure); applies to the DPF\n"
                f"   lane everywhere; flips e2e only where DPF-bound.\n\n")

        # ---- Table 3: DRU ablation (headline row) ----
        f.write("---- 3. Ablation 2: 4step NTT +-DRU, headline lam=128 c=4 t=16 ----\n")
        f.write(f"   {'dev':>11} {'gen':>6} |" +
                "".join(f"  2^{lg:<4}" for lg in LGNS) + "  (e2e noDRU/DRU)\n")
        for dev, (lab, gen, *_rest) in DEV.items():
            if dev not in pim_ns:
                continue
            hbm = DEV[dev][6]
            row = []
            for lg in LGNS:
                b = 16; c = 4; t = 16
                N = 1 << lg
                leaves = c*c*2*N*t
                dpf_l = leaves * pim_ns[dev]["lsu"] / 1e6
                e_nd = max(dpf_l, 2*c*c*fourstep(dev, lg, b, False))
                e_d = max(dpf_l*(1+(f_dru if hbm else 0)), 2*c*c*fourstep(dev, lg, b, True))
                row.append(e_nd / e_d)
            f.write(f"   {lab:>11} {gen:>6} |" +
                    "".join(f" {x:6.2f}x" for x in row) + "\n")
        f.write("\n")

        # ---- Table 4: memory-generation scaling, headline ----
        f.write("---- 4. memory generations, headline row (x vs GPU-4step, +L+D) ----\n")
        f.write(f"   {'gen':>6} {'dev':>11} {'PIM ns/leaf':>11} |" +
                "".join(f"  2^{lg:<4}" for lg in LGNS) + "\n")
        for dev in ("bc", "ada5k", "gddr7", "a100", "h200h", "b200"):
            if dev not in pim_ns:
                continue
            lab, gen = DEV[dev][0], DEV[dev][1]
            row = []
            for lg in LGNS:
                c, t, b = 4, 16, 16
                N = 1 << lg
                leaves = c*c*2*N*t
                gf = GPU_DEV_FACTOR[dev]
                g4 = leaves*interp(GPU_BC, leaves)*gf/1e6 + 2*c*c*fourstep(dev, lg, b, False)
                hbm = DEV[dev][6]
                pim = max(leaves*pim_ns[dev]["lsu"]*(1+(f_dru if hbm else 0))/1e6,
                          2*c*c*fourstep(dev, lg, b, True))
                row.append(g4/pim)
            f.write(f"   {gen:>6} {lab:>11} {pim_ns[dev]['lsu']:11.4f} |" +
                    "".join(f" {x:6.1f}x" for x in row) + "\n")

    # ================= L40S-only three-baseline report =================
    out2 = os.path.join(repo, "pim", "results", "l40s_cpu_gpu_pim.txt")
    dev = "bc"
    lab, gen, tck, C, card, _, hbm, _ = DEV[dev]
    base = pim_ns[dev]["lsu"] / r_lsu          # none-floor ns/leaf on L40S
    with open(out2, "w") as f:
        f.write("== L40S THREE-BASELINE report (ALL MEASURED on this host/GPU)\n"
                "== CPU  : bench_pcg_ole_2pc 2PC wall (Xeon Gold 6442Y, localhost,\n"
                "==        chacha8, verify=PASS gated), gen_and_expand total.\n"
                "== GPU  : L40S silicon, newDPF curve + naive/4step NTT.\n"
                "== PIM+GPU: Ramulator2 GDDR6_L40S 192-PU DPF lane (dual-issue,\n"
                "==        +LSU) overlapped with SM 4step NTT lane; GDDR DRU\n"
                "==        rule -> e2e DRU gain 0 (no shadow bandwidth).\n==\n")

        # ---- A. main table ----
        f.write("---- A. 30 security rows: ms per expansion + speedups ----\n")
        f.write(f"   {'lam':>4} {'N':>5} {'c':>2} {'t':>4} |"
                f" {'CPU':>10} {'GPUnaive':>9} {'GPU4step':>9} |"
                f" {'PIM-L':>8} {'PIM+L':>8} |"
                f" {'x_cpu':>7} {'x_naive':>7} {'x_4step':>7} | lane\n")
        for lam, c, t in SEC:
            b = BATCH[c]
            for lg in LGNS:
                N = 1 << lg
                leaves = c * c * 2 * N * t
                gdpf = leaves * interp(GPU_BC, leaves) / 1e6
                g_naive = gdpf + 2*c*c * NAIVE[lg][b]
                g_4step = gdpf + 2*c*c * fourstep(dev, lg, b, dru=False)
                dpf_l = leaves * pim_ns[dev]["lsu"] / 1e6
                dpf_n = leaves * pim_ns[dev]["nolsu"] / 1e6
                ntt = 2*c*c * fourstep(dev, lg, b, dru=False)   # GDDR: DRU=0
                pim_l = max(dpf_l, ntt)
                pim_n = max(dpf_n, ntt)
                lane = "DPF" if dpf_l >= ntt else "NTT"
                cm = cpu_ms.get((N, c, t))
                cs = f"{cm:10.0f}" if cm else f"{'pending':>10}"
                xs = f"{cm/pim_l:6.0f}x" if cm else f"{'--':>7}"
                f.write(f"   {lam:>4} 2^{lg:<3} {c:>2} {t:>4} |"
                        f" {cs} {g_naive:9.2f} {g_4step:9.2f} |"
                        f" {pim_n:8.2f} {pim_l:8.2f} |"
                        f" {xs} {g_naive/pim_l:6.1f}x {g_4step/pim_l:6.1f}x |"
                        f" {lane}\n")
        f.write("\n")

        # ---- B. DPF ablation: issue width x LSU (L40S sim, measured) ----
        f.write("---- B. DPF ablation: dual/single issue x +-LSU (ns/leaf, L40S) ----\n")
        f.write("   dual  = XADD + VXORL interleaved QR chains (cl 155/465)\n"
                "   single= one fused XOR/ADD ALU            (cl 282/846)\n")
        cells = {("dual", "+LSU"): ch_lsu, ("dual", "-LSU"): ch_nolsu,
                 ("single", "+LSU"): si_lsu, ("single", "-LSU"): si_nolsu}
        f.write(f"   {'issue':>7} | {'+LSU':>18} | {'-LSU':>18} | LSU gain\n")
        for iss in ("dual", "single"):
            row = []
            for l in ("+LSU", "-LSU"):
                cyc_v = cells[(iss, l)]
                if cyc_v is None:
                    row.append(f"{'pending':>18}")
                else:
                    ns = base * cyc_v / bc_none
                    rel = cyc_v / ch_lsu
                    row.append(f"{ns:8.4f} ({rel:5.2f}x)")
            gain = (f"{cells[(iss,'-LSU')]/cells[(iss,'+LSU')]:5.3f}x"
                    if cells[(iss, "+LSU")] and cells[(iss, "-LSU")] else "  --")
            f.write(f"   {iss:>7} | {row[0]} | {row[1]} | {gain}\n")
        f.write("   (x) = slowdown vs dual+LSU. DPF-bound e2e rows scale by the\n"
                "   same factor; NTT-bound rows are insensitive until the DPF\n"
                "   lane overtakes the NTT lane.\n\n")

        # ---- B2. data-layout ablation: node-major baseline -> word-major ----
        f.write("---- B2. Layout ablation: node-major (baseline) -> word-major ----\n")
        f.write("   BASELINE  node-major: one col holds complete words of a\n"
                "     node. Each vector register needs 8 words scattered\n"
                "     across 4 cols' 32b sub-slots; the column is the minimum\n"
                "     access unit and the fixed PACKLR cannot compose an 8x4\n"
                "     transpose -> per-word strided access, 4x rd+wr column\n"
                "     traffic (cl unchanged). Measured: the extra traffic\n"
                "     tips the loop past the 155-CK compute bound -- memory\n"
                "     becomes the critical path (the bank-pair read/write\n"
                "     split absorbs part of the 4x; the rest is exposed).\n"
                "   DESIGN    word-major: storage view == register view\n"
                "     (col j = word j of 8 seeds); VLD IS the transpose,\n"
                "     0 permute ops, same hardware.\n")
        ns_wm = base * ch_lsu / bc_none
        f.write(f"   {'config':>24} | {'ns/leaf':>9} | {'vs design':>9} | col traffic\n")
        if nm_str:
            ns_nm = base * nm_str / bc_none
            f.write(f"   {'node-major (baseline)':>24} | {ns_nm:9.4f} |"
                    f" {nm_str/ch_lsu:8.2f}x | 4x (measured)\n")
        else:
            f.write(f"   {'node-major (baseline)':>24} | {'pending':>9} |"
                    f" {'--':>9} | 4x\n")
        f.write(f"   {'word-major (design)':>24} | {ns_wm:9.4f} |"
                f" {'1.00x':>9} | 1x\n")
        f.write("   WHY: the SIMD lane axis IS the seed axis (QR ops are\n"
                "   word-vs-word), so the register view is fixed by the\n"
                "   compute; word-major makes the storage view identical --\n"
                "   the layout, not the datapath, performs the transpose.\n"
                "   Producer==consumer (children are emitted register-wise),\n"
                "   so the convention sustains itself across all levels for\n"
                "   free; only the final leaf->g scatter needs node order,\n"
                "   and that is AGU address arithmetic, not data movement.\n\n")

        # ---- C. NTT +-DRU on L40S ----
        f.write("---- C. NTT 4step +-DRU on L40S (per-mul, batch=c^2) ----\n")
        f.write(f"   {'N':>5} {'b':>3} | {'4step ms':>9} {'transp ms':>9} "
                f"{'transp%':>8} | {'NTT-lane ub':>11} | e2e gain\n")
        for lg in LGNS:
            for b in (4, 16, 64):
                ms, tms = FOURSTEP_BC[lg][b]
                f.write(f"   2^{lg:<3} {b:>3} | {ms:9.4f} {tms:9.4f} "
                        f"{100*tms/ms:7.1f}% | {ms/(ms-tms):10.2f}x | 1.00x\n")
        f.write("   NTT-lane ub = hypothetical NTT-lane speedup IF shadow\n"
                "   bandwidth existed (transpose fully hidden). e2e gain on\n"
                "   L40S/GDDR = 1.00x by the honest rule: SMs are BW-bound on\n"
                "   GDDR, there is no idle bandwidth for the DRU to borrow.\n\n")

        # ---- D. FULL ablation grid: PIM vs GPU speedup, every config ----
        if si_lsu and si_nolsu:
            ns_cfg = [  # (label, ns/leaf, dru)
                ("dual +LSU +DRU",   base * ch_lsu   / bc_none, True),
                ("dual +LSU -DRU",   base * ch_lsu   / bc_none, False),
                ("dual -LSU -DRU",   base * ch_nolsu / bc_none, False),
                ("single +LSU -DRU", base * si_lsu   / bc_none, False),
                ("single -LSU -DRU", base * si_nolsu / bc_none, False),
            ]
            f.write("---- D. FULL ablation grid: PIM speedup vs GPU-4step ----\n")
            f.write("   configs: [1] dual+LSU+DRU (flagship)  [2] dual+LSU-DRU\n"
                    "            [3] dual-LSU-DRU  [4] single+LSU-DRU\n"
                    "            [5] single-LSU-DRU\n"
                    "   On L40S/GDDR [1]==[2] (DRU e2e gain 0, no shadow BW;\n"
                    "   DRU differentiates only on HBM -- see e2e_secparams #3).\n")
            f.write(f"   {'lam':>4} {'N':>5} {'c':>2} {'t':>4} |"
                    f" {'GPU4step':>9} |"
                    + "".join(f" {'['+str(i+1)+']':>7}" for i in range(5))
                    + " | lane\n")
            for lam, c, t in SEC:
                b = BATCH[c]
                for lg in LGNS:
                    N = 1 << lg
                    leaves = c * c * 2 * N * t
                    gdpf = leaves * interp(GPU_BC, leaves) / 1e6
                    g_4step = gdpf + 2*c*c * fourstep(dev, lg, b, dru=False)
                    xs = []
                    for _lab, ns, dru in ns_cfg:
                        dl = leaves * ns / 1e6
                        nl = 2*c*c * fourstep(dev, lg, b, dru=dru)
                        xs.append(g_4step / max(dl, nl))
                    dl_f = leaves * ns_cfg[0][1] / 1e6
                    lane = "DPF" if dl_f >= 2*c*c*fourstep(dev, lg, b, True) else "NTT"
                    f.write(f"   {lam:>4} 2^{lg:<3} {c:>2} {t:>4} |"
                            f" {g_4step:9.2f} |"
                            + "".join(f" {x:6.2f}x" for x in xs)
                            + f" | {lane}\n")

    print(f"wrote {out}")
    print(f"wrote {out2}")
    print(f"r_lsu={r_lsu:.3f} r_nolsu={r_nolsu:.3f} f_dru={f_dru*100:.2f}%")
    if si_lsu:
        print(f"single-issue: lsu={si_lsu} ({si_lsu/ch_lsu:.3f}x of dual), "
              f"nolsu={si_nolsu}")
    print(f"cpu rows: {len(cpu_ms)}/25 unique (N,c,t)")


if __name__ == "__main__":
    main()
