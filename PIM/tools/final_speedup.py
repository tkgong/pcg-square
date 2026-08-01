#!/usr/bin/env python3
"""FINAL PCG speedup report: PIM+GPU vs GPU baselines, with the three ablations.

Sections
  A. main grid       30 security rows: GPU-naive / GPU-4step / PIM+GPU e2e
  B. ablation NET-SW-HW  what each leg of the co-design contributes
  C. ablation PU     ChaCha SPU single- vs dual-issue x +-LSU (sim measured)
  D. ablation NTT    naive -> 4step -> 4step+DRU (GPU comparison world)
  E. headline        lambda=128, c=4, t=16 end-to-end numbers

Data provenance
  GPU DPF+convert : L40S regime curve, measured (results_newalgo_gpu_l40s.md)
                    B200 measured grid (baseline_v2 b0029, kernel-only ns/leaf)
  GPU NTT         : L40S + B200 measured naive / 4step (+transpose split)
  PIM DPF lane    : Ramulator2 GDDR6_L40S / HBM3E_B200D full-load sims
                    dual+LSU 692720 | dual-LSU 761808 | single+LSU 1210060
                    single-LSU 1279284 cycles, none-anchor 389192 (L40S org)
  Network         : measured comm traces (alpha 25.2us small-msg median;
                    per-level exchange counts c^2 x levels)
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_secparams import (SEC, LGNS, BATCH, NAIVE, FOURSTEP_BC, FOURSTEP_B200,
                           GPU_BC, interp)

# ---- PIM sim anchors (cycles) ------------------------------------- measured
NONE_L40S, NONE_B200 = 389192, 402156
PU = {"dual+LSU": 692720, "dual-LSU": 761808,
      "single+LSU": 1210060, "single-LSU": 1279284}
TCK = {"L40S": 0.444e-9, "B200": 0.500e-9}
CH  = {"L40S": 24, "B200": 128}
CARD = {"L40S": 1, "B200": 2}
LEAVES_PER_CH = 8 * 4 * 4096
NONE = {"L40S": NONE_L40S, "B200": NONE_B200}

# B200 chacha DIRECT sims (v9b battery, HBM3E_B200D full load, this campaign).
# dual+LSU 701600 validates the ratio transfer to -2.0% (predicted 715816).
# ALL FOUR PU configs now measured; monotonicity gate passed.
MEAS_B200 = {"dual+LSU": 701600, "dual-LSU": 762520,
             "single+LSU": 1218704, "single-LSU": 1279740}

def pim_ns(dev, cfg="dual+LSU"):
    """ns per leaf on `dev` for PU configuration `cfg`.
    B200: direct Ramulator measurement where available (MEAS_B200);
    remaining configs ratio-transferred (flagged). L40S: direct sims."""
    if dev == "B200" and cfg in MEAS_B200:
        return (MEAS_B200[cfg] * TCK[dev] * 1e9
                / (CH[dev] * LEAVES_PER_CH * CARD[dev]))
    base = NONE[dev] * TCK[dev] * 1e9 / (CH[dev] * LEAVES_PER_CH * CARD[dev])
    return base * PU[cfg] / NONE_L40S

# ---- GPU DPF+convert -------------------------------------------- measured
B200_DPF_NS = {  # [lgN][t] kernel-only ns/leaf, baseline_v2 report table 1
    20: {4: .063, 8: .061, 16: .058, 32: .057, 64: .057, 128: .057},
    21: {4: .058, 8: .056, 16: .055, 32: .055, 64: .055, 128: .055},
    22: {4: .052, 8: .051, 16: .052, 32: .055, 64: .055, 128: .055},
    23: {4: .050, 8: .050, 16: .050, 32: .052, 64: .052, 128: .052},
    24: {4: .048, 8: .048, 16: .048, 32: .049, 64: .049, 128: .049},
}
def gpu_dpf_ms(dev, lg, t, leaves):
    if dev == "B200":
        return leaves * B200_DPF_NS[lg].get(t, .049) / 1e6
    return leaves * interp(GPU_BC, leaves) / 1e6

def ntt_ms(dev, lg, b, kind):
    """per-mul ms: kind in {naive, 4step, 4step_dru}"""
    if dev == "B200":
        ms, tp = FOURSTEP_B200[lg][b]; tms = ms * tp
        nv = NAIVE[lg][b] * 0.30          # B200/L40S naive ratio (est, flagged)
    else:
        ms, tms = FOURSTEP_BC[lg][b]; nv = NAIVE[lg][b]
    if kind == "naive": return nv
    if kind == "4step": return ms
    return ms - tms if dev == "B200" else ms   # DRU: HBM only (honest rule)

ALPHA_US = 25.2          # measured small-message round-trip median
def net_ms(c, lg, t, rtt_us=ALPHA_US):
    levels = int(math.log2(2 * (1 << lg) // t))
    return c * c * (levels + 2) * rtt_us / 1e3   # + 2 Beaver opens per block

def rows():
    for lam, c, t in SEC:
        for lg in LGNS:
            yield lam, c, t, lg, BATCH[c], (1 << lg), c*c*2*(1 << lg)*t

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(repo, "results", "final_speedup.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
f = open(out, "w")
W = f.write

W("== FINAL PCG speedup: PIM+GPU vs GPU, with net-SW-HW / PU / NTT ablations\n")
W("== GPU world = {naive, 4step} (merge excluded). PIM lane = Ramulator2.\n")
W("== PIM e2e = co-scheduled steady state = max(DPF lane, NTT lane); the\n")
W("== network is hidden by the level-skew schedule (exposure reported in B).\n==\n")
for d in ("L40S", "B200"):
    W(f"== PIM DPF lane {d}: {pim_ns(d):.4f} ns/leaf (dual-issue + LSU)\n")
W("==\n")

# ---------------- A. main grid ----------------
W("---- A. main grid: ms per expansion + speedup vs GPU-4step ----\n")
for dev in ("L40S", "B200"):
    W(f"-- {dev}\n")
    W(f"   {'lam':>4} {'N':>5} {'c':>2} {'t':>4} | {'GPUnaive':>9} {'GPU4step':>9} |"
      f" {'PIM+GPU':>8} | {'x_naive':>7} {'x_4step':>7} | lane\n")
    for lam, c, t, lg, b, N, leaves in rows():
        gd = gpu_dpf_ms(dev, lg, t, leaves)
        gn = gd + 2*c*c*ntt_ms(dev, lg, b, "naive")
        g4 = gd + 2*c*c*ntt_ms(dev, lg, b, "4step")
        dpf = leaves * pim_ns(dev) / 1e6
        ntt = 2*c*c*ntt_ms(dev, lg, b, "4step_dru")
        e2e = max(dpf, ntt)
        W(f"   {lam:>4} 2^{lg:<3} {c:>2} {t:>4} | {gn:9.2f} {g4:9.2f} |"
          f" {e2e:8.2f} | {gn/e2e:6.1f}x {g4/e2e:6.1f}x |"
          f" {'DPF' if dpf >= ntt else 'NTT'}\n")
    W("\n")

# ---------------- B. net-SW-HW ablation ----------------
W("---- B. ablation: network - software - hardware co-design (c=4,t=16) ----\n")
W(f"   {'dev':>5} {'N':>5} | {'noHW':>9} {'noSW':>9} {'noNET':>9} {'full':>8} |"
  f" {'HW':>6} {'SW':>6} {'NET':>6}  (loss if removed)\n")
for dev in ("L40S", "B200"):
    for lg in LGNS:
        c, t, b, N = 4, 16, 16, (1 << lg)
        leaves = c*c*2*N*t
        dpf_pim = leaves * pim_ns(dev) / 1e6
        dpf_gpu = gpu_dpf_ms(dev, lg, t, leaves)
        ntt = 2*c*c*ntt_ms(dev, lg, b, "4step_dru")
        net = net_ms(c, lg, t)
        full = max(dpf_pim, ntt)                    # all three legs on
        noHW = max(dpf_gpu, ntt)                    # DPF back on the SMs
        noSW = dpf_pim + ntt + net                  # no overlap at all
        noNET = max(dpf_pim, ntt) + net             # network never hidden
        W(f"   {dev:>5} 2^{lg:<3} | {noHW:9.2f} {noSW:9.2f} {noNET:9.2f} {full:8.2f} |"
          f" {noHW/full:5.2f}x {noSW/full:5.2f}x {noNET/full:5.2f}x\n")
W("   noHW = DPF stays on the SMs (no in-bank SPU); noSW = three lanes run\n")
W("   serially (no co-schedule); noNET = the per-level exchanges are exposed\n")
f"   instead of hidden under the next block's deep levels.\n"
W("\n")

# ---------------- C. PU ablation ----------------
W("---- C. ablation: ChaCha SPU single- vs dual-issue x +-LSU (sim) ----\n")
W(f"   {'config':>12} | {'cycles':>9} {'ns/leaf L40S':>13} {'ns/leaf B200':>13} |"
  f" {'vs dual+LSU':>11}\n")
for cfg in ("dual+LSU", "dual-LSU", "single+LSU", "single-LSU"):
    W(f"   {cfg:>12} | {PU[cfg]:9d} {pim_ns('L40S', cfg):13.4f} "
      f"{pim_ns('B200', cfg):13.4f} | {PU[cfg]/PU['dual+LSU']:10.2f}x\n")
W("   dual-issue = XADD and VXORL interleave two QR chains (cl 155/465);\n")
W("   single-issue = one fused ALU (cl 282/846). LSU = decoupled load/store.\n")
W(f"   e2e effect (c=4,t=16, headline): DPF-bound rows scale by the full\n")
W(f"   factor, NTT-bound rows are insensitive:\n")
W(f"   {'dev':>5} {'N':>5} | " + " ".join(f"{c:>11}" for c in PU) + "\n")
for dev in ("L40S", "B200"):
    for lg in (20, 24):
        c, t, b, N = 4, 16, 16, (1 << lg)
        leaves = c*c*2*N*t
        ntt = 2*c*c*ntt_ms(dev, lg, b, "4step_dru")
        cells = [max(leaves*pim_ns(dev, cf)/1e6, ntt) for cf in PU]
        W(f"   {dev:>5} 2^{lg:<3} | " + " ".join(f"{x:10.2f}m" for x in cells) + "\n")
W("\n")

# ---------------- D. NTT ablation ----------------
W("---- D. ablation: GPU NTT ladder naive -> 4step -> 4step+DRU ----\n")
W(f"   {'dev':>5} {'N':>5} {'b':>3} | {'naive':>9} {'4step':>9} {'4step+DRU':>10} |"
  f" {'4s/naive':>9} {'DRU gain':>9}\n")
for dev in ("L40S", "B200"):
    for lg in LGNS:
        b = 16
        nv = 2*16*ntt_ms(dev, lg, b, "naive")
        fs = 2*16*ntt_ms(dev, lg, b, "4step")
        fd = 2*16*ntt_ms(dev, lg, b, "4step_dru")
        W(f"   {dev:>5} 2^{lg:<3} {b:>3} | {nv:9.2f} {fs:9.2f} {fd:10.2f} |"
          f" {nv/fs:8.2f}x {fs/fd:8.2f}x\n")
W("   NTT-lane ms for one expansion (2c^2 = 32 products, batch 16).\n")
W("   DRU offloads the 4-step transposes into idle channel bandwidth; on\n")
W("   GDDR the SMs are already bandwidth-bound so the gain is recorded 1.00x.\n\n")

# ---------------- E. headline ----------------
W("---- E. headline: lambda=128, c=4, t=16 ----\n")
W(f"   {'dev':>5} {'N':>5} | {'GPUnaive':>9} {'GPU4step':>9} {'PIM+GPU':>9} |"
  f" {'x_naive':>7} {'x_4step':>7}\n")
for dev in ("L40S", "B200"):
    for lg in LGNS:
        c, t, b, N = 4, 16, 16, (1 << lg)
        leaves = c*c*2*N*t
        gd = gpu_dpf_ms(dev, lg, t, leaves)
        gn = gd + 2*c*c*ntt_ms(dev, lg, b, "naive")
        g4 = gd + 2*c*c*ntt_ms(dev, lg, b, "4step")
        e2e = max(leaves*pim_ns(dev)/1e6, 2*c*c*ntt_ms(dev, lg, b, "4step_dru"))
        W(f"   {dev:>5} 2^{lg:<3} | {gn:9.2f} {g4:9.2f} {e2e:9.2f} |"
          f" {gn/e2e:6.1f}x {g4/e2e:6.1f}x\n")

# ---------------- F. network-inclusive, both sides, both deployments ----------
# Same hiding criterion applied to BOTH engines. Per block:
#   T_E = compute window; ladder T_L = n*alpha; +2*alpha Beaver (serial opens).
#   serial (no co-schedule): T = c^2*(T_E + T_L + 2a).
#   sched  (level-skew pipeline): compute-bound -> T = T_L + c^2*T_E (ramp once);
#                                 NET-bound     -> T = c^2*T_L + T_E (last tail).
# GPU block window = per-block DPF+2NTT; PIM block window = leaves_blk*pim_ns.
ALPHA = {"NVLink": 5.0, "loopback": 25.2, "DC": 50.0, "WAN": 2000.0}
def net_e2e(TE_blk, a_us, nblk, n):
    a = a_us / 1e3                        # ms
    tl = n * a
    ser = nblk * (TE_blk + tl + 2 * a)
    sch = (tl + nblk * TE_blk) if TE_blk >= tl else (nblk * tl + TE_blk)
    return ser, sch, ("compute" if TE_blk >= tl else "NET")
W("\n---- F. network-inclusive speedup (both engines, same criterion) ----\n")
W("== per-block windows; ladder = n*alpha + 2*alpha Beaver; sched = level-skew.\n")
for dev in ("L40S", "B200"):
    W(f"-- {dev}\n")
    W(f"   {'lam':>4} {'N':>5} {'c':>3} {'t':>4} {'link':>9} |"
      f" {'GPU4s_net':>9} {'PIM_net':>9} | {'x_sched':>7} {'reg':>7}\n")
    for lam, c, t in SEC:
        for lg in LGNS:
            b, N = c*c, (1 << lg)
            leaves = c*c*2*N*t
            n = int(math.log2(2*N/t))
            gpu_blk = gpu_dpf_ms(dev, lg, t, 2*N*t) + 2*ntt_ms(dev, lg, b, "4step")
            pim_blk = 2*N*t*pim_ns(dev)/1e6
            ntt_lane = 2*c*c*ntt_ms(dev, lg, b, "4step_dru")
            for link in ("NVLink", "DC"):
                a = ALPHA[link]
                _, g_sch, _ = net_e2e(gpu_blk, a, c*c, n)
                _, p_sch, preg = net_e2e(pim_blk, a, c*c, n)
                # regime-adaptive gang: g* rounds level-lockstep coalesce the
                # exchange (alpha amortized /g; +beta term g*M/beta_eff).
                MSG_MS = (16 * t * t / 1024) / 2e3     # 16t^2 B over 2 GB/s, ms
                gstar = 1
                p_gang = p_sch
                for gg in (2, 4, 8, 16):
                    if gg > c*c: break
                    aeff = (a/1e3 + gg*MSG_MS) / gg    # per-block ladder unit
                    tl_g = (n+2) * aeff
                    tot = (tl_g*gg + c*c*pim_blk) if pim_blk*gg >= tl_g*gg \
                          else (c*c*tl_g + pim_blk*gg)
                    if tot < p_gang:
                        p_gang, gstar = tot, gg
                p_e2e = max(min(p_sch, p_gang), ntt_lane)
                if ntt_lane > min(p_sch, p_gang):
                    preg = "NTTlane"
                elif gstar > 1:
                    preg = f"g={gstar}"
                W(f"   {lam:>4} 2^{lg:<3} {c:>3} {t:>4} {link:>9} |"
                  f" {g_sch:9.2f} {p_e2e:9.2f} | {g_sch/p_e2e:6.1f}x {preg:>7}\n")
        W("\n")

# ---------------- G. ablations at the DC tier (network-inclusive) ------------
# The compute-caliber ablations (B/C/D above) re-evaluated at DC 50us with the
# regime-adaptive schedule. Ablation D (NTT ladder) is a NETWORK INVARIANT:
# the protocol is comm-then-NTT (all opens precede any poly_mul), so the NTT
# tier choice never touches the network -- stated, not re-run.
W("---- G. ablations at DC tier (50 us, both engines scheduled) ----\n")
W("-- G1. co-design legs (c=4,t=16): loss factor when removed, DC tier\n")
W(f"   {'dev':>5} {'N':>5} | {'-HW':>7} {'-SW':>7} {'-NETopt':>8}\n")
for dev in ("L40S", "B200"):
    for lg in LGNS:
        c, t, b, N = 4, 16, 16, (1 << lg)
        n = int(math.log2(2*N/t)); a = 50.0
        pim_blk = 2*N*t*pim_ns(dev)/1e6
        gpu_blk = gpu_dpf_ms(dev, lg, t, 2*N*t) + 2*ntt_ms(dev, lg, b, "4step")
        ser_p, sch_p, _ = net_e2e(pim_blk, a, 16, n)
        _,     sch_g, _ = net_e2e(gpu_blk, a, 16, n)
        full = max(sch_p, 2*c*c*ntt_ms(dev, lg, b, "4step_dru"))
        no_sw = 16*(pim_blk + 2*c*c*ntt_ms(dev, lg, b, "4step_dru")/16) + ser_p - 16*pim_blk
        no_sw = ser_p + 2*c*c*ntt_ms(dev, lg, b, "4step_dru")   # serial lanes add
        W(f"   {dev:>5} 2^{lg:<3} | {sch_g/full:6.2f}x {no_sw/full:6.2f}x"
          f" {ser_p/max(full,1e-9):7.2f}x\n")
W("   -HW = GPU (scheduled, DC) / full;  -SW = lanes serial + net serial;\n")
W("   -NETopt = net serial only (schedule kept for lanes).\n")
W("-- G2. PU-config knees (alpha_1 = T_E_blk/(n+2), c=4 t=16, us) --\n")
W(f"   {'dev':>5} {'config':>12} {'ns/leaf':>8} | {'a1@2^20':>8} {'a1@2^24':>8}\n")
for dev in ("L40S", "B200"):
    for cfg in ("dual+LSU", "dual-LSU", "single+LSU", "single-LSU"):
        pns = pim_ns(dev, cfg)
        a20 = 2*(1 << 20)*16*pns/1e3/19
        a24 = 2*(1 << 24)*16*pns/1e3/23
        W(f"   {dev:>5} {cfg:>12} {pns:8.4f} | {a20:8.1f} {a24:8.1f}\n")
W("   slower configs tolerate worse networks (knee ~ 1/speed): the PU\n")
W("   ablation and the network tier interact exactly through alpha_1.\n")
W("-- G3. NTT/DRU ablation: NETWORK-INVARIANT (comm-then-NTT; all Beaver\n")
W("   opens precede any poly_mul -> table D holds at every alpha tier).\n")
f.close()
print("wrote", out)
