#!/usr/bin/env python3
"""Independent lane model for the PCG^2 end-to-end results (read-only on repo).

Builds, per (machine, c, t, logN), the lanes used by the paper's E2E model:

  gpu_dpf   GPU DPF+convert compute with the network wait subtracted
            (expand + convert + beaver - (exchanges + 2c^2) * alpha_meas)
  ntt_dev   device-only four-step NTT: ms_per_mul(logN, batch=c^2) * 2c^2
  ntt_pcie  the measured NTT phase of the e2e run (includes D->H->D copies)
  spu       leaves * pim_ns * occupancy
  sm        ntt_dev (L40S) / ntt_dev * (1 - DRU share) (B200)
  n         GGM depth = logN + 1 - log2 t

All sources are CSVs in the repo; nothing is written into the repo.
"""
import csv, math, os
from collections import defaultdict

REPO = "/home/tkgong/pcg-square"
GPU = os.path.join(REPO, "GPU_baseline")

# ---- PIM anchors ---------------------------------------------------------
PIM_NS_L40S = 718660 * 0.444 / (768 * 4096)        # l40s_pim_vs_gpu.py:39
PIM_NS_B200_DERIVED = PIM_NS_L40S * 0.1069         # l40s_pim_vs_gpu.py:46
# direct B200 anchors (measured_cycles.txt): B_n12 = 732,860 CK on one
# 1024-SPU die domain with I=4096 (4/SPU, same as L40S I=768 on 192 SPU);
# the full card is two independent domains -> 2x leaves in the same time.
PIM_NS_B200_DIRECT_n12 = 732860 * 0.500 / (2 * 4096 * 4096)
PIM_NS_B200_DIRECT_n10 = 181648 * 0.500 / (2 * 4096 * 1024)
# final_speedup.py:40 MEAS_B200 dual+LSU 701,600 CK, 128 ch x 131072 x 2 cards
PIM_NS_B200_FINAL = 701600 * 0.500 / (128 * 8 * 4 * 4096 * 2)
NSPU = {"L40S": 192, "B200": 2048}
CK = {}
for _l in open(os.path.join(REPO, "PIM/results/sim_ck/measured_cycles.txt")):
    _k, _v = _l.split()
    CK[_k] = int(_v)

CFG = [(8, 4), (4, 16), (8, 8), (2, 64), (2, 128)]
LAM = {(8, 4): 80, (4, 16): 80, (8, 8): 128, (2, 64): 80, (2, 128): 128}
LOGN = [20, 21, 22, 23, 24]


def gm(v):
    v = list(v)
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


def rd(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ntt_table(path, mode="standalone"):
    """mode: 'standalone' | 'fold' | 'best' (per-cell faster of the two)."""
    if mode == "best":
        a, b = ntt_table(path, "standalone"), ntt_table(path, "fold")
        return {k: (a[k] if a[k]["ms"] <= b[k]["ms"] else b[k]) for k in a}
    out = {}
    for r in rd(path):
        if r["status"] != "ok" or r["backend"] != "square" \
                or r["brev_mode"] != mode:
            continue
        k = (int(r["logN"]), int(r["batch"]))
        st = [float(r[x]) for x in ("transpose_ms", "brev_ms", "ntt_ms",
                                    "twiddle_ms", "twist_ms", "pointwise_ms")]
        out[k] = dict(ms=float(r["ms_per_mul"]), share=(st[0] + st[1]) / sum(st),
                      tshare=st[0] / sum(st), stages=st, sum=float(r["sum_ms"]))
    return out


def gpu_rows(mach):
    """(c,t,logN,arm) -> list of measured rows, each with its alpha (ms)."""
    if mach == "L40S":
        rows = rd(os.path.join(GPU, "data_l40s", "alpha_sweep.csv"))
        akey = "alpha_us"
    else:
        rows = rd(os.path.join(GPU, "pcg_baseline_out", "e2e_full.csv"))
        akey = "rtt_us"
    d = defaultdict(list)
    for r in rows:
        if r["status"] != "OK":
            continue
        c, t, lg = int(r["c"]), int(r["t"]), int(r["logN"])
        a = float(r[akey]) / 1000.0
        ex = int(r["exchanges"])
        comp_exp = float(r["expand_ms"]) - ex * a
        conv = float(r["convert_ms"])
        bv = float(r["beaver_ms"]) - 2 * c * c * a
        d[(c, t, lg, r["arm"])].append(dict(
            alpha=a, exp=float(r["expand_ms"]), comp=comp_exp + conv + bv,
            comp_exp=comp_exp, conv=conv, bv=bv, ntt=float(r["ntt_ms"]),
            wall=float(r["wall_ms"]), n=int(r["n"]), ex=ex))
    return d


def lanes(mach, dpf_alpha_ms=None, dpf_pick="min_alpha", arm_pref="serial",
          batch_cost=1.026, pim_ns=None, occupancy=True, dru=None,
          lam_last=True, spu_model="linear", ntt_mode="standalone",
          spu_scale=1.0):
    """Return {(c,t,logN): lane dict}.

    dpf_pick: 'min_alpha' -> the measured row with the smallest alpha
              'median'    -> median of the compute residual over all alphas
              'alpha'     -> the row at dpf_alpha_ms
    arm_pref: 'serial' | 'batched_if_available'
    """
    ntt = ntt_table(os.path.join(GPU, "data_l40s" if mach == "L40S"
                                 else "pcg_baseline_out", "ntt_stages.csv"), ntt_mode)
    g = gpu_rows(mach)
    if pim_ns is None:
        pim_ns = PIM_NS_L40S if mach == "L40S" else PIM_NS_B200_DERIVED
    if dru is None:
        dru = (mach == "B200")
    out = {}
    for (c, t) in CFG:
        for lg in LOGN:
            arm = "serial"
            rows = g.get((c, t, lg, "serial"))
            if arm_pref == "batched_if_available" and g.get((c, t, lg, "batched")):
                rows, arm = g[(c, t, lg, "batched")], "batched"
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: r["alpha"])
            if dpf_pick == "min_alpha":
                # dict semantics "last row wins": for (4,16), which appears in
                # both lambda blocks, the lambda=128 row (later in the file)
                same = [r for r in rows if r["alpha"] == rows[0]["alpha"]]
                comp = same[-1 if lam_last else 0]["comp"]
            elif dpf_pick == "median":
                v = sorted(r["comp"] for r in rows)
                comp = v[len(v) // 2]
            elif dpf_pick == "min":
                comp = min(r["comp"] for r in rows)
            else:
                comp = [r for r in rows if abs(r["alpha"] - dpf_alpha_ms) < 1e-9][0]["comp"]
            nt = ntt[(lg, c * c)]
            ntt_dev = nt["ms"] * 2 * c * c
            n = lg + 1 - int(math.log2(t))
            assert n == rows[0]["n"], (c, t, lg, n, rows[0]["n"])
            I = c * c * t * t
            occ = max(1.0, NSPU[mach] / I) if occupancy else 1.0
            leaves = c * c * 2 * (1 << lg) * t
            spu = leaves * pim_ns / 1e6 * occ * spu_scale
            if spu_model == "ladder" and mach == "L40S":
                # per-level ramulator2 ladder (measured_cycles.txt PL_L_n*),
                # deepest measured increment doubled per extra level, output
                # layer (L_n12 - PL_L_n12) scaled with the leaf count
                E, inc = CK["PL_L_n12"], CK["PL_L_n12"] - CK["PL_L_n11"]
                O = CK["L_n12"] - CK["PL_L_n12"]
                ck = E + inc * (2 ** (n - 11) - 2) + O * 2 ** (n - 12)
                spu = ck * 0.444e-6 * I / 768 * occ * spu_scale
            sm = ntt_dev * (1 - nt["share"]) if dru else ntt_dev
            out[(c, t, lg)] = dict(
                c=c, t=t, lg=lg, n=n, I=I, arm=arm, gpu_dpf=comp,
                gpu_dpf_g=comp * (batch_cost if arm == "serial" else 1.0),
                ntt_dev=ntt_dev, ntt_nodru=ntt_dev, sm=sm, spu=spu, occ=occ,
                dru_share=nt["share"], ntt_pcie=rows[0]["ntt"],
                rows=rows, leaves=leaves, ntt_brev=nt["stages"][1])
    return out


def alpha_bw(c, t, beta_bps):
    """paper/plot_bw_bars: alpha = 2 * c^2 * 16 t^2 bytes / beta (ms)."""
    return 2 * c * c * 16 * t * t * 8 / beta_bps * 1e3
