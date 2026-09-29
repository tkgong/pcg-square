#!/usr/bin/env python3
"""NTT poly-mul job pipelines for Ramulator2 host_jobs (GPU kernels with compute
floors + DRU transposes), built from measured per-mul kernel times.

Designs
  merge : GPU-NTT merge poly-mul on the SMs, one phase, 168N B (52% reads).
  f4dru : fused four-step on GPU-NTT kernels, transposes on the DRU:
          per mul  T K1 T K2 (a) | T K1 T K2 (b) | PW | K3 T K4 T
          SM kernels 6 x 20N B + PW 24N B = 144N B (61% reads); 6 transposes x 16N B
          on the DRU (50% reads), issued through the controller's DRU tap (one column per
          nBL on the shared data bus, no ACT/PRE). SM compute floor = measured f4g_sm time split by bytes.
Bytes and floors are per real channel (device bytes / channels) times `scale`.
"""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = {"l40s": dict(ch=24, tck=0.444), "b200": dict(ch=256, tck=0.500)}
# B200: AICR job on fused4-dru a392780 (merge_ms, f4g_sm_ms per multiply, min of 3 reps)
_B200 = {(20, 4): (.1291, .1251), (20, 16): (.1251, .1213), (20, 64): (.1234, .1203),
         (21, 4): (.2650, .2625), (21, 16): (.2594, .2567), (21, 64): (.2578, .2556),
         (22, 4): (.5463, .5369), (22, 16): (.5390, .5311), (22, 64): (.5368, .5291),
         (23, 4): (1.1297, 1.1074), (23, 16): (1.1232, 1.1023), (23, 64): (1.1210, 1.1002),
         (24, 4): (2.4100, 2.2850), (24, 16): (2.4015, 2.2766), (24, 64): (2.4008, 2.2769)}


def ntt_table(dev):
    if dev == "b200":
        return dict(_B200)
    lines = open(os.path.join(HERE, "ntt_l40s_fused4.csv")).read().splitlines()
    rows = csv.DictReader([lines[0]] + [l for l in lines if not l.startswith(("gpu,", "#"))])
    t = {}
    for r in rows:
        k = (int(r["logN"]), int(r["batch"])); m, s = float(r["merge_ms_mul"]), float(r["f4g_sm_ms_mul"])
        if s <= 0: continue
        o = t.get(k); t[k] = (min(m, o[0]), min(s, o[1])) if o else (m, s)
    return t


# DRAM access pattern per design, fitted to the L40S silicon interference curves
# (calib_ntt_patterns.py vs GPU_baseline/fused4 interfere_ntt): reads in flight per
# channel and columns per row visit.
PATTERN = {"merge": (32, 16), "f4dru": (32, 32)}


def phases(dev, design, logN, batch, scale=1.0, window=None, run=None, gap=0, dru_window=64):
    window = window or PATTERN[design][0]; run = run or PATTERN[design][1]
    d = DEV[dev]; N = 1 << logN; tm, ts = ntt_table(dev)[(logN, batch)]
    ck = lambda ms: int(round(ms * 1e6 / d["tck"] * scale))
    by = lambda b: int(round(b * N / d["ch"] * scale))
    if design == "merge":
        # GPU-NTT merge at 2^20..2^24: 3 kernels per transform (8N rd + 8N wr each);
        # poly-mul = fwd a, fwd b, pointwise (16N rd + 8N wr), inverse -> 10 kernels, 168N B
        MK = lambda: (0, 1, by(16), 0.5, window, gap, run, 16384, ck(tm * 16 / 168))
        MPW = (0, 1, by(24), 16 / 24, window, gap, run, 16384, ck(tm * 24 / 168))
        return [MK(), MK(), MK(), MK(), MK(), MK(), MPW, MK(), MK(), MK()]
    K = lambda: (0, 1, by(20), 0.6, window, gap, run, 16384, ck(ts * 20 / 144))
    T = lambda: (1, 2, by(16), 0.5, dru_window, 0, 4, 32768, 0)
    PW = (0, 1, by(24), 16 / 24, window, gap, run, 16384, ck(ts * 24 / 144))
    return [T(), K(), T(), K(), T(), K(), T(), K(), PW, K(), T(), K(), T()]


def write_jobs(path, pipes):
    """pipes: list of (njobs, start_ck, phase_list)."""
    with open(path, "w") as f:
        f.write("# jobs <count> <start_ck> / phase <res> <class> <bytes_per_ch> <rd_frac> <window> <gap> <run> <row_base> <min_ck>\n")
        for n, st, ph in pipes:
            f.write(f"jobs {n} {st}\n")
            for p in ph:
                f.write("phase " + " ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in p) + "\n")
