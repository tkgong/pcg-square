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
# Channel-count sweep (Q7): B200's HBM3e channels/timing/kernels with a hypothetical channel count;
# bytes per channel scale with 256/ch, kernel compute floors stay (same SMs), SPUs = 8 per channel.
for _ch in (48, 96, 128, 192, 512):
    DEV[f"b200_ch{_ch}"] = dict(ch=_ch, tck=0.500)
def base(dev):
    """measured device a derived org inherits its tables from"""
    return dev.split("_")[0]
# B200: AICR job on fused4-dru a392780 (merge_ms, f4g_sm_ms per multiply, min of 3 reps)
_B200 = {(20, 4): (.1291, .1251), (20, 16): (.1251, .1213), (20, 64): (.1234, .1203),
         (21, 4): (.2650, .2625), (21, 16): (.2594, .2567), (21, 64): (.2578, .2556),
         (22, 4): (.5463, .5369), (22, 16): (.5390, .5311), (22, 64): (.5368, .5291),
         (23, 4): (1.1297, 1.1074), (23, 16): (1.1232, 1.1023), (23, 64): (1.1210, 1.1002),
         (24, 4): (2.4100, 2.2850), (24, 16): (2.4015, 2.2766), (24, 64): (2.4008, 2.2769)}


def ntt_table(dev):
    dev = base(dev)
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


# DRAM access pattern per design (reads in flight per channel, columns per row visit),
# fitted JOINTLY to L40S silicon (fit_ntt_pattern.py): standalone per-mul time and the
# interference curve of GPU_baseline/fused4 interfere_ntt, both with this kernel model.
#   merge : W=48 R=16 -> standalone +5.2% (logN 22) / +2.7% (24), interference rms 0.053
#   f4dru : W=48 R=16 -> standalone +0.0% / +0.0%, interference rms 0.042
PATTERN = {"merge": (48, 16), "f4dru": (48, 16), "sq": (48, 16), "sqdru": (48, 16)}


# The submission's own NTT: the square four-step (poly_mul_gpuntt_square.cu, standalone bit-reversal), measured per
# stage in ntt_stages.csv. Per multiply: pre-twist a, b | 3 transforms, each [transpose, brev, sub-NTT, twiddle,
# transpose, brev, sub-NTT] | pointwise | post-twist = 25 DRAM passes, 408N B. "sq" runs everything on the SMs;
# "sqdru" is the paper's DRU design: the 6 transposes and 6 bit-reversal passes go to the DRU (its dru_share).
_SQ = {}
def sq_table(dev):
    dev = base(dev)
    if dev in _SQ: return _SQ[dev]
    path = os.path.join(os.path.dirname(os.path.dirname(HERE)), "..", "GPU_baseline", "pcg_baseline_out" if dev == "b200" else "data_l40s", "ntt_stages.csv")
    t = {}
    for r in csv.DictReader(open(os.path.abspath(path))):
        if r["status"] != "ok" or r["backend"] != "square" or r["brev_mode"] != "standalone": continue
        k = (int(r["logN"]), int(r["batch"])); st = {x: float(r[x + "_ms"]) / k[1] for x in ("transpose", "brev", "ntt", "twiddle", "twist", "pointwise")}
        t[k] = (float(r["ms_per_mul"]), st)          # per-multiply total, per-multiply stage times
    _SQ[dev] = t; return t


def mul_ms(dev, design, logN, batch):
    """measured per-multiply NTT time the lane is sized from (sqdru: the SM side only, the DRU runs in parallel)"""
    if design in ("merge", "f4dru"): return ntt_table(dev)[(logN, batch)][0 if design == "merge" else 1]
    tot, st = sq_table(dev)[(logN, batch)]; ssum = sum(st.values())
    return tot * (1.0 if design == "sq" else (ssum - st["transpose"] - st["brev"]) / ssum)


# Effective DRAM-traffic fraction per config (L2 absorbs the rest), fitted so that the
# simulated standalone per-mul time equals silicon (calibrate_ntt_bytes.py). Missing = 1.
_BS_PATH = os.path.join(HERE, "ntt_dram_fraction.json")
DRAM_FRACTION = __import__("json").load(open(_BS_PATH)) if os.path.exists(_BS_PATH) else {}


def phases(dev, design, logN, batch, scale=1.0, window=None, run=None, gap=0, dru_window=64, frac=None):
    window = window or PATTERN[design][0]; run = run or PATTERN[design][1]
    if frac is None:
        frac = DRAM_FRACTION.get(f"{base(dev)}/{design}/{logN}/{batch}", 1.0)
    d = DEV[dev]; N = 1 << logN; tm, ts = ntt_table(dev)[(logN, batch)]
    ck = lambda ms: int(round(ms * 1e6 / d["tck"] * scale))
    by = lambda b: int(round(b * N / d["ch"] * scale * frac))
    byd = lambda b: int(round(b * N / d["ch"] * scale))      # DRU transposes: data not in L2
    if design in ("sq", "sqdru"):
        tot, st = sq_table(dev)[(logN, batch)]; f = tot / sum(st.values())      # stage floors scaled to the per-mul total
        K = lambda ms, nb=16: (0, 1, by(nb), 0.5, window, gap, run, 16384, ck(ms * f))
        if design == "sq":
            T = lambda: K(st["transpose"] / 6); B = lambda: K(st["brev"] / 6)
        else:
            T = lambda: (1, 2, byd(16), 0.5, dru_window, 0, 4, 32768, 0); B = T
        X = lambda: [T(), B(), K(st["ntt"] / 6), K(st["twiddle"] / 3), T(), B(), K(st["ntt"] / 6)]
        tw = lambda: K(st["twist"] / 3)
        return [tw(), tw()] + X() + X() + [(0, 1, by(24), 16 / 24, window, gap, run, 16384, ck(st["pointwise"] * f))] + X() + [tw()]
    if design == "merge":
        # GPU-NTT merge at 2^20..2^24: 3 kernels per transform (8N rd + 8N wr each);
        # poly-mul = fwd a, fwd b, pointwise (16N rd + 8N wr), inverse -> 10 kernels, 168N B
        MK = lambda: (0, 1, by(16), 0.5, window, gap, run, 16384, ck(tm * 16 / 168))
        MPW = (0, 1, by(24), 16 / 24, window, gap, run, 16384, ck(tm * 24 / 168))
        return [MK(), MK(), MK(), MK(), MK(), MK(), MPW, MK(), MK(), MK()]
    K = lambda: (0, 1, by(20), 0.6, window, gap, run, 16384, ck(ts * 20 / 144))
    T = lambda: (1, 2, byd(16), 0.5, dru_window, 0, 4, 32768, 0)
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
