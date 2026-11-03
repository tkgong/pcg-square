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
for _ch in (8, 16, 32, 48, 96, 128, 192, 512):
    DEV[f"b200_ch{_ch}"] = dict(ch=_ch, tck=0.500)
# L40S variants: GDDR6 channels/timing/kernels with a hypothetical channel count (native 24 = 0.86 TB/s)
for _ch in (8, 16, 48, 96, 128, 192, 256, 512):
    DEV[f"l40s_ch{_ch}"] = dict(ch=_ch, tck=0.444)
def base(dev):
    """measured device a derived org inherits its tables from"""
    return dev.split("_")[0]
# B200: AICR job on fused4-dru a392780 (per multiply, min of 3 reps): merge_ms, f4g_sm_ms (four-step on GPU-NTT kernels,
# SM lane only = the DRU design's SM time), f4g_full_ms (the same four-step with the transposes on the SMs)
_B200 = {(20, 4): (.1291, .1251, .1422), (20, 16): (.1251, .1213, .1409), (20, 64): (.1234, .1203, .1383),
         (21, 4): (.2650, .2625, .3012), (21, 16): (.2594, .2567, .2939), (21, 64): (.2578, .2556, .2911),
         (22, 4): (.5463, .5369, .6155), (22, 16): (.5390, .5311, .6038), (22, 64): (.5368, .5291, .6002),
         (23, 4): (1.1297, 1.1074, 1.2583), (23, 16): (1.1232, 1.1023, 1.2477), (23, 64): (1.1210, 1.1002, 1.2442),
         (24, 4): (2.4100, 2.2850, 2.5838), (24, 16): (2.4015, 2.2766, 2.5695), (24, 64): (2.4008, 2.2769, 2.5678)}


def ntt_table(dev):
    dev = base(dev)
    if dev == "b200":
        return dict(_B200)
    lines = open(os.path.join(HERE, "ntt_l40s_fused4.csv")).read().splitlines()
    rows = csv.DictReader([lines[0]] + [l for l in lines if not l.startswith(("gpu,", "#"))])
    t = {}
    for r in rows:
        k = (int(r["logN"]), int(r["batch"])); m, s, f = float(r["merge_ms_mul"]), float(r["f4g_sm_ms_mul"]), float(r["f4g_full_ms_mul"])
        if s <= 0: continue
        o = t.get(k); t[k] = (min(m, o[0]), min(s, o[1]), min(f, o[2])) if o else (m, s, f)
    return t


# DRAM access pattern per design (reads in flight per channel, columns per row visit),
# fitted JOINTLY to L40S silicon (fit_ntt_pattern.py): standalone per-mul time and the
# interference curve of GPU_baseline/fused4 interfere_ntt, both with this kernel model.
#   merge : W=48 R=16 -> standalone +5.2% (logN 22) / +2.7% (24), interference rms 0.053
#   f4dru : W=48 R=16 -> standalone +0.0% / +0.0%, interference rms 0.042
PATTERN = {"merge": (48, 16), "f4dru": (48, 16), "f4sm": (48, 16), "sq": (48, 16), "sqdru": (48, 16), "sqdruh": (48, 16),
           "cheddar": (64, 24), "f4c": (48, 16), "f4csm": (48, 16),   # cheddar, f4c: fit_ntt_pattern.py on cheddar_interfere_l40s.csv
           "ole": (64, 24), "ole_t": (48, 16), "olep": (64, 24), "olepb": (64, 24)}   # ole: Cheddar's strided column kernel dominates -> Cheddar's pattern; ole_t: contiguous column reads -> f4c's (assumption, not refitted)


# Cheddar's NTT kernels (scale-snu/cheddar-fhe 8df8b26, GPU_baseline/cheddar), measured per multiply, best config per
# (logN, batch), min over configs: cheddar = Cheddar's own two-kernel NTT on the SMs (2 passes per transform);
# f4c = four-step on Cheddar's kernels, both passes contiguous, SM lane only (the transposes on the DRU);
# f4csm = the same four-step with the six transposes on the SMs.
_CH = {}
def cheddar_table(dev):
    dev = base(dev)
    if dev in _CH: return _CH[dev]
    # cheddar_bigN_<dev>.csv once it covers logN 20..24: both designs swept over the same configs (column M1 3..5, row M2
    # 3..5, register caps); cheddar_f4dru_<dev>.csv swept the four-step's column kernel at M1 = 3 only
    res = os.path.abspath(os.path.join(HERE, "..", "..", "..", "GPU_baseline", f"results_{dev}"))
    path = os.path.join(res, f"cheddar_bigN_{dev}.csv")
    if not os.path.exists(path) or not {20, 21, 22, 23, 24} <= {int(r["logN"]) for r in csv.DictReader(open(path)) if r["design"] == "f4c_sm"}:
        path = os.path.join(res, f"cheddar_f4dru_{dev}.csv")
    t = {}
    for r in csv.DictReader(l for l in open(path) if not l.startswith("#")):
        if r["design"] not in ("cheddar", "f4c_sm", "f4c_full") or int(r["mismatches"]) != 0: continue
        k = (int(r["logN"]), int(r["batch"])); i = ("cheddar", "f4c_sm", "f4c_full").index(r["design"])
        v = list(t.get(k, (1e9, 1e9, 1e9))); v[i] = min(v[i], float(r["ms_per_poly"])); t[k] = tuple(v)
    if F4C_SM_BOUND:
        # the four-step's SM kernels are Cheddar's kernels (row kernel as is, column kernel on contiguous columns), same 7
        # launches, 120N B and butterflies per multiply: Cheddar's measured time bounds them from above, so the port's
        # launch-shape loss (no warp batching before b1b5653) is not charged; the SM transposes keep their measured extra
        t = {k: (ch, min(sm, ch), full - sm + min(sm, ch)) for k, (ch, sm, full) in t.items()}
    _CH[dev] = t; return t


F4C_SM_BOUND = False   # set by e2e_window.py --f4c-sm-bound


# PCG output on Cheddar's kernels (cheddar_bigN_bench.cu --designs ole): z = INTT( sum_ij a^_i a^_j NTT(g_ij) ) per output,
# c^2 = batch polynomials g_ij: c forward transforms of the public a_i, one column pass over the g_ij, the row pass with the
# multiply-accumulate epilogue (G partial sums), the partial-sum reduction, one inverse transform.  ole = g_ij read in
# Cheddar's natural layout; ole_t = g_ij read in the column-contiguous layout the SPUs write.  The csv's ms_per_poly is per
# g_ij (time per output / c^2); the table holds (ms per OUTPUT, P1 of that config) per design, best config per (logN, batch).
OLE = ("ole", "ole_t", "olep", "olepb")   # olep: the SPUs accumulated g_ij + g_ji into one buffer -> c(c+1)/2 unordered pairs
_OLE = {}
def ole_table(dev):
    dev = base(dev)
    if dev in _OLE: return _OLE[dev]
    path = os.path.abspath(os.path.join(HERE, "..", "..", "..", "GPU_baseline", f"results_{dev}", f"cheddar_bigN_{dev}.csv"))
    t = {}
    if os.path.exists(path):
        for r in csv.DictReader(l for l in open(path) if not l.startswith("#")):
            if r["design"] not in OLE or int(r["mismatches"]) != 0: continue
            k = (int(r["logN"]), int(r["batch"])); i = OLE.index(r["design"]); ms = float(r["ms_per_poly"]) * k[1]
            v = list(t.get(k, ((1e9, 0),) * len(OLE)))
            if ms < v[i][0]: v[i] = (ms, int(r["P1"]))
            t[k] = tuple(v)
    _OLE[dev] = t; return t


def muls_per_output(design, c):
    """how many `mul_ms` units one PCG output takes: 2c^2 multiplies (old algorithm), one output (ole designs)"""
    return 1 if design in OLE else 2 * c * c


def ole_groups(logN, P1, items):
    """partial-sum groups of the RowsAcc kernel (cheddar_bigN_bench.cu RowsAcc::groups): 2^P1 row blocks x G >= 1024, G <= items, G | items"""
    G = 1
    while (1 << P1) * G < 1024 and G < items: G *= 2
    while items % G: G -= 1
    return G


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
    if design in ("merge", "f4dru", "f4sm"): return ntt_table(dev)[(logN, batch)][{"merge": 0, "f4dru": 1, "f4sm": 2}[design]]
    if design in ("cheddar", "f4c", "f4csm"): return cheddar_table(dev)[(logN, batch)][{"cheddar": 0, "f4c": 1, "f4csm": 2}[design]]
    if design in OLE: return ole_table(dev).get((logN, batch), ((1e9, 0),) * len(OLE))[OLE.index(design)][0]   # per OUTPUT (c^2 = batch polynomials)
    if design == "sqdruh": design = "sqdru"
    tot, st = sq_table(dev)[(logN, batch)]; ssum = sum(st.values())
    return tot * (1.0 if design == "sq" else (ssum - st["transpose"] - st["brev"]) / ssum)


# Effective DRAM-traffic fraction per config (L2 absorbs the rest), fitted so that the
# simulated standalone per-mul time equals silicon (calibrate_ntt_bytes.py). Missing = 1.
_BS_PATH = os.path.join(HERE, "ntt_dram_fraction.json")
DRAM_FRACTION = __import__("json").load(open(_BS_PATH)) if os.path.exists(_BS_PATH) else {}


# First DRAM row of the SPUs' g output in every bank (gen_dpf_tree_trace.py, pass 2: all-bank GGM_REDUCE writes at
# g_row = (1 << 10) + 512 + ...). With handoff=True the phase that reads g (the DPF -> NTT handoff) reads these rows
# (optional 10th phase field rd_row_base) instead of a private region, so it meets the SPUs' rows in the same banks.
SPU_G_ROW = (1 << 10) + 512


def phases(dev, design, logN, batch, scale=1.0, window=None, run=None, gap=0, dru_window=64, frac=None, handoff=False):
    window = window or PATTERN[design][0]; run = run or PATTERN[design][1]
    if frac is None:
        frac = DRAM_FRACTION.get(f"{base(dev)}/{design}/{logN}/{batch}", 1.0)
    d = DEV[dev]; N = 1 << logN; tm, ts = ntt_table(dev)[(logN, batch)][:2]
    ck = lambda ms: int(round(ms * 1e6 / d["tck"] * scale))
    by = lambda b: int(round(b * N / d["ch"] * scale * frac))
    byd = lambda b: int(round(b * N / d["ch"] * scale))      # DRU transposes: data not in L2
    if design in ("sq", "sqdru", "sqdruh"):
        tot, st = sq_table(dev)[(logN, batch)]; f = tot / sum(st.values())      # stage floors scaled to the per-mul total
        K = lambda ms, nb=16: (0, 1, by(nb), 0.5, window, gap, run, 16384, ck(ms * f))
        if design == "sq":
            T = lambda: K(st["transpose"] / 6); B = lambda: K(st["brev"] / 6)
        elif design == "sqdruh":
            # conservative DRU: its transposes go through the ordinary bank scheduler as a host stream of class 3
            # (below the GPU in priority): they pay ACT/PRE, row misses and read/write turnarounds like GPU traffic
            T = lambda: (1, 3, byd(16), 0.5, window, 0, run, 32768, 0); B = T
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
    if design == "cheddar":
        # Cheddar: 2 kernels per transform (8N rd + 8N wr each); poly-mul = fwd a, fwd b, pointwise, inverse -> 7 kernels, 120N B
        tc = cheddar_table(dev)[(logN, batch)][0]
        CK = lambda: (0, 1, by(16), 0.5, window, gap, run, 16384, ck(tc * 16 / 120))
        return [CK(), CK(), CK(), CK(), (0, 1, by(24), 16 / 24, window, gap, run, 16384, ck(tc * 24 / 120)), CK(), CK()]
    if design in OLE:
        # PCG output on Cheddar's kernels, per output (c^2 = batch polynomials g_ij, c public a_i), 7 launches:
        #   Cols a (16N c B) | Rows a (16N c) | Cols g (16N c^2) | RowsAcc (reads g 8N c^2 + a^_i, a^_j 16N c^2, writes G partials 8N G)
        #   | reduce (G partials -> 1: 8N G + 8N; absent when G = 1) | Rows inv (16N) | Cols inv (16N)
        # compute floor = measured per-output time split by bytes (as the other designs); the a^ reads are c distinct
        # polynomials (L2 reuse is what the DRAM-fraction calibration absorbs).  No DPF->NTT gather: the column pass reads
        # the SPUs' output directly (ole_t: in the column-contiguous layout they write, the same bytes).
        to, P1 = ole_table(dev)[(logN, batch)][OLE.index(design)]
        c = int(round(batch ** 0.5)); items = c * (c + 1) // 2 if design in ("olep", "olepb") else batch; G = ole_groups(logN, P1, items)
        if design == "olepb":   # b^_ij precomputed once per public a: no a transforms; the accumulate pass reads g and one b^ per item
            parts = [(16 * items, 0.5), (16 * items + 8 * G, 16 * items / (16 * items + 8 * G))]
        else:
            parts = [(16 * c, 0.5), (16 * c, 0.5), (16 * items, 0.5), (24 * items + 8 * G, 24 * items / (24 * items + 8 * G))]
        if G > 1: parts.append((8 * G + 8, 8 * G / (8 * G + 8)))
        parts += [(16, 0.5), (16, 0.5)]
        tot = sum(b for b, _ in parts)
        ph = [(0, 1, by(b), rf, window, gap, run, 16384, ck(to * b / tot)) for b, rf in parts]
        if handoff:   # the column pass over g (index 0 for olepb, 2 otherwise) reads the SPUs' g rows
            i = 0 if design == "olepb" else 2
            ph[i] = ph[i] + (SPU_G_ROW,)
        return ph
    if design in ("f4c", "f4csm"):
        # four-step on Cheddar's kernels: per mul  T K T K (a) | T K T K (b) | PW | K T K T; SM kernels 6 x 16N + PW 24N = 120N B,
        # 6 transposes x 16N B on the DRU (f4c) or on the SMs (f4csm, floor = measured f4c_full - f4c_sm)
        _, tc, tf = cheddar_table(dev)[(logN, batch)]
        K = lambda: (0, 1, by(16), 0.5, window, gap, run, 16384, ck(tc * 16 / 120))
        if design == "f4csm":
            T = lambda: (0, 1, byd(16), 0.5, window, gap, run, 32768, ck((tf - tc) / 6))
        else:
            T = lambda: (1, 2, byd(16), 0.5, dru_window, 0, 4, 32768, 0)
        PW = (0, 1, by(24), 16 / 24, window, gap, run, 16384, ck(tc * 24 / 120))
        return [T(), K(), T(), K(), T(), K(), T(), K(), PW, K(), T(), K(), T()]
    K = lambda: (0, 1, by(20), 0.6, window, gap, run, 16384, ck(ts * 20 / 144))
    if design == "f4sm":
        # the same four-step with the six transposes on the SMs: measured f4g_full - f4g_sm is their SM time
        tf = ntt_table(dev)[(logN, batch)][2]
        T = lambda: (0, 1, byd(16), 0.5, window, gap, run, 32768, ck((tf - ts) / 6))
    else:
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
