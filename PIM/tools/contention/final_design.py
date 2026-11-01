#!/usr/bin/env python3
"""The one place that says what "PCG^2" and "GPU baseline" mean in every reported number (import this; do not hard-code
designs in the analysis scripts).

Accounting
  GPU baseline : the submission's GPU DPF (two-pass H') + Cheddar's NTT (the strongest SM-only NTT measured on both
                 cards, GPU_baseline/cheddar), network co-scheduled: T_base = max(T_dpf + T_ntt, T_net).
                 Row keys: GPU_KEY (DPF + Cheddar NTT), BASE_KEY (max with the network).
  PCG^2        : expansion on the SPUs; NTT = the runtime's choice per card among CANDIDATES (the design with the best
                 geomean end-to-end speedup over the suite at 40 Gbps under the final co-simulation flags), recorded in
                 <RUN>/final_choice.json the first time it is computed so every script reads the same choice;
                 PCG^2 = max(SPU lane, NTT lane, network) with the co-simulated contention.
  Table IV     : GPU baseline / +SPU (NTT = SM_NTT, Cheddar's kernels on the SMs, serial) / +SPU+DRU (NTT = the DRU
                 design, serial) / +SPU+co-scheduled / full. The DRU design is the butterfly DRU (f4b) where it is the
                 runtime's choice, else "DRU not used" (the rows repeat).
Designs (ntt_jobs.py): merge = GPU-NTT merge; cheddar = Cheddar's two-kernel NTT on the SMs; f4c / f4csm = four-step on
Cheddar's kernels with the transposes on the DRU / on the SMs; f4b = four-step whose column pass runs inside the DRU
(butterfly DRU, DRU_BF butterfly units per channel), SMs = row kernel + pointwise.

Table III units: SPU and reorder-DRU area/power are the synthesised values (2.25 GHz). The butterfly unit's area and power
are gate-count ESTIMATES (Shoup modmul: 1 full + 2 low-half 64x64 Booth-r4/Wallace multipliers, 4/3/3 pipeline stages,
~7,600 flip-flops; 28 nm std-cell density), NOT synthesised: every consumer prints ESTIMATE_NOTE next to them.
Environment: PCG2_RUN selects the window set (default win23)."""
import json, os
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, "..", "..", "results", "contention", "final_window"))
RUN = os.environ.get("PCG2_RUN", "win23")
RUN_DIR = os.path.join(FW, RUN)
MACH = {"l40s": "L40S", "b200": "B200"}; ORG = {v: k for k, v in MACH.items()}
GPU_KEY, BASE_KEY = "gpu_cheddar_ms", "base_overlap_cheddar_ms"
BASE_NTT = "cheddar"
SM_NTT = "cheddar"                                   # Table IV "+SPU": PCG^2 with its NTT entirely on the SMs
CANDIDATES = ("f4b", "f4c", "cheddar", "merge")      # the runtime's menu
gm = lambda v: (lambda w: exp(sum(map(log, w)) / len(w)))(list(v))


def cells(rows, org, des, tier="fast"):
    return {(r["c"], r["t"], r["logN"]): r for r in rows if r["org"] == org and r["design"] == des and r["tier"] == tier}


def speedup(r, nic=None):
    nic = r["nic_ms"] if nic is None else nic
    return max(r[GPU_KEY], nic) / max(r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"], nic)


def pick(rows, org, tier="fast"):
    """runtime choice: the candidate with the best geomean speedup over the cells every candidate has"""
    C = {d: cells(rows, org, d, tier) for d in CANDIDATES}; C = {d: c for d, c in C.items() if c}
    ks = set.intersection(*(set(c) for c in C.values()))
    g = {d: gm(speedup(c[k]) for k in ks) for d, c in C.items()}
    best = max(g, key=g.get)
    return best, g


def choice(run_dir=None):
    """{org: design} of the final run; computed from <run>/e2e_nom.json once and stored in <run>/final_choice.json"""
    run_dir = run_dir or RUN_DIR; p = os.path.join(run_dir, "final_choice.json")
    if os.path.exists(p): return json.load(open(p))
    src = next((os.path.join(run_dir, f) for f in ("e2e_nom.json", "e2e.json") if os.path.exists(os.path.join(run_dir, f))), None)
    if src is None:                                   # a raw run dir without rows of its own: the window set's choice
        if os.path.abspath(run_dir) != os.path.abspath(RUN_DIR): return choice(RUN_DIR)
        raise FileNotFoundError(f"{run_dir}: no e2e_nom.json / e2e.json to pick the final design from")
    rows = json.load(open(src))
    ch = {}
    for org in MACH:
        best, g = pick(rows, org); ch[org] = best; ch[org + "_geomean"] = g
    json.dump(ch, open(p, "w"), indent=1); return ch


def des(org, run_dir=None):
    return choice(run_dir)[org]


def dru_ntt(org, run_dir=None):
    """Table IV "+DRU" NTT: the DRU design when the runtime picked one, else the SM NTT (DRU not used)"""
    d = des(org, run_dir); return d if d in ("f4b", "f4c") else SM_NTT


def DES(run_dir=None):
    """{"L40S": ("l40s", design), "B200": ("b200", design)} - the old scripts' table"""
    return {MACH[o]: (o, des(o, run_dir)) for o in MACH}


# ---- Table III: units -------------------------------------------------------------------------------------------
SPU_MM2, SPU_MW = 0.253970, 34.391          # synthesised, 2.25 GHz (Table III)
DRU_REORDER_MM2, DRU_REORDER_MW = 0.024924, 4.994
BF_UNIT_MM2, BF_UNIT_MW = 0.065, 55.0       # ESTIMATE: one Shoup butterfly unit (3 multipliers, pipeline regs), 28 nm, 2 GHz
BF_BUF_MM2 = lambda p1: 2 * (8 << p1) / 1024 * 0.0011 + 0.015   # ESTIMATE: 2 column buffers of 2^p1 words (1.1 um^2/B 28 nm SRAM) + 16 KB twiddle SRAM
ESTIMATE_NOTE = "butterfly-DRU unit area/power are gate-count estimates (28 nm), not synthesised; SPU and reorder DRU are synthesised"
NSPU, NDRU = {"l40s": 192, "b200": 2048}, {"l40s": 24, "b200": 256}
CLK = {"l40s": 2.25, "b200": 2.0}           # GHz: DRAM clock the PIM logic runs at


def dru_bf(org, run_dir=None):
    """butterfly units per channel chosen for this card (<run>/dru_bf.json, written by dru_bf_sweep.py), 1 if absent"""
    p = os.path.join(run_dir or RUN_DIR, "dru_bf.json")
    return json.load(open(p)).get(org, 1) if os.path.exists(p) else 1


def dru_mm2(org, p1=12, run_dir=None):
    """DRU area per channel: reorder DRU + (when the butterfly DRU is the design) its units and buffers"""
    if dru_ntt(org, run_dir) != "f4b": return DRU_REORDER_MM2
    return DRU_REORDER_MM2 + dru_bf(org, run_dir) * BF_UNIT_MM2 + BF_BUF_MM2(p1)


def dru_mw(org, run_dir=None):
    """DRU power per channel at this card's clock (mW), units counted at full activity (the energy script scales by busy fraction)"""
    s = CLK[org] / 2.25
    if dru_ntt(org, run_dir) != "f4b": return DRU_REORDER_MW * s
    return (DRU_REORDER_MW + dru_bf(org, run_dir) * BF_UNIT_MW * 2.25 / 2.0) * s   # BF_UNIT_MW is quoted at 2 GHz


def spu_mw(org): return SPU_MW * CLK[org] / 2.25


def area_mm2(n, run_dir=None, p1=12):
    """PIM logic area of n SPUs + one DRU per 8 SPUs, the DRU as built for the card that uses the butterfly DRU (the same
    hardware on both cards; the runtime decides whether to use it): Fig. 9's x axis"""
    d = max(dru_mm2(o, p1, run_dir) for o in MACH)
    return n * SPU_MM2 + n / 8 * d


if __name__ == "__main__":
    ch = choice()
    for o in MACH: print(f"{MACH[o]}: PCG^2 NTT = {ch[o]}  (geomean speedup per candidate: " + ", ".join(f"{d} {g:.3f}" for d, g in ch[o + '_geomean'].items())
                         + f"); DRU {dru_mm2(o):.3f} mm^2 / {dru_mw(o):.1f} mW per channel, {dru_bf(o)} butterfly unit(s)")
    print(ESTIMATE_NOTE)
