#!/usr/bin/env python3
"""The NTT designs in the final accounting (22-multiply co-simulation window, contention included; final_design.py):
merge = GPU-NTT merge; cheddar = Cheddar's two-kernel NTT on the SMs (the GPU baseline's NTT); f4c = four-step on Cheddar's
kernels, transposes on the reorder DRU; f4csm = the same four-step, transposes on the SMs; f4b = butterfly DRU (column pass
inside the DRU, SMs run the row kernel + pointwise).
  1. NTT lane per card (kernels x co-run slowdown), ms geomean, and each DRU design against Cheddar; whether the DRU
     traffic stays hidden (simulated standalone lane vs the design's measured SM-only time); co-run slowdowns; per logN.
  2. End-to-end speedup over the GPU baseline (DPF + Cheddar NTT, network co-scheduled) for PCG^2 with each NTT design.
Usage: ntt_designs.py RUN_DIR OUT_TXT"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e")); sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
    from lanes import alpha_bw
    from fig8 import BETA
import final_design as FD
W, OUT = sys.argv[1], sys.argv[2]
R = json.load(open(f"{W}/e2e_nom.json"))
TIERS = (("500us", "alpha = 500 us"), ("fast", "40 Gbps"), ("slow", "400 Mbps"))
ALL = ("merge", "cheddar", "f4c", "f4csm", "f4b")
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]


def nic(org, k, tier):
    n = L_(FD.MACH[org])[k]["n"]
    return (n + 2) * (0.5 if tier == "500us" else alpha_bw(k[0], k[1], BETA[tier]))


def pcg(r, nk): return max(r["spu_ms"] * r["spu_slow"], lane(r), nk)


def panel(org, des, tier):
    """per-(c,t) geomean speedups -> (max of those, geomean over all cells)"""
    C = FD.cells(R, org, des, "fast" if tier == "500us" else tier); per, allv = [], []
    for (c, t) in CFG:
        v = [max(r[FD.GPU_KEY], nic(org, k, tier)) / pcg(r, nic(org, k, tier)) for k, r in C.items() if k[:2] == (c, t)]
        if v: per.append(gm(v)); allv += v
    return (max(per), gm(allv)) if per else (float("nan"), float("nan"))


L = [__doc__.split("Usage")[0].strip(), "", "Runtime choice per card: " + ", ".join(f"{FD.MACH[o]} {FD.des(o, W)}" for o in FD.MACH), ""]
L.append("1. NTT lane (kernels x co-run slowdown), ms geomean over the cells every design has; ratios > 1 mean the DRU design is faster:")
for org in ("b200", "l40s"):
    D = {d: FD.cells(R, org, d) for d in ALL}; D = {d: c for d, c in D.items() if c}
    ks = sorted(set.intersection(*(set(c) for c in D.values())))
    g = {d: gm(lane(D[d][k]) for k in ks) for d in D}
    L.append(f"  {FD.MACH[org]} ({len(ks)} cells): " + " | ".join(f"{d} {g[d]:.2f}" for d in D))
    for d in ("f4c", "f4b"):
        if d not in D: L.append(f"        {d}: not simulated on this card (no measured SM lane)"); continue
        rr = [lane(D["cheddar"][k]) / lane(D[d][k]) for k in ks]
        L.append(f"        Cheddar/{d}: {gm(rr):.3f} (range {min(rr):.2f}-{max(rr):.2f}); per logN " + ", ".join(f"2^{lg} {gm(lane(D['cheddar'][k]) / lane(D[d][k]) for k in ks if k[2] == lg):.3f}" for lg in sorted({k[2] for k in ks}))
                 + (f"; merge/{d} {gm(lane(D['merge'][k]) / lane(D[d][k]) for k in ks):.3f}" if "merge" in D else ""))
    hid = {d: gm(D[d][k]["ntt_ms"] / D[d][k]["ntt_meas_ms"] for k in ks) for d in D if d != "merge"}
    L.append("        standalone sim lane / measured SM time: " + ", ".join(f"{d} {hid[d]:.3f}" for d in hid) + "  (> 1: the DRU pipe or the gather bound the lane)")
    L.append("        co-run NTT slowdown: " + ", ".join(f"{d} {min(D[d][k]['ntt_slow'] for k in ks):.2f}-{max(D[d][k]['ntt_slow'] for k in ks):.2f}" for d in D)
             + "; SPU slowdown beside: " + ", ".join(f"{d} {gm(D[d][k]['spu_slow'] for k in ks):.3f}" for d in D))
    nb = [k for k in ks if lane(D["cheddar"][k]) >= D["cheddar"][k]["spu_ms"] * D["cheddar"][k]["spu_slow"]]
    L.append(f"        NTT-bound cells (Cheddar lane >= SPU lane): {len(nb)}/{len(ks)}")
L.append("")
L.append("2. End-to-end speedup over the GPU baseline (DPF + Cheddar NTT, network co-scheduled), up to (best (c,t)) / geomean, PCG^2 with each NTT:")
L.append(f"  {'':5s} {'network':15s} | " + " | ".join(f"{d:>14s}" for d in ALL) + " | f4c/Cheddar e2e | f4b/Cheddar e2e")
for org in ("b200", "l40s"):
    for tier, lab in TIERS:
        cols = [panel(org, d, tier) for d in ALL]
        Cc = FD.cells(R, org, "cheddar", "fast" if tier == "500us" else tier); rat = []
        for d in ("f4c", "f4b"):
            Cd = FD.cells(R, org, d, "fast" if tier == "500us" else tier)
            r = [pcg(Cc[k], nic(org, k, tier)) / pcg(Cd[k], nic(org, k, tier)) for k in Cd if k in Cc]
            rat.append(f"{gm(r):.3f} ({min(r):.2f}-{max(r):.2f})" if r else "-")
        L.append(f"  {FD.MACH[org]:5s} {lab:15s} | " + " | ".join(f"{u:6.2f} / {g:5.2f}" if u == u else f"{'-':>14s}" for u, g in cols) + " | " + " | ".join(f"{x:>15s}" for x in rat))
L.append("")
L.append("GPU baseline geomean ms (DPF + NTT), merge NTT vs Cheddar NTT:")
for org in ("b200", "l40s"):
    C = FD.cells(R, org, "cheddar")
    L.append(f"  {FD.MACH[org]}: {gm(r['gpu_ms'] for r in C.values()):.1f} -> {gm(r['gpu_cheddar_ms'] for r in C.values()):.1f}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
