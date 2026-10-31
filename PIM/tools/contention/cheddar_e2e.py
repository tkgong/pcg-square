#!/usr/bin/env python3
"""Cheddar's NTT kernels in the final accounting (22-multiply co-simulation window, contention included).
Designs (ntt_jobs.py): merge = GPU-NTT merge (HEonGPU's NTT, the submission's baseline NTT); cheddar = Cheddar's two-kernel NTT
on the SMs; f4c = four-step on Cheddar's kernels, transposes on the DRU; f4csm = the same four-step, transposes on the SMs.
  1. NTT lane per machine (gather + kernels, co-run slowdown applied), and whether the DRU traffic stays hidden
     (simulated standalone f4c lane vs its measured SM-only time, against the same ratio for Cheddar).
  2. End-to-end speedup with a Cheddar GPU baseline (DPF + Cheddar NTT, network co-scheduled) for PCG^2 with the
     Cheddar NTT on the SMs and with f4c, next to the current accounting (merge baseline; B200 f4dru, L40S merge).
Usage: cheddar_e2e.py CHEDDAR_WIN22_DIR FINAL_WIN22_DIR OUT_TXT"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
    from lanes import alpha_bw
    from fig8 import BETA
CW, FW, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NEW, OLD = json.load(open(f"{CW}/e2e.json")), json.load(open(f"{FW}/e2e_nom.json"))
MACH = {"l40s": "L40S", "b200": "B200"}; FINAL = {"l40s": "merge", "b200": "f4dru"}
TIERS = (("500us", "alpha = 500 us"), ("fast", "40 Gbps"), ("slow", "400 Mbps"))


def cells(rows, org, des, tier="fast"):
    return {(r["c"], r["t"], r["logN"]): r for r in rows if r["org"] == org and r["design"] == des and r["tier"] == tier}


lane = lambda r: r["ntt_ms"] * r["ntt_slow"]


def nic(org, k, tier):
    n = L_(MACH[org])[k]["n"]
    return (n + 2) * (0.5 if tier == "500us" else alpha_bw(k[0], k[1], BETA[tier]))


def pcg(r, nk):
    return max(r["spu_ms"] * r["spu_slow"], lane(r), nk)


def panel(org, des, rows, tier, gpu_key):
    """per-(c,t) geomean speedups, max of those, geomean over all cells"""
    C = cells(rows, org, des, "fast" if tier == "500us" else tier); per, allv = [], []
    for (c, t) in CFG:
        v = []
        for k, r in C.items():
            if k[:2] != (c, t): continue
            nk = nic(org, k, tier); v.append(max(r[gpu_key], nk) / pcg(r, nk))
        if v: per.append(gm(v)); allv += v
    return max(per), gm(allv), per


L = ["Cheddar's NTT in the final accounting (22-multiply window, contention included). Geomean over the suite's cells.", ""]
L.append("1. NTT lane (gather + kernels, x co-run slowdown), ms geomean; ratios > 1 mean f4c is faster:")
for org in ("b200", "l40s"):
    D = {d: cells(NEW, org, d) for d in ("merge", "cheddar", "f4c", "f4csm")}; ks = sorted(D["f4c"])
    g = {d: gm(lane(D[d][k]) for k in ks) for d in D}
    L.append(f"  {MACH[org]}: merge {g['merge']:.2f} | Cheddar {g['cheddar']:.2f} | f4c (DRU) {g['f4c']:.2f} | f4csm (SM transposes) {g['f4csm']:.2f}"
             f"  ->  merge/f4c {gm(lane(D['merge'][k]) / lane(D['f4c'][k]) for k in ks):.3f}, Cheddar/f4c {gm(lane(D['cheddar'][k]) / lane(D['f4c'][k]) for k in ks):.3f}"
             f" (range {min(lane(D['cheddar'][k]) / lane(D['f4c'][k]) for k in ks):.2f}-{max(lane(D['cheddar'][k]) / lane(D['f4c'][k]) for k in ks):.2f}),"
             f" f4csm/f4c {gm(lane(D['f4csm'][k]) / lane(D['f4c'][k]) for k in ks):.3f}")
    # DRU hidden?  simulated standalone lane / measured SM time (both include the same DPF->NTT gather)
    hid = {d: gm(D[d][k]["ntt_ms"] / D[d][k]["ntt_meas_ms"] for k in ks) for d in ("cheddar", "f4c")}
    L.append(f"        standalone sim lane / measured SM time: Cheddar {hid['cheddar']:.3f}, f4c {hid['f4c']:.3f}"
             f"  (f4c/Cheddar {hid['f4c'] / hid['cheddar']:.3f}: the DRU's 96N B cost the lane that much over its SM kernels)")
    L.append(f"        co-run NTT slowdown: Cheddar {min(D['cheddar'][k]['ntt_slow'] for k in ks):.2f}-{max(D['cheddar'][k]['ntt_slow'] for k in ks):.2f},"
             f" f4c {min(D['f4c'][k]['ntt_slow'] for k in ks):.2f}-{max(D['f4c'][k]['ntt_slow'] for k in ks):.2f};"
             f" SPU slowdown with Cheddar {gm(D['cheddar'][k]['spu_slow'] for k in ks):.3f}, with f4c {gm(D['f4c'][k]['spu_slow'] for k in ks):.3f}")
    L.append("        Cheddar/f4c lane per logN: " + ", ".join(f"2^{lg} {gm(lane(D['cheddar'][k]) / lane(D['f4c'][k]) for k in ks if k[2] == lg):.3f}"
                                                         for lg in sorted({k[2] for k in ks})))
    nb = [k for k in ks if lane(D["cheddar"][k]) >= D["cheddar"][k]["spu_ms"] * D["cheddar"][k]["spu_slow"]]
    L.append(f"        NTT-bound cells (Cheddar lane >= SPU lane): {len(nb)}/{len(ks)}")
    # rerun consistency: merge rows of this run vs the final run
    O = cells(OLD, org, "merge"); dd = [abs(D["merge"][k]["pcg_ms"] / O[k]["pcg_ms"] - 1) for k in ks if k in O]
    L.append(f"        merge rows reproduce the final run: max |diff| {max(dd):.2%} over {len(dd)} cells")
L.append("")
L.append("2. End-to-end speedup, up to (best (c,t)) / geomean:")
L.append(f"  {'':5s} {'network':15s} | {'current (merge base; final NTT)':>32s} | {'Cheddar base, PCG^2 Cheddar NTT':>32s} | {'Cheddar base, PCG^2 f4c (DRU)':>30s} | f4c/Cheddar e2e")
for org in ("b200", "l40s"):
    for tier, lab in TIERS:
        a = panel(org, FINAL[org], OLD, tier, "gpu_ms"); b = panel(org, "cheddar", NEW, tier, "gpu_cheddar_ms"); c = panel(org, "f4c", NEW, tier, "gpu_cheddar_ms")
        Cc, Cf = cells(NEW, org, "cheddar", "fast" if tier == "500us" else tier), cells(NEW, org, "f4c", "fast" if tier == "500us" else tier)
        r = [pcg(Cc[k], nic(org, k, tier)) / pcg(Cf[k], nic(org, k, tier)) for k in Cf]
        L.append(f"  {MACH[org]:5s} {lab:15s} | {a[0]:15.2f} / {a[1]:5.2f}          | {b[0]:15.2f} / {b[1]:5.2f}          | {c[0]:13.2f} / {c[1]:5.2f}          | {gm(r):.3f} (range {min(r):.2f}-{max(r):.2f})")
L.append("")
L.append("GPU baseline geomean ms (DPF + NTT), merge vs Cheddar NTT:")
for org in ("b200", "l40s"):
    C = cells(NEW, org, "cheddar")
    L.append(f"  {MACH[org]}: {gm(r['gpu_ms'] for r in C.values()):.1f} -> {gm(r['gpu_cheddar_ms'] for r in C.values()):.1f}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
