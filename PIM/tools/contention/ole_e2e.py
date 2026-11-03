#!/usr/bin/env python3
"""The PCG-output NTT (ole: z = INTT(sum_ij a^_i a^_j NTT(g_ij)) on Cheddar's kernels, c^2 forward transforms + 1 inverse per
output instead of 2c^2 multiplies) in the final accounting (co-simulation window, contention included), next to Cheddar's
two-kernel NTT run the old way (cheddar) and GPU-NTT merge.  Designs from ntt_jobs.py; ole_t reads the g_ij in the
column-contiguous layout the SPUs write.
  1. NTT lane per machine (kernels x co-run slowdown; ole has no gather pass), per logN, and the co-run slowdowns.
  2. End-to-end speedup of PCG^2 (SPU DPF + the design's NTT on the SMs) over the GPU baseline with the SAME NTT algorithm
     (GPU DPF + ole, network co-scheduled), and over the baseline with Cheddar's old algorithm; best (c,t) / geomean.
Usage: ole_e2e.py WIN_DIR[,WIN_DIR2] OUT_TXT"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
    from lanes import alpha_bw
    from fig8 import BETA
W, OUT = sys.argv[1], sys.argv[2]            # W: one window dir, or several comma-separated (one per machine)
ROWS = [r for w in W.split(",") for r in json.load(open(f"{w}/e2e.json"))]
MACH = {"l40s": "L40S", "b200": "B200"}
TIERS = (("500us", "alpha = 500 us"), ("fast", "40 Gbps"), ("slow", "400 Mbps"))
DES = [d for d in ("merge", "cheddar", "cheddar_t", "ole", "ole_t", "olep", "olepb", "oleu", "olepu") if any(r["design"] == d for r in ROWS)]


def cells(org, des, tier="fast"):
    return {(r["c"], r["t"], r["logN"]): r for r in ROWS if r["org"] == org and r["design"] == des and r["tier"] == tier}


lane = lambda r: r["ntt_ms"] * r["ntt_slow"]


def nic(org, k, tier):
    n = L_(MACH[org])[k]["n"]
    return (n + 2) * (0.5 if tier == "500us" else alpha_bw(k[0], k[1], BETA[tier]))


def pcg(r, nk):
    return max(r["spu_ms"] * r["spu_slow"], lane(r), nk)


def panel(org, des, tier, gpu_key):
    """per-(c,t) geomean speedups -> (max of those, geomean over all cells, per-(c,t) list)"""
    C = cells(org, des, "fast" if tier == "500us" else tier); per, allv = [], []
    for (c, t) in CFG:
        v = []
        for k, r in C.items():
            if k[:2] != (c, t) or r.get(gpu_key) is None or r[gpu_key] != r[gpu_key]: continue
            nk = nic(org, k, tier); v.append(max(r[gpu_key], nk) / pcg(r, nk))
        if v: per.append(gm(v)); allv += v
    return (max(per), gm(allv), per) if allv else (float("nan"), float("nan"), [])


L = ["PCG-output NTT (ole) in the final accounting (co-simulation window, contention included). Geomean over the suite's cells.", ""]
L.append("1. NTT lane (kernels x co-run slowdown; cheddar/merge include the DPF->NTT gather, ole reads the SPU output directly), ms geomean:")
for org in ("b200", "l40s"):
    D = {d: cells(org, d) for d in DES}
    ks = sorted(set.intersection(*(set(D[d]) for d in DES)))
    if not ks: L.append(f"  {MACH[org]}: no common cells"); continue
    g = {d: gm(lane(D[d][k]) for k in ks) for d in DES}
    L.append("  " + MACH[org] + ": " + " | ".join(f"{d} {g[d]:.2f}" for d in DES))
    for d in DES:
        if d == "cheddar": continue
        L.append(f"      Cheddar/{d}: {gm(lane(D['cheddar'][k]) / lane(D[d][k]) for k in ks):.3f} (range {min(lane(D['cheddar'][k]) / lane(D[d][k]) for k in ks):.2f}-{max(lane(D['cheddar'][k]) / lane(D[d][k]) for k in ks):.2f})"
                 + "; per logN " + ", ".join(f"2^{lg} {gm(lane(D['cheddar'][k]) / lane(D[d][k]) for k in ks if k[2] == lg):.3f}" for lg in sorted({k[2] for k in ks})))
    for d in DES:
        sl = [D[d][k]["ntt_slow"] for k in ks]; ss = [D[d][k]["spu_slow"] for k in ks]; er = gm(D[d][k]["ntt_ms"] / D[d][k]["ntt_meas_ms"] for k in ks)
        nb = sum(lane(D[d][k]) >= D[d][k]["spu_ms"] * D[d][k]["spu_slow"] for k in ks)
        L.append(f"      {d:8s}: co-run NTT slowdown {min(sl):.3f}-{max(sl):.3f} (geo {gm(sl):.3f}), SPU slowdown beside it {min(ss):.3f}-{max(ss):.3f} (geo {gm(ss):.3f}),"
                 f" standalone sim lane / measured {er:.3f}, NTT-bound cells {nb}/{len(ks)}")
COLS = [(d, k) for d, k in (("cheddar", "gpu_cheddar_ms"), ("cheddar_t", "gpu_cheddar_ms"), ("ole", "gpu_cheddar_ms"), ("ole", "gpu_ole_ms"), ("ole_t", "gpu_ole_ms"), ("olep", "gpu_cheddar_ms"),
                           ("olep", "gpu_olep_ms"), ("olepb", "gpu_cheddar_ms"), ("olepb", "gpu_olep_ms"), ("oleu", "gpu_cheddar_ms"), ("oleu", "gpu_ole_ms"),
                           ("olepu", "gpu_cheddar_ms"), ("olepu", "gpu_olep_ms")) if d in DES]
COLS_H = [f"base {'Cheddar' if k == 'gpu_cheddar_ms' else k[4:-3]}, PCG^2 {d}" for d, k in COLS]
L += ["", "2. End-to-end speedup, best (c,t) / geomean; PCG^2 = SPU DPF + the design's NTT on the SMs; baselines = GPU DPF + NTT, network co-scheduled:",
      "   (oleu/olepu: the channel-unit design, SM = column passes only, unit = row pass + a^ product + accumulate; same-algorithm baseline = ole / olep on the GPU)",
      "        network         | " + " | ".join(f"{h:^22s}" for h in COLS_H)]
for org in ("b200", "l40s"):
    for tier, name in TIERS:
        f = lambda p: f"{p[0]:6.2f} / {p[1]:5.2f}"
        L.append(f"  {MACH[org]}  {name:15s} | " + " | ".join(f"{f(panel(org, d, tier, k)):^22s}" for d, k in COLS))
L.append("")
L.append("GPU baseline geomean ms (DPF + NTT), Cheddar old algorithm -> ole:")
for org in ("b200", "l40s"):
    C = cells(org, "ole")
    if C: L.append(f"  {MACH[org]}: {gm(r['gpu_cheddar_ms'] for r in C.values()):.1f} -> {gm(r['gpu_ole_ms'] for r in C.values()):.1f}"
                   f"  (NTT share of the old baseline: {gm(r['gpu_cheddar_ms'] / (r['gpu_cheddar_ms'] - r['ntt_meas_ms']) for r in cells(org, 'cheddar').values()) - 1:.1%} geomean)")
txt = "\n".join(L) + "\n"; open(OUT, "w").write(txt); print(txt)
