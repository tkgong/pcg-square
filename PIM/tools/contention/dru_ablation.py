#!/usr/bin/env python3
"""DRU ablation inside the final design, per cell (replaces the inline script that wrote the first dru_ablation.txt):
PCG^2 with Cheddar's NTT on the SMs (cheddar) vs the reorder DRU (f4c: six transposes on the DRU, SM kernels unchanged)
vs the butterfly DRU (f4b: column pass inside the DRU, SMs run the row kernel + pointwise). Contention included, 40 Gbps.
Per cell: NTT lane (ms, x co-run slowdown), SPU lane, which lane bounds the cell, and the end-to-end gain of each DRU over
the SM NTT; geomeans over all cells and over the NTT-bound cells; the Table IV DRU increments at the three network tiers.
Usage: dru_ablation.py RUN_DIR OUT_TXT"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e")); sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, gm
    from lanes import alpha_bw
    from fig8 import BETA
import final_design as FD
W, OUT = sys.argv[1], sys.argv[2]
R = json.load(open(f"{W}/e2e_nom.json")); LN = {"l40s": L_("L40S"), "b200": L_("B200")}
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]
L = [__doc__.split("Usage")[0].strip(), ""]
for org in ("b200", "l40s"):
    S = FD.cells(R, org, FD.SM_NTT); D = {d: FD.cells(R, org, d) for d in ("f4c", "f4b")}; D = {d: c for d, c in D.items() if c}
    ks = sorted(k for k in S if all(k in c for c in D.values()))
    if not ks: L.append(f"{FD.MACH[org]}: no DRU design simulated\n"); continue
    L.append(f"{FD.MACH[org]}, 40 Gbps, per cell: SM NTT lane x slow | " + " | ".join(f"{d} lane x slow" for d in D) + " | SPU lane | bound (SM NTT) | " + " | ".join(f"gain {d}" for d in D))
    gains = {d: [] for d in D}; nb_gains = {d: [] for d in D}
    for k in ks:
        s = S[k]; spu = s["spu_ms"] * s["spu_slow"]; bound = "NTT" if lane(s) >= spu else "SPU"
        g = {d: s["pcg_ms"] / D[d][k]["pcg_ms"] for d in D}
        for d in D:
            gains[d].append(g[d])
            if bound == "NTT": nb_gains[d].append(g[d])
        L.append(f"  ({k[0]},{k[1]:3d}) 2^{k[2]}: {s['ntt_ms']:7.1f} x{s['ntt_slow']:.3f} | " + " | ".join(f"{D[d][k]['ntt_ms']:7.1f} x{D[d][k]['ntt_slow']:.3f}" for d in D)
                 + f" | {spu:7.1f} | {bound} | " + " | ".join(f"{g[d]:.3f}" for d in D))
    for d in D:
        L.append(f"  {d}: geomean {gm(gains[d]):.3f} (range {min(gains[d]):.2f}-{max(gains[d]):.2f}); NTT-bound cells {len(nb_gains[d])}/{len(ks)}: {gm(nb_gains[d]) if nb_gains[d] else float('nan'):.3f}")
        for tier, lab in (("a500", "alpha = 500 us"), ("fast", "40 Gbps"), ("slow", "400 Mbps")):
            nic = {k: (LN[org][k]["n"] + 2) * (0.5 if tier == "a500" else alpha_bw(k[0], k[1], BETA[tier])) for k in ks}
            ser = gm((S[k]["spu_ms"] + S[k]["ntt_ms"] + nic[k]) / (D[d][k]["spu_ms"] + D[d][k]["ntt_ms"] + nic[k]) for k in ks)
            co = gm(max(S[k]["spu_ms"] * S[k]["spu_slow"], lane(S[k]), nic[k]) / max(D[d][k]["spu_ms"] * D[d][k]["spu_slow"], lane(D[d][k]), nic[k]) for k in ks)
            L.append(f"      Table IV DRU increment ({d}), {lab:14s}: serial {ser:.3f}, co-scheduled {co:.3f}")
    L.append("")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
