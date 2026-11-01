#!/usr/bin/env python3
"""Table IV (component ablation), accounting of final_design.py: GPU baseline = DPF + Cheddar NTT, network overlapped;
"+SPU" rows = PCG^2 with Cheddar's NTT on the SMs (SM_NTT); "+DRU" rows = the DRU design the runtime picked on that card
(butterfly DRU f4b; "DRU not used" where it picked an SM NTT). Co-scheduled rows include the co-simulated contention.
Also the two DRU variants side by side where both were simulated: f4c (reorder DRU, transposes only) and f4b.
Usage: table4.py RUN_DIR OUT_TXT"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e")); sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, gm
    from lanes import alpha_bw
    from fig8 import BETA
import final_design as FD
W, OUT = sys.argv[1], sys.argv[2]
LN = {"l40s": L_("L40S"), "b200": L_("B200")}
R = json.load(open(f"{W}/e2e_nom.json"))
nic_of = lambda org, k, tier: (LN[org][k]["n"] + 2) * (0.5 if tier == "a500" else alpha_bw(k[0], k[1], BETA[tier]))
ser = lambda r, nic: r["spu_ms"] + r["ntt_ms"] + nic
cos = lambda r, nic: max(r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"], nic)
L = [__doc__.split("Usage")[0].strip(), "", "PCG^2 NTT per card: " + ", ".join(f"{FD.MACH[o]} {FD.des(o, W)}" for o in FD.MACH), ""]
for tier, lab in (("a500", "alpha = 500 us"), ("fast", "40 Gbps"), ("slow", "400 Mbps")):
    for org in ("l40s", "b200"):
        A = FD.cells(R, org, FD.SM_NTT); D = FD.cells(R, org, FD.dru_ntt(org, W))
        ks = [k for k in A if k in D]; nic = {k: nic_of(org, k, tier) for k in ks}
        base = gm(max(A[k][FD.GPU_KEY], nic[k]) for k in ks)
        rows = [("GPU baseline", base), ("+SPU", gm(ser(A[k], nic[k]) for k in ks)), ("+SPU+DRU", gm(ser(D[k], nic[k]) for k in ks)),
                ("+SPU+co-schedule", gm(cos(A[k], nic[k]) for k in ks)), ("full", gm(cos(D[k], nic[k]) for k in ks))]
        inc = [None, rows[0][1] / rows[1][1], rows[1][1] / rows[2][1], rows[1][1] / rows[3][1], rows[3][1] / rows[4][1]]
        extra = ""
        for d, name in (("f4c", "reorder DRU (f4c)"), ("f4b", "butterfly DRU (f4b)")):
            X = FD.cells(R, org, d)
            if X and all(k in X for k in ks) and d != FD.dru_ntt(org, W):
                extra += f" || with the {name} instead: +SPU+DRU {gm(ser(X[k], nic[k]) for k in ks):.1f} ms, full {gm(cos(X[k], nic[k]) for k in ks):.1f} ms ({base / gm(cos(X[k], nic[k]) for k in ks):.2f}x)"
        L.append(f"{org.upper()} {lab} ({len(ks)} cells; +DRU = {FD.dru_ntt(org, W)}): " + " | ".join(f"{n} {v:.1f} ms {base/v:.2f}x" + (f" inc {i:.3f}" if i else "") for (n, v), i in zip(rows, inc)) + extra)
    L.append("")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
