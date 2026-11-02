#!/usr/bin/env python3
"""Table IV with one NTT family on both cards: GPU baseline = DPF + HEonGPU merge NTT (network overlapped);
"+SPU" rows = four-step on GPU-NTT kernels with the transposes as shared-memory tile transposes on the SMs (f4sm);
"+DRU" rows = the same kernels with the transposes on the DRU (f4dru). Co-scheduled rows include the co-simulated contention.
Usage: table4_sm.py FINAL_WINDOW_DIR OUT_TXT"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, gm
    from lanes import alpha_bw
    from fig8 import BETA
W, OUT = sys.argv[1], sys.argv[2]
LN = {"l40s": L_("L40S"), "b200": L_("B200")}
def cells(path, org, des): return {(r["c"], r["t"], r["logN"]): r for r in json.load(open(path)) if r["org"] == org and r["design"] == des and r["tier"] == "fast"}
nic_of = lambda org, k, tier: (LN[org][k]["n"] + 2) * (0.5 if tier == "a500" else alpha_bw(k[0], k[1], BETA[tier]))
ser = lambda r, nic: r["spu_ms"] + r["ntt_ms"] + nic
cos = lambda r, nic: max(r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"], nic)
L = [__doc__.split("Usage")[0].strip(), ""]
for tier, lab in (("a500", "alpha = 500 us"), ("fast", "40 Gbps"), ("slow", "400 Mbps")):
    for org in ("l40s", "b200"):
        M = cells(f"{W}/win22/e2e_nom.json", org, "merge"); F = cells(f"{W}/win22/e2e_nom.json", org, "f4dru"); S = cells(f"{W}/win22/e2e_f4sm.json", org, "f4sm")
        ks = [k for k in M if k in S and k in F]; nic = {k: nic_of(org, k, tier) for k in ks}
        base = gm(max(M[k]["gpu_ms"], nic[k]) for k in ks)
        rows = [("GPU baseline", base), ("+SPU (SM transpose)", gm(ser(S[k], nic[k]) for k in ks)), ("+SPU+DRU", gm(ser(F[k], nic[k]) for k in ks)),
                ("+SPU+co-schedule (SM transpose)", gm(cos(S[k], nic[k]) for k in ks)), ("full (+DRU, co-scheduled)", gm(cos(F[k], nic[k]) for k in ks))]
        inc = [None, rows[0][1] / rows[1][1], rows[1][1] / rows[2][1], rows[1][1] / rows[3][1], rows[3][1] / rows[4][1]]
        L.append(f"{org.upper()} {lab} ({len(ks)} cells): " + " | ".join(f"{n} {v:.1f} ms {base/v:.2f}x" + (f" inc {i:.3f}" if i else "") for (n, v), i in zip(rows, inc)))
    L.append("")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
