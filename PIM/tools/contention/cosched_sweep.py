#!/usr/bin/env python3
"""Network co-scheduling ablation: how much hiding the network buys the GPU baseline and PCG^2 as a function of
the network bandwidth, from the final co-sim rows (e2e_nom.json; the network is a separate lane, so no re-simulation).
Per cell:  T_c(GPU) = DPF (two-pass H') + merge NTT;  T_c(PCG^2) = max(SPU lane x slowdown, NTT lane x slowdown);
           T_n(beta) = (n + 2) exchanges x alpha(c, t, beta)
           serial = T_c + T_n;  co-scheduled = max(T_c, T_n);  gain = serial / co-scheduled  (<= 2, peaks at T_c = T_n)
Writes the table (geomean over the suite) and the sweep rows.    Usage: cosched_sweep.py WIN22_DIR OUT_TXT [OUT_JSON]"""
import io, contextlib, json, os, sys
from math import exp, log, log10
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
    from lanes import alpha_bw
W, OUT = sys.argv[1], sys.argv[2]; OUTJ = sys.argv[3] if len(sys.argv) > 3 else None
DES = {"L40S": ("l40s", "merge"), "B200": ("b200", "f4dru")}
R = json.load(open(os.path.join(W, "e2e_nom.json")))
LN = {m: L_(m) for m in DES}
BETAS = [100e9, 40e9, 10e9, 4e9, 1e9, 400e6, 100e6, 40e6, 10e6]
GRID = [10 ** (x / 8) for x in range(8 * 7, 8 * 11 + 1)]        # 10 Mbps .. 100 Gbps, 8 points per decade


def fmt_b(b): return f"{b/1e9:g} Gbps" if b >= 1e9 else f"{b/1e6:g} Mbps"


def cells(mach):
    org, des = DES[mach]
    out = []
    for r in R:
        if r["org"] != org or r["design"] != des or r["tier"] != "fast": continue
        k = (r["c"], r["t"], r["logN"])
        out.append(dict(k=k, n=LN[mach][k]["n"], tg=r["gpu_ms"], tp=max(r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"])))
    return out


def evaluate(mach, beta):
    """per-cell (gain_gpu, gain_pcg, speedup both co-scheduled, neither, PCG^2 only, net-bound flags)"""
    v = []
    for cl in cells(mach):
        c, t, lg = cl["k"]; tn = (cl["n"] + 2) * alpha_bw(c, t, beta)
        gs, gc = cl["tg"] + tn, max(cl["tg"], tn); ps, pc = cl["tp"] + tn, max(cl["tp"], tn)
        v.append(dict(k=cl["k"], gain_gpu=gs / gc, gain_pcg=ps / pc, sp_both=gc / pc, sp_none=gs / ps, sp_pcg_only=gs / pc,
                      nb_gpu=tn > cl["tg"], nb_pcg=tn > cl["tp"], fn_gpu=tn / gs, fn_pcg=tn / ps))
    return v


L = ["Network co-scheduling ablation (final accounting, contention included). gain = serial / co-scheduled, geomean over the suite;",
     "net share = network time / serial time; net-bound = cells where the network is the longest lane.",
     "speedup: both = GPU and PCG^2 co-scheduled (headline), none = neither, PCG^2 only = the submission's accounting.", ""]
rows = []
for mach in ("L40S", "B200"):
    L.append(f"{mach}: {'bandwidth':>10s} | {'GPU gain':>8s} {'net share':>9s} {'net-bound':>9s} | {'PCG2 gain':>9s} {'net share':>9s} {'net-bound':>9s} | {'T_c GPU':>8s} {'T_c PCG2':>8s} {'T_net':>8s} | speedup both / none / PCG2-only")
    for b in BETAS:
        v = evaluate(mach, b); n = len(v)
        tg, tp = gm(c["tg"] for c in cells(mach)), gm(c["tp"] for c in cells(mach))
        tn = gm((c["n"] + 2) * alpha_bw(c["k"][0], c["k"][1], b) for c in cells(mach))
        row = dict(mach=mach, beta=b, gain_gpu=gm(x["gain_gpu"] for x in v), gain_pcg=gm(x["gain_pcg"] for x in v),
                   share_gpu=sum(x["fn_gpu"] for x in v) / n, share_pcg=sum(x["fn_pcg"] for x in v) / n,
                   nb_gpu=sum(x["nb_gpu"] for x in v), nb_pcg=sum(x["nb_pcg"] for x in v), ncell=n,
                   sp_both=gm(x["sp_both"] for x in v), sp_none=gm(x["sp_none"] for x in v), sp_pcg_only=gm(x["sp_pcg_only"] for x in v), tg=tg, tp=tp, tn=tn)
        rows.append(row)
        L.append(f"      {fmt_b(b):>10s} | {row['gain_gpu']:8.3f} {row['share_gpu']:9.1%} {row['nb_gpu']:5d}/{n:<3d} | {row['gain_pcg']:9.3f} {row['share_pcg']:9.1%} {row['nb_pcg']:5d}/{n:<3d} | {tg:8.1f} {tp:8.1f} {tn:8.1f} | {row['sp_both']:.2f} / {row['sp_none']:.2f} / {row['sp_pcg_only']:.2f}")
    L.append("")
# per-(c,t) at the two Fig. 8 tiers
L.append("Per (c,t) at the Fig. 8 tiers: GPU gain / PCG2 gain (geomean over logN)")
for mach in ("L40S", "B200"):
    for b, lab in ((40e9, "40 Gbps"), (400e6, "400 Mbps")):
        v = evaluate(mach, b)
        parts = []
        for (c, t) in CFG:
            vv = [x for x in v if x["k"][0] == c and x["k"][1] == t]
            if vv: parts.append(f"({c},{t}) {gm(x['gain_gpu'] for x in vv):.2f}/{gm(x['gain_pcg'] for x in vv):.2f}")
        L.append(f"  {mach} {lab:8s}: " + "  ".join(parts))
# where each system's gain peaks (T_c = T_n): the bandwidth at which the geomean network time equals the geomean compute time
L.append("\nBandwidth at which the network time equals the compute time (gain peak, geomean cell):")
for mach in ("L40S", "B200"):
    cs = cells(mach); tg, tp = gm(c["tg"] for c in cs), gm(c["tp"] for c in cs)
    tn1 = gm((c["n"] + 2) * alpha_bw(c["k"][0], c["k"][1], 1e9) for c in cs)      # T_net at 1 Gbps; scales as 1/beta
    L.append(f"  {mach}: GPU baseline {fmt_b(1e9 * tn1 / tg)}  |  PCG^2 {fmt_b(1e9 * tn1 / tp)}   (T_c GPU {tg:.0f} ms, T_c PCG^2 {tp:.0f} ms)")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
if OUTJ:
    grid = []
    for mach in ("L40S", "B200"):
        for b in GRID:
            v = evaluate(mach, b)
            grid.append(dict(mach=mach, beta=b, gain_gpu=gm(x["gain_gpu"] for x in v), gain_pcg=gm(x["gain_pcg"] for x in v),
                             sp_both=gm(x["sp_both"] for x in v), sp_none=gm(x["sp_none"] for x in v), sp_pcg_only=gm(x["sp_pcg_only"] for x in v)))
    json.dump(dict(table=rows, grid=grid), open(OUTJ, "w"), indent=1)
