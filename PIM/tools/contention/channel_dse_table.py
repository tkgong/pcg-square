#!/usr/bin/env python3
"""Channel-count DSE table (Fig. 9(b) data): SPU array fixed per card, per-channel bandwidth fixed, one DRU per channel.
Per channel count: NTT lane (co-simulated, incl. contention) of the butterfly DRU (f4b) and of the reorder DRU (f4c), vs
the real card's two SM-only NTTs (Cheddar, GPU-NTT merge); and end-to-end runtime (alpha = 500 us) vs the real card with
Cheddar's NTT.  Geomean over the suite.  Inputs: <RUN_DIR>/e2e_nom.json + <RUN_DIR>/channel_sweep.json.
Usage: channel_dse_table.py RUN_DIR [OUT]"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e")); sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_
import final_design as FD
W = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
load = lambda p: json.load(open(p)) if os.path.exists(p) else []
R = load(f"{W}/e2e_nom.json") + load(f"{W}/channel_sweep.json")
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]
A500 = 500
CARD = {"B200": dict(org="b200", native=256, chs=[8, 16, 32, 48, 96, 128, 192, 256, 512], bw_ch=31.25),
        "L40S": dict(org="l40s", native=24, chs=[8, 16, 24, 48, 96, 128, 192, 256, 512], bw_ch=36.0)}
lines = []
for mach, c in CARD.items():
    LN = L_(mach); nat = FD.cells(R, c["org"], FD.SM_NTT); SPU = {k: r["spu_ms"] * r["spu_slow"] for k, r in nat.items()}
    rt = lambda d: FD.gm(max(SPU[k], lane(r), (LN[k]["n"] + 2) * A500 / 1000.0) for k, r in d.items() if k in SPU)
    ch_, mg = nat, FD.cells(R, c["org"], "merge")
    t_ch, t_mg, e_ch = FD.gm(lane(ch_[k]) for k in SPU), FD.gm(lane(mg[k]) for k in SPU if k in mg), rt(ch_)
    lines += [f"{mach}: {len(SPU)} cells, SPU lane geomean {FD.gm(SPU.values()):.1f} ms; real card ({c['native']} ch): NTT lane Cheddar {t_ch:.1f} ms, merge {t_mg:.1f} ms; e2e with Cheddar {e_ch:.1f} ms",
              f"  ch   BW(TB/s) | f4b lane (ms)  vs Cheddar  vs merge | f4b e2e (ms)  vs Cheddar e2e | f4c lane (ms)  vs Cheddar | f4c e2e (ms)  vs Cheddar e2e"]
    for ch in c["chs"]:
        org = c["org"] if ch == c["native"] else f"{c['org']}_ch{ch}"
        B, C4 = FD.cells(R, org, "f4b"), FD.cells(R, org, "f4c")
        if not B and not C4: continue
        f = lambda d, fn: fn(d) if d else float("nan")
        lines.append(f"  {ch:3d}  {ch * c['bw_ch'] / 1e3:7.2f} | {f(B, lambda d: FD.gm(lane(d[k]) for k in SPU if k in d)):12.1f}  {f(B, lambda d: t_ch / FD.gm(lane(d[k]) for k in SPU if k in d)):10.3f}  {f(B, lambda d: t_mg / FD.gm(lane(d[k]) for k in SPU if k in d)):8.3f} |"
                     f" {f(B, rt):11.1f}  {f(B, lambda d: e_ch / rt(d)):14.3f} | {f(C4, lambda d: FD.gm(lane(d[k]) for k in SPU if k in d)):13.1f}  {f(C4, lambda d: t_ch / FD.gm(lane(d[k]) for k in SPU if k in d)):10.3f} | {f(C4, rt):11.1f}  {f(C4, lambda d: e_ch / rt(d)):14.3f}")
    lines.append("")
s = "\n".join(lines); print(s)
if OUT: open(OUT, "w").write(s + "\n")
