#!/usr/bin/env python3
"""Reviewer B: the DRU's gain as a function of the channel count (card variants <org>_ch<C>: the card's DRAM timing,
per-channel bandwidth, GPU kernels and SPU design fixed; 8 SPUs + 1 DRU per channel; device bytes spread over C channels).
Per channel count C and card: NTT lane and end-to-end gain of each DRU design over Cheddar's NTT on the SMs (SM_NTT):
f4c = reorder DRU (transposes only), f4b = butterfly DRU (column pass in the DRU); the runtime enables a DRU only where
its lane gain is > 1; the DRU's bus share (DRU columns / NTT lane) and which lane bounds the cells.
Inputs: <RUN_DIR>/e2e_nom.json (native channel count) + <RUN_DIR>/channel_sweep.json (the _ch variants, both cards).
Usage: channel_dse.py RUN_DIR OUT_TXT"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import final_design as FD
W, OUT = sys.argv[1], sys.argv[2]
load = lambda p: json.load(open(p)) if os.path.exists(p) else []
R = load(f"{W}/e2e_nom.json") + load(f"{W}/channel_sweep.json")
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]; spul = lambda r: r["spu_ms"] * r["spu_slow"]
CARD = {"b200": dict(native=256, chs=(8, 16, 32, 48, 96, 128, 192, 256, 512), bw_ch=8.2e12 / 256), "l40s": dict(native=24, chs=(8, 16, 24, 48, 96, 128, 192, 256, 512), bw_ch=0.864e12 / 24)}
L = [__doc__.split("Usage")[0].strip(), "", FD.ESTIMATE_NOTE if False else "Gains: geomean over the suite's cells (NTT-bound subset, by the SM NTT, in brackets); > 1 = the DRU design is faster."]
for org, c in CARD.items():
    L += ["", f"{FD.MACH[org]} (native {c['native']} channels; runtime choice at native: {FD.des(org, W)})",
          f"{'ch':>4s} {'SPUs':>5s} {'agg BW':>7s} | {'Cheddar lane':>12s} | {'f4c lane gain':>13s} {'e2e [NTT-bound]':>16s} {'on?':>4s} | {'f4b lane gain':>13s} {'e2e [NTT-bound]':>16s} {'on?':>4s} | NTT-bound"]
    for ch in c["chs"]:
        o = org if ch == c["native"] else f"{org}_ch{ch}"
        S = FD.cells(R, o, FD.SM_NTT)
        if not S: continue
        D = {d: FD.cells(R, o, d) for d in ("f4c", "f4b")}
        ks = sorted(S); nb = [k for k in ks if lane(S[k]) >= spul(S[k])]
        def g(d, keys, f):
            keys = [k for k in keys if k in D[d]]
            return FD.gm(f(S[k]) / f(D[d][k]) for k in keys) if keys else float("nan")
        row = f"{ch:4d} {8*ch:5d} {c['bw_ch']*ch/1e12:6.2f}T | {FD.gm(lane(S[k]) for k in ks):12.1f} |"
        for d in ("f4c", "f4b"):
            gl = g(d, ks, lane); row += f" {gl:13.3f} {g(d, ks, lambda r: r['pcg_ms']):7.3f} [{g(d, nb, lambda r: r['pcg_ms']):5.3f}] {('yes' if gl > 1 else 'no') if gl == gl else '-':>4s} |"
        L.append(row + f" {len(nb)}/{len(ks)}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
