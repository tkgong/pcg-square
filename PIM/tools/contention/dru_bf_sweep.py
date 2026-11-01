#!/usr/bin/env python3
"""Butterfly units per channel (U) for the butterfly DRU (design f4b), from the main window re-run at U = 1, 2, 4
(e2e_window.py --dru-bf U, raw run dirs with the per-cell .out files). Per card and U: the DRU pipe vs the SM pipe of the
standalone NTT lane per cell (pipe1 / pipe0 finish), the lane and end-to-end gain over Cheddar's NTT on the SMs, and the
choice: the smallest U whose DRU pipe is not the longer one in any NTT-bound cell (else the largest U swept).
Writes OUT_TXT and OUT_JSON ({org: U}, read by final_design.dru_bf).   Usage: dru_bf_sweep.py OUT_TXT OUT_JSON U=RUN_DIR [U=RUN_DIR ...]"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import final_design as FD
OUT, OUTJ = sys.argv[1], sys.argv[2]; RUNS = sorted((int(a.split("=")[0]), a.split("=")[1]) for a in sys.argv[3:])
parse = lambda p: {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]
L = [__doc__.split("Usage")[0].strip(), "", FD.ESTIMATE_NOTE, ""]; choice = {}
for org in FD.MACH:
    L.append(f"{FD.MACH[org]}: U | cells | DRU pipe / SM pipe, standalone: geomean (max) | cells with the DRU pipe longer: all / NTT-bound | NTT lane Cheddar/f4b | e2e Cheddar/f4b | e2e f4b alone (no contention) / Cheddar")
    ok = None
    for U, d in RUNS:
        R = json.load(open(os.path.join(d, "e2e.json")))
        S, B = FD.cells(R, org, FD.SM_NTT), FD.cells(R, org, "f4b"); ks = sorted(k for k in B if k in S)
        if not ks: L.append(f"  {U} | no f4b cells (SM lane not measured on this card)"); continue
        ratio, longer, longer_nb = {}, [], []
        for k in ks:
            o = parse(os.path.join(d, f"{org}_{k[0]}_{k[1]}_{k[2]}_f4b_alone.out")); ratio[k] = o.get("pipe1_finish_max", 0) / o["pipe0_finish_max"]
            nb = lane(S[k]) >= S[k]["spu_ms"] * S[k]["spu_slow"]
            if ratio[k] > 1: longer.append(k); longer_nb += [k] if nb else []
        L.append(f"  {U} | {len(ks)} | {FD.gm(ratio.values()):.3f} ({max(ratio.values()):.3f}) | {len(longer)} / {len(longer_nb)} | {FD.gm(lane(S[k]) / lane(B[k]) for k in ks):.3f} | "
                 f"{FD.gm(S[k]['pcg_ms'] / B[k]['pcg_ms'] for k in ks):.3f} | {FD.gm(S[k]['pcg_nocont_ms'] / B[k]['pcg_nocont_ms'] for k in ks):.3f}")
        if ok is None and not longer_nb: ok = U
    if any(True for U, d in RUNS):
        choice[org] = ok if ok is not None else RUNS[-1][0]
        L.append(f"  -> U = {choice[org]} ({'smallest U with the DRU pipe hidden in every NTT-bound cell' if ok is not None else 'no swept U hides the DRU pipe everywhere: the largest swept'}); "
                 f"DRU per channel {FD.DRU_REORDER_MM2 + choice[org] * FD.BF_UNIT_MM2 + FD.BF_BUF_MM2(12):.3f} mm^2 (estimate), {FD.NDRU[org]} channels -> {FD.NDRU[org] * (FD.DRU_REORDER_MM2 + choice[org] * FD.BF_UNIT_MM2 + FD.BF_BUF_MM2(12)):.1f} mm^2 vs SPUs {FD.NSPU[org] * FD.SPU_MM2:.1f} mm^2")
    L.append("")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
json.dump(choice, open(OUTJ, "w"), indent=1)
