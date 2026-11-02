#!/usr/bin/env python3
"""Reviewer B: the DRU's gain as a function of the channel count (B200 variants: HBM3e timing, per-channel bandwidth,
GPU kernels and SPU design fixed; 8 SPUs + 1 DRU per channel; device bytes spread over C channels), and the knee.
Per channel count C: NTT-lane and end-to-end gain of the DRU inside the four-step (f4sm -> f4dru), the Table IV
"DRU design" gain (submission's four-step on SMs, sq -> f4dru), the SOTA check (merge -> f4dru: the runtime enables the
DRU only where this is > 1), the conservative-DRU gain on the submission's four-step (sq -> sqdruh), the bus
utilisation u of Eq. dru_enable and its needed-vs-available bandwidth at (4,16) 2^24.
Usage: channel_dse.py FINAL_WINDOW_DIR F4SM_SWEEP_DIR OUT_TXT"""
import json, os, sys
from math import exp, log
W, F4, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
def gm(v): v = list(v); return exp(sum(map(log, v)) / len(v)) if v else float("nan")
def load(p): return json.load(open(p)) if os.path.exists(p) else []
def cells(rows, org, des): return {(r["c"], r["t"], r["logN"]): r for r in rows if r["org"] == org and r["design"] == des and r["tier"] == "fast"}
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]; spul = lambda r: r["spu_ms"] * r["spu_slow"]
SRC = {"merge": load(f"{W}/channel_sweep/e2e.json") + load(f"{W}/win22/e2e_nom.json"), "f4dru": load(f"{W}/channel_sweep/e2e.json") + load(f"{W}/win22/e2e_nom.json"),
       "f4sm": load(f"{F4}/e2e.json") + load(f"{W}/win22/e2e_f4sm.json"), "sq": load(f"{W}/fourstep_dru/e2e_chsweep_conservative_dru.json") + load(f"{W}/fourstep_dru/e2e_nom_conservative_dru.json"),
       "sqdruh": load(f"{W}/fourstep_dru/e2e_chsweep_conservative_dru.json") + load(f"{W}/fourstep_dru/e2e_nom_conservative_dru.json")}
BW_CH, U_SPU, T_F4SM, N = 8.2e12 / 256, 0.21, 2.2766e-3, 1 << 24          # per-channel bandwidth, SPU column-slot share, SM lane at (24,16)
L = ["DRU gain vs channel count (B200 variants). Gains are geomean over the 25 cells (NTT-bound subset in brackets); u = (B_SM+B_DRU)/(BW t_f4sm) + u_SPU at (4,16) 2^24.", "",
     f"{'ch':>4s} {'SPUs':>5s} {'agg BW':>7s} | {'needed':>6s} {'avail':>6s} {'u':>5s} | {'DRU in four-step: NTT lane':>26s} {'e2e [NTT-bound]':>16s} | {'sq->f4dru e2e':>13s} | {'merge->f4dru NTT':>16s} {'e2e':>6s} {'on?':>4s} | {'sq->sqdruh e2e':>14s} | NTT-bound"]
for ch in (48, 96, 128, 192, 256, 512):
    org = "b200" if ch == 256 else f"b200_ch{ch}"
    C = {d: cells(SRC[d], org, d) for d in SRC}
    ks = [k for k in C["f4dru"] if k in C["merge"]]
    nb = [k for k in ks if k in C["f4sm"] and lane(C["f4sm"][k]) >= spul(C["f4sm"][k])]
    bw = BW_CH * ch; need = 240 * N / T_F4SM; avail = bw * (1 - U_SPU); u = need / bw + U_SPU
    def g(a, b, keys=None):
        keys = [k for k in (keys if keys is not None else ks) if k in C[a] and k in C[b]]
        return gm(C[a][k]["pcg_ms"] / C[b][k]["pcg_ms"] for k in keys) if keys else float("nan")
    def gl(a, b):
        keys = [k for k in ks if k in C[a] and k in C[b]]
        return gm(lane(C[a][k]) / lane(C[b][k]) for k in keys) if keys else float("nan")
    on = gl("merge", "f4dru")
    L.append(f"{ch:4d} {8*ch:5d} {bw/1e12:6.2f}T | {need/1e12:5.2f}T {avail/1e12:5.2f}T {u:5.2f} | {gl('f4sm','f4dru'):26.3f} {g('f4sm','f4dru'):7.3f} [{g('f4sm','f4dru',nb):5.3f}] | {g('sq','f4dru'):13.3f} | {on:16.3f} {g('merge','f4dru'):6.3f} {'yes' if on > 1 else 'no':>4s} | {g('sq','sqdruh'):14.3f} | {len(nb)}/{len(ks)}")
L.append("")
L.append(f"Eq. dru_enable knee (needed = available): C* = {need / (BW_CH * (1 - U_SPU)):.0f} channels = {need/(1-U_SPU)/1e12:.2f} TB/s aggregate; L40S (24 GDDR6 channels, 0.86 TB/s) is below it and the runtime keeps merge there.")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
