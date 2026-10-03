#!/usr/bin/env python3
"""Q7: the DRU's gain as a function of channel count. Derived orgs b200_ch{48,96,128,192,512} keep B200's
HBM3e timing, GPU kernels (compute floors) and SPU design; bytes per channel scale with 256/ch and the SPU
count is 8 per channel. Per channel count: PCG^2 time without / with the DRU (merge vs four-step+DRU),
the SM-side NTT's data-bus occupancy, the number of NTT-bound cells, and the lane growth vs 256 channels.
Usage: channel_sweep.py SWEEP_DIR B200_DIR OUT_FILE"""
import json, os, re, sys
from math import exp, log
def gm(v): v = list(v); return exp(sum(map(log, v)) / len(v))
def parse(p): return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}
sweep, b200, out = sys.argv[1], sys.argv[2], sys.argv[3]
def rows(d, org): return [r for r in json.load(open(os.path.join(d, "e2e.json"))) if r["org"] == org and r["tier"] == "fast"]
ref = {(r["c"], r["t"], r["logN"], r["design"]): r for r in rows(b200, "b200")}
lines = [f"{'channels':>8s} {'SPUs':>5s} | {'DRU gain geomean':>16s} {'max':>5s} | {'NTT-bound cells':>15s} | {'SM-NTT bus (merge / 4-step+DRU)':>31s} | {'SPU lane':>8s} {'NTT lane':>8s} (vs 256 ch, (4,16) 2^24)"]
for ch in (48, 96, 128, 192, 256, 512):
    org, d = ("b200", b200) if ch == 256 else (f"b200_ch{ch}", sweep)
    R = {(r["c"], r["t"], r["logN"], r["design"]): r for r in rows(d, org)}
    cells = sorted({k[:3] for k in R})
    gain = [R[k + ("merge",)]["pcg_ms"] / R[k + ("f4dru",)]["pcg_ms"] for k in cells]
    nb = sum(R[k + ("f4dru",)]["ntt_ms"] * R[k + ("f4dru",)]["ntt_slow"] > R[k + ("f4dru",)]["spu_ms"] * R[k + ("f4dru",)]["spu_slow"] for k in cells)
    bus = []
    for des in ("merge", "f4dru"):
        a = parse(os.path.join(d, f"{org}_4_16_24_{des}_alone.out"))
        g = a.get("CH0_gpu_bus_cycles", 0); u = max(0, a.get("CH0_num_RD_commands", 0) + a.get("CH0_num_WR_commands", 0) - a.get("CH0_gpu_rd_cols", 0) - a.get("CH0_gpu_wr_cols", 0)) * 2
        bus.append((g / a["pipe0_finish_max"], u / a["pipe0_finish_max"]))
    k = (4, 16, 24, "f4dru")
    lines.append(f"{ch:8d} {8*ch:5d} | {gm(gain):16.3f} {max(gain):5.2f} | {nb:7d} of {len(cells):2d}   | {bus[0][0]:.2f} / {bus[1][0]:.2f} + DRU {bus[1][1]:.2f}             | "
                 f"{R[k]['spu_ms'] / ref[k]['spu_ms']:8.2f} {R[k]['ntt_ms'] * R[k]['ntt_slow'] / (ref[k]['ntt_ms'] * ref[k]['ntt_slow']):8.2f}")
open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
