#!/usr/bin/env python3
"""Fit the simulator's DRAM access pattern of each NTT to silicon.
Silicon: GPU_baseline/fused4 interfere_ntt (L40S logN 22): net slowdown of merge and
of the four-step SM lane vs a streaming aggressor's co-run bandwidth. Simulator:
victim stream (the NTT's bytes / read mix, window W, run R columns per row visit)
next to an aggressor stream (calibrated GPU pattern) at several gaps; slowdown vs
the aggressor's bandwidth while the victim ran. Reports the RMS error per (W, R)."""
import argparse, concurrent.futures as cf, csv, os, re, subprocess
PIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YAML = os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml")
RD = {"merge": 88 / 168, "f4g_sm": 88 / 144}
PEAK = 16.0  # B/CK/channel
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--silicon", required=True); ap.add_argument("--jobs", type=int, default=60)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
# silicon points: victim -> [(agg co-run BW fraction, net slowdown)], streaming aggressor
rows = list(csv.DictReader(open(a.silicon)))
ctrl = {(r["victim"], r["blocks"]): float(r["victim_ms"]) for r in rows if r["mode"] == "sleep_only"}
SIL = {}
for r in rows:
    if r["mode"] == "stream":
        SIL.setdefault(r["victim"], []).append((float(r["agg_GBps"]) / 864.0, float(r["victim_ms"]) / ctrl[(r["victim"], r["blocks"])]))
def run(tag, lines):
    sp = os.path.join(a.out, tag + ".streams"); out = os.path.join(a.out, tag + ".out")
    open(sp, "w").write("".join(" ".join(map(str, l)) + "\n" for l in lines))
    if not (os.path.exists(out) and "memory_system_cycles" in open(out).read()):
        with open(out, "w") as f:
            subprocess.run([a.sim, "-f", YAML, "-t", eoc, "-p", "MemorySystem.DRAM.org.channel=2", "-p", "Frontend.issue_width=2",
                            "-p", f"MemorySystem.host_streams={sp}", "-p", "MemorySystem.Controller.wr_max_age=1000",
                            "-p", "MemorySystem.Controller.class_priority=1,3,0,2"], stdout=f, stderr=subprocess.STDOUT)
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(out).read(), re.M)}
VB = 400_000
GAPS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
def curve(v, W, R):
    base = run(f"{v}_W{W}_R{R}_alone", [(1, VB, f"{RD[v]:.4f}", W, 0, R, 16384, 0)])["stream0_class1_finish_mean"]
    pts = []
    for g in GAPS:
        d = run(f"{v}_W{W}_R{R}_g{g}", [(1, VB, f"{RD[v]:.4f}", W, 0, R, 16384, 0), (3, 40 * VB, 0.5, 128, g, 32, 40000, 0)])
        t = d["stream0_class1_finish_mean"]
        pts.append((d["stream1_cols_at_s0_done"] * 32 / t / PEAK, t / base))
    return pts
def interp(pts, x):
    pts = sorted(pts)
    if x <= pts[0][0]: return 1 + (pts[0][1] - 1) * x / pts[0][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]
grid = [(v, W, R) for v in RD for W in (32, 64, 128) for R in (1, 2, 4, 8, 16, 32)]
res = {}
with cf.ThreadPoolExecutor(a.jobs) as ex:
    for (v, W, R), pts in zip(grid, ex.map(lambda g: curve(*g), grid)): res[(v, W, R)] = pts
for v in RD:
    sil = [p for p in SIL[v] if p[0] <= 0.25]
    best = []
    for (vv, W, R), pts in res.items():
        if vv != v: continue
        err = (sum((interp(pts, x) - y) ** 2 for x, y in sil) / len(sil)) ** 0.5
        best.append((err, W, R, pts))
    best.sort()
    print(f"== {v}: silicon (agg BW frac, net) {[(round(x,3), round(y,3)) for x, y in sorted(sil)]}")
    for err, W, R, pts in best[:4]:
        print(f"   W={W:3d} R={R:2d} rms={err:.3f}  sim {[(round(x,3), round(y,3)) for x, y in sorted(pts)]}")
