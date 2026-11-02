#!/usr/bin/env python3
"""Channel-count DSE table (Fig. 9(b) data): SPU array fixed per card, per-channel bandwidth fixed, one DRU per channel.
Per channel count: NTT lane (co-simulated, incl. gather and contention) with the DRU, vs the real card's two GPU-only NTTs
(four-step with the transposes in shared memory, HEonGPU merge); and end-to-end runtime (alpha = 500 us) vs the real card
with the transposes in shared memory.  Geomean over the suite.   Usage: channel_dse_table.py FINAL_WINDOW_DIR [OUT]"""
import io, contextlib, json, os, sys
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_
W = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
gm = lambda v: (lambda l: exp(sum(map(log, l)) / len(l)))(list(v))
load = lambda p: json.load(open(p)) if os.path.exists(p) else []
cells = lambda rows, org, des: {(r["c"], r["t"], r["logN"]): r for r in rows if r["org"] == org and r["design"] == des and r["tier"] == "fast"}
lane = lambda r: r["ntt_ms"] * r["ntt_slow"]
A500 = 500
CARD = {"B200": dict(org="b200", native=256, chs=[8, 16, 32, 48, 96, 128, 192, 256, 512], bw_ch=31.25,
                     rows=load(f"{W}/win22/e2e_nom.json") + load(f"{W}/win22/e2e_f4sm.json") + load(f"{W}/channel_sweep/e2e.json") + load(f"{W}/channel_sweep/e2e_f4sm.json") + load(f"{W}/channel_sweep/e2e_b200_low.json") + load(f"{W}/channel_sweep/e2e_f4sm_b200_low.json")),
        "L40S": dict(org="l40s", native=24, chs=[8, 16, 24, 48, 96, 128, 192, 256, 512], bw_ch=36.0,
                     rows=load(f"{W}/win22/e2e_nom.json") + load(f"{W}/win22/e2e_f4sm.json") + load(f"{W}/channel_sweep/e2e_l40s.json") + load(f"{W}/channel_sweep/e2e_l40s_low.json") + load(f"{W}/channel_sweep/e2e_f4sm_l40s.json"))}
lines = []
for mach, c in CARD.items():
    LN = L_(mach); nat = cells(c["rows"], c["org"], "f4dru"); SPU = {k: r["spu_ms"] * r["spu_slow"] for k, r in nat.items()}
    rt = lambda d: gm(max(SPU[k], lane(r), (LN[k]["n"] + 2) * A500 / 1000.0) for k, r in d.items() if k in SPU)
    sm, mg = cells(c["rows"], c["org"], "f4sm"), cells(c["rows"], c["org"], "merge")
    t_sm, t_mg, e_sm, e_mg = gm(lane(sm[k]) for k in SPU), gm(lane(mg[k]) for k in SPU), rt(sm), rt(mg)
    lines += [f"{mach}: {len(SPU)} cells, SPU lane geomean {gm(SPU.values()):.1f} ms; real card ({c['native']} ch): NTT lane sm-transpose {t_sm:.1f} ms, HEonGPU {t_mg:.1f} ms; e2e sm-transpose {e_sm:.1f} ms, HEonGPU {e_mg:.1f} ms",
              f"  ch   BW(TB/s) | NTT lane DRU (ms)  vs sm-transpose  vs HEonGPU | e2e DRU (ms)  vs sm-transpose  vs HEonGPU"]
    for ch in c["chs"]:
        org = c["org"] if ch == c["native"] else f"{c['org']}_ch{ch}"; d = cells(c["rows"], org, "f4dru")
        if not d: continue
        tl, te = gm(lane(d[k]) for k in SPU), rt(d)
        lines.append(f"  {ch:3d}  {ch * c['bw_ch'] / 1e3:7.2f} | {tl:12.1f}  {t_sm / tl:14.3f}  {t_mg / tl:10.3f} | {te:10.1f}  {e_sm / te:14.3f}  {e_mg / te:10.3f}")
    lines.append("")
s = "\n".join(lines); print(s)
if OUT: open(OUT, "w").write(s + "\n")
