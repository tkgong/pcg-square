#!/usr/bin/env python3
"""Attribution from the paper's Fig. 8 to the final design (final_design.py accounting; replaces the inline script that wrote
attribution_vs_paper.txt). Cumulative steps, up to (best (c,t), geomean over logN) / geomean over all cells, delta = step /
previous:
  0 paper as published            1 reproduced: paper's lanes (square NTT on both sides, paper's SPU lane), network serial on the GPU side
  2 GPU baseline NTT -> Cheddar   3 PCG^2 NTT -> the final design, simulated (incl. the DPF->NTT handoff)
  4 SPU lane -> final (leaf-range assignment, H' on the SPU, LSU overlap)     5 contention (co-simulated)
  6 GPU baseline co-schedules the network (= Fig. 8 final)
Usage: attribution.py RUN_DIR OUT_TXT"""
import io, contextlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e")); sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
import final_design as FD
W, OUT = sys.argv[1], sys.argv[2]
R = json.load(open(f"{W}/e2e_nom.json"))
PAPER = {"fast": {"L40S": (3.19, 2.54), "B200": (9.49, 7.25)}, "slow": {"L40S": (3.38, 2.61), "B200": (5.93, 3.49)}}
STEPS = ["paper as published", "reproduced: paper's lanes, GPU network serial", "GPU baseline NTT -> Cheddar", "PCG^2 NTT -> final design (simulated, incl. DPF->NTT handoff)",
         "SPU lane -> final (leaf-range assignment, H' on the SPU, LSU overlap)", "contention (co-simulated)", "GPU baseline co-schedules the network (= Fig. 8 final)"]
L = [__doc__.split("Usage")[0].strip(), ""]
for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
    for org, mach in FD.MACH.items():
        LN = L_(mach); des = FD.des(org, W); C = FD.cells(R, org, des, tier)
        per = {i: [] for i in range(1, 7)}; allv = {i: [] for i in range(1, 7)}
        for (c, t) in CFG:
            v = {i: [] for i in range(1, 7)}
            for k, r in C.items():
                if k[:2] != (c, t) or k not in LN: continue
                Lk = LN[k]; nic = r["nic_ms"]; bs = r[FD.GPU_KEY] + nic
                v[1].append(r["base_paper_ms"] / max(Lk["spu"], Lk["sm"], nic))
                v[2].append(bs / max(Lk["spu"], Lk["sm"], nic))
                v[3].append(bs / max(Lk["spu"], r["ntt_ms"], nic))
                v[4].append(bs / max(r["spu_ms"], r["ntt_ms"], nic))
                v[5].append(bs / r["pcg_ms"])
                v[6].append(r[FD.BASE_KEY] / r["pcg_ms"])
            for i in v:
                if v[i]: per[i].append(gm(v[i])); allv[i] += v[i]
        pu, pg = PAPER[tier][mach]
        L.append(f"{mach} {lab} (PCG^2 NTT {des}): paper {pu:.2f} / {pg:.2f}")
        prev = (pu, pg); L.append(f"   0  {STEPS[0]:78s} {pu:5.2f} / {pg:5.2f}")
        for i in range(1, 7):
            u, g = max(per[i]), gm(allv[i])
            L.append(f"   {i}  {STEPS[i]:78s} {u:5.2f} / {g:5.2f}   x{u/prev[0]:.3f} / x{g/prev[1]:.3f}"); prev = (u, g)
        L.append(f"   final vs paper: {prev[0]/pu-1:+.0%} / {prev[1]/pg-1:+.0%}\n")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
