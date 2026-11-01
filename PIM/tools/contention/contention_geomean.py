#!/usr/bin/env python3
"""Per-channel bandwidth utilization, DRAM queueing latency and throughput degradation for the three
configurations (GPU-only NTT = Cheddar alone; GPU + SPU = Cheddar next to the SPUs; GPU + SPU + DRU = the card's DRU
design (final_design.py) next to the SPUs), per cell of the suite and as the geomean over all cells of each card.
Usage: contention_geomean.py RUN_DIR"""
import glob, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_design as FD
parse = lambda p: {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}
gm = lambda v: math.exp(sum(math.log(max(x, 1e-9)) for x in v) / len(v))
TCK = {"b200": 0.5, "l40s": 0.444}; W = sys.argv[1]
def cols(d):
    gpu_rd, gpu_wr = d.get("CH0_gpu_rd_cols", 0), d.get("CH0_gpu_wr_cols", 0)
    dru = max(0, d.get("CH0_num_RD_commands", 0) + d.get("CH0_num_WR_commands", 0) - gpu_rd - gpu_wr)
    return d.get("CH0_gpu_bus_cycles", 0), d.get("CH0_pim_allbank_slot_cycles", 0), dru * 2
rows = json.load(open(f"{W}/e2e.json"))
for org in ("l40s", "b200"):
    tck = TCK[org]; cells = sorted({(r["c"], r["t"], r["logN"]) for r in rows if r["org"] == org})
    acc = {k: [] for k in ("g0", "q0", "p0", "g1", "q1", "p1", "n1", "s1", "g2", "u2", "q2", "p2", "n2", "s2")}
    print(f"\n{org.upper()} (DRU design {FD.dru_ntt(org, W)}): per-channel data-bus utilization (GPU + DRU; SPU column-slot share in brackets), GPU queueing mean/p99 (ns), throughput degradation (1 - 1/slowdown)")
    print(f"  {'cell':16s} | {'GPU-only NTT':>22s} | {'GPU + SPU':>40s} | {'GPU + SPU + DRU':>46s}")
    for (c, t, lg) in cells:
        cell = f"{c}_{t}_{lg}"
        R = {r["design"]: r for r in rows if r["org"] == org and r["tier"] == "fast" and (r["c"], r["t"], r["logN"]) == (c, t, lg)}
        SM, DR = FD.SM_NTT, FD.dru_ntt(org, W)
        A = parse(f"{W}/{org}_{cell}_{SM}_alone.out"); M = parse(f"{W}/{org}_{cell}_{SM}.out"); F = parse(f"{W}/{org}_{cell}_{DR}.out")
        g, _, _ = cols(A); g0 = g / A["pipe0_finish_max"]; q0, p0 = A.get("CH0_gpu_q_mean", 0) * tck, A.get("CH0_gpu_q_p99", 0) * tck
        g, p, _ = cols(M); g1 = g / M["pipe0_finish_max"]; s1 = p / M["pim_done_cycles"]; q1, p1 = M["CH0_gpu_q_mean"] * tck, M["CH0_gpu_q_p99"] * tck
        g, p, u = cols(F); g2 = g / F["pipe0_finish_max"]; u2 = u / F["pipe0_finish_max"]; s2 = p / F["pim_done_cycles"]; q2, p2 = F["CH0_gpu_q_mean"] * tck, F["CH0_gpu_q_p99"] * tck
        n1, sp1 = R[SM]["ntt_slow"], R[SM]["spu_slow"]; n2, sp2 = R[DR]["ntt_slow"], R[DR]["spu_slow"]
        for k, v in zip(acc, (g0, q0, p0, g1, q1, p1, n1, sp1, g2, u2, q2, p2, n2, sp2)): acc[k].append(v)
        print(f"  ({c},{t:3d}) 2^{lg:<2d}    | {g0*100:4.0f}%  {q0:5.0f}/{p0:5.0f}        | {g1*100:4.0f}% [{s1*100:3.0f}%]  {q1:5.0f}/{p1:5.0f}  NTT -{(1-1/n1)*100:4.1f}% SPU -{(1-1/sp1)*100:3.1f}% | {g2*100:4.0f}%+{u2*100:3.0f}% [{s2*100:3.0f}%]  {q2:5.0f}/{p2:5.0f}  NTT -{(1-1/n2)*100:4.1f}% SPU -{(1-1/sp2)*100:3.1f}%")
    a = {k: gm(v) for k, v in acc.items()}
    print(f"  {'geomean':16s} | {a['g0']*100:4.0f}%  {a['q0']:5.0f}/{a['p0']:5.0f}        | {a['g1']*100:4.0f}%        {a['q1']:5.0f}/{a['p1']:5.0f}  NTT -{(1-1/a['n1'])*100:4.1f}% SPU -{(1-1/a['s1'])*100:3.1f}% | {a['g2']*100:4.0f}%+{a['u2']*100:3.0f}%        {a['q2']:5.0f}/{a['p2']:5.0f}  NTT -{(1-1/a['n2'])*100:4.1f}% SPU -{(1-1/a['s2'])*100:3.1f}%")
    print(f"  {'':16s}   (GPU+SPU total bus {a['g1']*100:.0f}%, GPU+SPU+DRU total bus {(a['g2']+a['u2'])*100:.0f}%; geomean of slowdowns {a['n1']:.3f}/{a['s1']:.3f} and {a['n2']:.3f}/{a['s2']:.3f})")
