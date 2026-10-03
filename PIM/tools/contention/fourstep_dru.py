#!/usr/bin/env python3
"""The DRU against the submission's own four-step NTT (designs sq = all on the SMs, sqdru = transposes and
bit-reversal passes on the DRU), from e2e_window.py runs:
  * per-cell DRU gain with contention (PCG^2 sq / PCG^2 sqdru), NTT-bound cells, Table IV-style rows
  * the bandwidth rule u = (SM-side NTT + DRU + SPU bus occupancy) / channel, from the simulator's own counters
  * optional channel sweep (derived orgs b200_chNN) -> DRU gain and u per channel count (the knee)
Usage: fourstep_dru.py RUN_DIR [SWEEP_DIR] [DRU_DESIGN=sqdru] > out.txt   (DRU_DESIGN sqdruh = conservative DRU paying ACT/PRE)"""
import glob, json, os, re, sys
from math import exp, log
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e"))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import CFG, gm


def parse(p):
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}


def bus(d, T):
    """(SM-side GPU, DRU) data-bus occupancy of a host pipeline over its own span; the DRU is either the tap
    (columns outside the GPU class) or, for the conservative model, the class-3 host stream"""
    gpu_rd, gpu_wr = d.get("CH0_gpu_rd_cols", 0), d.get("CH0_gpu_wr_cols", 0)
    dru = max(0, d.get("CH0_num_RD_commands", 0) + d.get("CH0_num_WR_commands", 0) - gpu_rd - gpu_wr - d.get("CH0_agg_rd_cols", 0) - d.get("CH0_agg_wr_cols", 0))
    return d.get("CH0_gpu_bus_cycles", 0) / T, (dru * 2 + d.get("CH0_agg_bus_cycles", 0)) / T


def cells(run, org):
    R = {}
    for r in json.load(open(os.path.join(run, "e2e.json"))):
        if r["org"] == org: R[(r["design"], r["tier"], r["c"], r["t"], r["logN"])] = r
    return R


def report(run, org, label):
    R = cells(run, org); ks = sorted({k[2:] for k in R if k[0] == "sq" and k[1] == "fast"})
    if not ks: return
    spu = parse(sorted(glob.glob(os.path.join(run, f"spu_{org}_*clk.out")))[-1]); spu_bus = spu["CH0_pim_allbank_slot_cycles"] / spu["pim_done_cycles"]
    print(f"\n{label}: PCG^2 with the four-step on the SMs (sq) vs transposes+brev on the DRU (sqdru); SPU bus share {spu_bus:.2f}")
    print(f"  {'cell':14s} {'NTT lane sq':>11s} {'x slow':>6s} {'sqdru':>8s} {'x slow':>6s} {'SPU lane':>8s} | {'bus: sq SM':>10s} {'sqdru SM':>8s} {'+DRU':>5s} {'u':>5s} | {'bound':>5s} | {'DRU gain':>8s} {'no-cont':>7s}")
    g, gn, nb, us = [], [], 0, []
    for k in ks:
        a, b = R[("sq", "fast") + k], R[(DRU, "fast") + k]
        A = parse(os.path.join(run, f"{org}_{k[0]}_{k[1]}_{k[2]}_sq_alone.out")); B = parse(os.path.join(run, f"{org}_{k[0]}_{k[1]}_{k[2]}_sqdru_alone.out"))
        ba, _ = bus(A, A["pipe0_finish_max"]); bb, bd = bus(B, B["pipe0_finish_max"]); u = bb + bd + spu_bus; us.append(u)
        bound = "NTT" if a["ntt_ms"] * a["ntt_slow"] > a["spu_ms"] * a["spu_slow"] else "SPU"; nb += bound == "NTT"
        ga, g0 = a["pcg_ms"] / b["pcg_ms"], a["pcg_nocont_ms"] / b["pcg_nocont_ms"]; g.append(ga); gn.append(g0)
        print(f"  ({k[0]},{k[1]:3d}) 2^{k[2]}  {a['ntt_ms']:11.1f} {a['ntt_slow']:6.3f} {b['ntt_ms']:8.1f} {b['ntt_slow']:6.3f} {a['spu_ms']:8.1f} | {ba:10.2f} {bb:8.2f} {bd:5.2f} {u:5.2f} | {bound:>5s} | {ga:8.3f} {g0:7.3f}")
    print(f"  geomean DRU gain {gm(g):.3f} (range {min(g):.2f}-{max(g):.2f}), without contention {gm(gn):.3f}; NTT-bound cells {nb}/{len(ks)}; u {min(us):.2f}-{max(us):.2f}")
    # Table IV-style rows: geomean over cells, both tiers
    for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
        rows = {"+SPU (serial)": [], "+SPU+DRU (serial)": [], "+SPU+co-sched": [], "full (+DRU, co-sched)": []}
        for k in ks:
            a, b = R[("sq", tier) + k], R[(DRU, tier) + k]; nic = a["nic_ms"]
            rows["+SPU (serial)"].append(a["spu_ms"] + a["ntt_ms"] + nic); rows["+SPU+DRU (serial)"].append(b["spu_ms"] + b["ntt_ms"] + nic)
            rows["+SPU+co-sched"].append(a["pcg_ms"]); rows["full (+DRU, co-sched)"].append(b["pcg_ms"])
        g0 = gm(rows["+SPU (serial)"])
        print(f"  Table IV-style ({lab}), geomean PCG^2 time relative to +SPU serial: " + "; ".join(f"{n} {gm(v)/g0:.3f}" for n, v in rows.items())
              + f" | DRU increment: serial {gm(rows['+SPU (serial)'])/gm(rows['+SPU+DRU (serial)']):.3f}, co-scheduled {gm(rows['+SPU+co-sched'])/gm(rows['full (+DRU, co-sched)']):.3f}")


run = sys.argv[1]
DRU = sys.argv[3] if len(sys.argv) > 3 else "sqdru"
for org, lab in (("b200", "B200"), ("l40s", "L40S")):
    report(run, org, lab)
if len(sys.argv) > 2:
    sweep = sys.argv[2]
    print("\nChannel sweep, four-step with vs without the DRU (B200 HBM3e timing, 8 SPUs per channel):")
    print(f"  {'channels':>8s} {'DRU gain geomean':>16s} {'max':>5s} {'NTT-bound':>9s} | {'sq SM bus':>9s} {'sqdru SM':>8s} {'+DRU':>5s} {'SPU':>5s} {'u':>5s} ((4,16) 2^24) | {'NTT lane sqdru/sq':>17s}")
    for ch in (48, 96, 128, 192, 256, 512):
        org, d = ("b200", run) if ch == 256 else (f"b200_ch{ch}", sweep)
        R = cells(d, org); ks = sorted({k[2:] for k in R if k[0] == "sq" and k[1] == "fast"})
        if not ks: continue
        g = [R[("sq", "fast") + k]["pcg_ms"] / R[(DRU, "fast") + k]["pcg_ms"] for k in ks]
        nb = sum(R[("sq", "fast") + k]["ntt_ms"] * R[("sq", "fast") + k]["ntt_slow"] > R[("sq", "fast") + k]["spu_ms"] * R[("sq", "fast") + k]["spu_slow"] for k in ks)
        spu = parse(sorted(glob.glob(os.path.join(d, f"spu_{org}_*clk.out")))[-1]); sb = spu["CH0_pim_allbank_slot_cycles"] / spu["pim_done_cycles"]
        A = parse(os.path.join(d, f"{org}_4_16_24_sq_alone.out")); B = parse(os.path.join(d, f"{org}_4_16_24_sqdru_alone.out"))
        ba, _ = bus(A, A["pipe0_finish_max"]); bb, bd = bus(B, B["pipe0_finish_max"])
        k = (4, 16, 24); lane = (R[(DRU, "fast") + k]["ntt_ms"] * R[(DRU, "fast") + k]["ntt_slow"]) / (R[("sq", "fast") + k]["ntt_ms"] * R[("sq", "fast") + k]["ntt_slow"])
        print(f"  {ch:8d} {gm(g):16.3f} {max(g):5.2f} {nb:4d}/{len(ks):<4d} | {ba:9.2f} {bb:8.2f} {bd:5.2f} {sb:5.2f} {bb+bd+sb:5.2f}              | {lane:17.3f}")
