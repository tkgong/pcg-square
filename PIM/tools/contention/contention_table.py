#!/usr/bin/env python3
"""Per-channel contention table (Q4) from an e2e_window.py run directory: data-bus occupancy of
each unit over its own active span, GPU queueing (arrival -> first command) and read latency,
SPU queueing vs alone, GPU row conflicts and the measured slowdowns, for SPU-only / GPU-only /
GPU+SPU / GPU+SPU+DRU.  Usage: contention_table.py RUN_DIR [org:cell:label ...]"""
import json, re, sys


def parse(p):
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}


TCK = {"b200": 0.5, "l40s": 0.444}
W = sys.argv[1]
CELLS = [a.split(":") for a in sys.argv[2:]] or [("b200", "4_16_24", "B200 (4,16) 2^24"), ("l40s", "4_16_22", "L40S (4,16) 2^22")]


def cols(d):
    gpu_rd, gpu_wr = d.get("CH0_gpu_rd_cols", 0), d.get("CH0_gpu_wr_cols", 0)
    dru = max(0, d.get("CH0_num_RD_commands", 0) + d.get("CH0_num_WR_commands", 0) - gpu_rd - gpu_wr)   # DRU tap columns
    return d.get("CH0_gpu_bus_cycles", 0), d.get("CH0_pim_allbank_slot_cycles", 0), dru * 2


for org, cell, lab in CELLS:
    tck = TCK[org]
    R = {r["design"]: r for r in json.load(open(f"{W}/e2e.json")) if r["org"] == org and r["tier"] == "fast" and f'{r["c"]}_{r["t"]}_{r["logN"]}' == cell}
    spu = parse(f"{W}/spu_{org}_clk.out"); pq0 = spu["CH0_pim_q_mean"] * tck
    print(f"\n{lab}: per-channel data-bus occupancy over each unit's own active span; GPU queueing = arrival -> first command (ns)")
    print(f"  {'config':28s} {'GPU':>5s} {'SPU':>5s} {'DRU':>5s} | {'GPU q mean/p95/p99':>19s} | {'GPU rdlat mean/p99':>18s} | {'SPU q vs alone':>14s} | GPU row conflicts | slowdown")

    def pr(name, d, Tg, Tp, sl="", pq="", rc=""):
        g, p, u = cols(d)
        print(f"  {name:28s} {g/Tg if Tg else 0:5.2f} {p/Tp if Tp else 0:5.2f} {u/Tg if Tg else 0:5.2f} | "
              f"{d.get('CH0_gpu_q_mean',0)*tck:6.0f}/{d.get('CH0_gpu_q_p95',0)*tck:5.0f}/{d.get('CH0_gpu_q_p99',0)*tck:5.0f}    | "
              f"{d.get('CH0_gpu_rdlat_mean',0)*tck:7.0f}/{d.get('CH0_gpu_rdlat_p99',0)*tck:6.0f}      | {pq:>14} | {rc:>17} | {sl}")

    pr("SPU only", spu, 0, spu["pim_done_cycles"], pq="+0 ns")
    for des, nm in (("merge", "GPU merge NTT only"), ("f4dru", "GPU 4-step NTT + DRU only")):
        d = parse(f"{W}/{org}_{cell}_{des}_alone.out")
        pr(nm, d, d["pipe0_finish_max"], 0, rc=f"{d.get('CH0_gpu_row_conflict',0):.0f}/{d.get('CH0_gpu_q_count',0):.0f}")
    for des, nm in (("merge", "GPU merge + SPU"), ("f4dru", "GPU 4-step + SPU + DRU")):
        d = parse(f"{W}/{org}_{cell}_{des}.out"); r = R[des]
        pr(nm, d, d["pipe0_finish_max"], d["pim_done_cycles"], f"SPU x{r['spu_slow']:.3f}  NTT x{r['ntt_slow']:.3f}",
           pq=f"{d['CH0_pim_q_mean']*tck - pq0:+.0f} ns", rc=f"{d.get('CH0_gpu_row_conflict',0):.0f}/{d.get('CH0_gpu_q_count',0):.0f}")
