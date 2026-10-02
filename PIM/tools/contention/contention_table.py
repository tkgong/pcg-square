#!/usr/bin/env python3
"""Per-channel contention table (Q4): bus occupancy per unit, GPU queueing / read latency, SPU queueing
vs alone, row conflicts and the measured slowdowns, for GPU-only / GPU+SPU / GPU+SPU+DRU, from a
e2e_window.py run directory.  Usage: contention_table.py RUN_DIR"""
import re, json, sys
def parse(p): return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}
TCK = {"b200": 0.5, "l40s": 0.444}
def row(d, tck, T=None):
    T = T or d["memory_system_cycles"]
    gpu_rd, gpu_wr = d.get("CH0_gpu_rd_cols", 0), d.get("CH0_gpu_wr_cols", 0)
    nrd, nwr = d.get("CH0_num_RD_commands", 0), d.get("CH0_num_WR_commands", 0)
    dru_cols = max(0, nrd + nwr - gpu_rd - gpu_wr)
    gpu_bus = d.get("CH0_gpu_bus_cycles", 0); pim_bus = d.get("CH0_pim_allbank_slot_cycles", 0); dru_bus = dru_cols * 2
    return dict(T=T, gpu=gpu_bus / T, pim=pim_bus / T, dru=dru_bus / T, tot=(gpu_bus + pim_bus + dru_bus) / T,
                gq=(d.get("CH0_gpu_q_mean", 0) * tck, d.get("CH0_gpu_q_p50", 0) * tck, d.get("CH0_gpu_q_p95", 0) * tck, d.get("CH0_gpu_q_p99", 0) * tck),
                grl=(d.get("CH0_gpu_rdlat_mean", 0) * tck, d.get("CH0_gpu_rdlat_p99", 0) * tck),
                pq=d.get("CH0_pim_q_mean", 0) * tck, pim_conf=d.get("CH0_pim_row_conflict", 0) + d.get("CH0_pim_row_miss", 0), gpu_conf=d.get("CH0_gpu_row_conflict", 0))
for org, cell, lab in (("b200", "4_16_24", "B200 (4,16) 2^24"), ("l40s", "4_16_22", "L40S (4,16) 2^22")):
    W = sys.argv[1]; tck = TCK[org]
    R = {r["design"]: r for r in json.load(open(f"{W}/e2e.json")) if r["org"] == org and r["tier"] == "fast" and f'{r["c"]}_{r["t"]}_{r["logN"]}' == cell}
    spu = parse(f"{W}/spu_{org}_clk.out") if org == "b200" else parse(f"{W}/spu_{org}_clk.out")
    print(f"\n{lab}  (per channel; bus = fraction of DRAM cycles the data bus is occupied; queue = ns from arrival to first command)")
    print(f"  {'config':26s} {'bus GPU':>8s} {'bus SPU':>8s} {'bus DRU':>8s} {'total':>6s} | {'GPU q mean/p50/p95/p99 ns':>28s} | {'GPU rdlat mean/p99':>19s} | {'SPU q mean':>10s} | slowdown")
    def pr(name, d, T=None, sl=""):
        r = row(d, tck, T)
        print(f"  {name:26s} {r['gpu']:8.2f} {r['pim']:8.2f} {r['dru']:8.2f} {r['tot']:6.2f} | {r['gq'][0]:6.0f}/{r['gq'][1]:5.0f}/{r['gq'][2]:5.0f}/{r['gq'][3]:6.0f}      | {r['grl'][0]:8.0f}/{r['grl'][1]:8.0f}    | {r['pq']:10.0f} | {sl}")
    pr("SPU only", spu, spu["pim_done_cycles"], "")
    for des, nm in (("merge", "GPU NTT (merge) only"), ("f4dru", "GPU NTT (4-step+DRU) only")):
        d = parse(f"{W}/{org}_{cell}_{des}_alone.out"); pr(nm, d, d["pipe0_finish_max"])
    for des, nm in (("merge", "GPU merge + SPU"), ("f4dru", "GPU 4-step + SPU + DRU")):
        d = parse(f"{W}/{org}_{cell}_{des}.out"); r = R[des]
        pr(nm, d, d["memory_system_cycles"], f"SPU x{r['spu_slow']:.3f}  NTT x{r['ntt_slow']:.3f}")
