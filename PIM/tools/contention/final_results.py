#!/usr/bin/env python3
"""The one accounting used for every reported number:
  GPU baseline = the submission's GPU implementation (DPF with the two-pass leaf conversion) + GPU-NTT merge
                 (SOTA NTT), with the network co-scheduled: T_base = max(T_GPU, T_net)
  PCG^2        = final design: expansion on the SPUs (SPU at the DRAM clock, H' on the SPU, LSU overlap), NTT =
                 four-step on GPU-NTT kernels with the transposes on the DRU on B200; on L40S the runtime selects
                 merge (the DRU design moves 43% more bytes and GDDR6 is the limiter: 0.90x of merge there);
                 PCG^2 = max(SPU lane, NTT lane, network) with the co-simulated contention (22-multiply window).
  Table IV     = GPU baseline / +SPU (the same four-step with the transposes on the SMs, serial) / +SPU+DRU (serial)
                 / +SPU+co-sched / full; the DRU rows are the ablation inside the final design (e2e_f4sm.json).
Writes headline (Fig. 8, all SPU clocks, single-ALU), Table IV, the mechanism decomposition, the DRU
increment and the window-size check into OUT_DIR.   Usage: final_results.py WIN22_DIR WINDOW_CHECK_DIR OUT_DIR"""
import io, contextlib, json, os, sys
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, gm
    from lanes import alpha_bw
    from fig8 import BETA
W, WC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True)
LN = {"L40S": L_("L40S"), "B200": L_("B200")}
DES = {"L40S": ("l40s", "merge"), "B200": ("b200", "f4dru")}
PAPER = {"fast": {"L40S": ([2.43, 2.62, 3.19, 2.21, 2.10], 3.19, 2.54), "B200": ([5.01, 9.50, 6.86, 7.92, 7.78], 9.49, 7.25)},
         "slow": {"L40S": ([2.49, 2.98, 3.38, 2.38, 1.69], 3.38, 2.61), "B200": ([5.26, 4.90, 5.93, 2.11, 1.60], 5.93, 3.49)}}
TIERS = (("fast", "40 Gbps"), ("slow", "400 Mbps"))


def rows(path, mach, des=None, tier="fast"):
    org, d = DES[mach]; d = des or d
    return [r for r in json.load(open(path)) if r["org"] == org and r["design"] == d and r["tier"] == tier]


def speed(r, mode="full", nic=None):
    nic = r["nic_ms"] if nic is None else nic
    base = max(r["gpu_ms"], nic)
    p = {"full": max(r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"], nic), "nttovl": max(r["spu_ms"] * r["spu_slow"], r["ntt_ms"] * r["ntt_slow"]) + nic,
         "serial": r["spu_ms"] + r["ntt_ms"] + nic, "nocont": max(r["spu_ms"], r["ntt_ms"], nic)}[mode]
    return base / p


def panel(path, mach, tier, mode="full"):
    per, allv = [], []
    for (c, t) in CFG:
        v = [speed(r, mode) for r in rows(path, mach, tier=tier) if r["c"] == c and r["t"] == t]
        per.append(gm(v)); allv += v
    return per, max(per), gm(allv)


L = ["Accounting: GPU baseline = two-pass-H' DPF + merge NTT, network co-scheduled (max); PCG^2 = final design with contention; up to / geomean.", ""]
L.append("Fig. 8 (SPU = DRAM clock):")
for tier, lab in TIERS:
    for mach in ("L40S", "B200"):
        per, u, g = panel(f"{W}/e2e_nom.json", mach, tier); pp, pu, pg = PAPER[tier][mach]
        L.append(f"  {lab:8s} {mach}: " + "  ".join(f"({c},{t}) {x:.2f}" for (c, t), x in zip(CFG, per)) + f"  | up to {u:.2f} geomean {g:.2f}  | paper {pu:.2f}/{pg:.2f}  diff {u/pu-1:+.0%}/{g/pg-1:+.0%}")
L.append("\nSPU clock sensitivity and single-ALU SPU (up to / geomean):")
for lab, f in (("DRAM clock", "e2e_nom.json"), ("1.75 GHz", "e2e_1.75.json"), ("1.5 GHz", "e2e_1.5.json"), ("1 GHz", "e2e_1.0.json"), ("single ALU @ DRAM clock", "e2e_1alu.json")):
    L.append(f"  {lab:24s} " + " | ".join(f"{mach} {tl} {panel(f'{W}/{f}', mach, tier)[1]:.2f}/{panel(f'{W}/{f}', mach, tier)[2]:.2f}" for mach in ("L40S", "B200") for tier, tl in TIERS))
L.append("\nMechanism decomposition (geomean): expansion on the SPUs, everything serial -> + NTT overlap -> + network overlap (= PCG^2):")
for tier, lab in TIERS:
    for mach in ("L40S", "B200"):
        L.append(f"  {lab:8s} {mach}: " + " -> ".join(f"{panel(f'{W}/e2e_nom.json', mach, tier, m)[2]:.2f}" for m in ("serial", "nttovl", "full")) + f"   (no contention: {panel(f'{W}/e2e_nom.json', mach, tier, 'nocont')[2]:.2f})")
L.append("\nTable IV (geomean ms over the suite; alpha = 500 us per exchange as in the paper, then the Fig. 8 tiers):")
for mach in ("L40S", "B200"):
    org = DES[mach][0]
    M = {(r["c"], r["t"], r["logN"]): r for r in rows(f"{W}/e2e_f4sm.json", mach, "f4sm")}; F = {(r["c"], r["t"], r["logN"]): r for r in rows(f"{W}/e2e_f4sm.json", mach, "f4dru")}
    for alab, amode in (("alpha = 500 us", 500), ("40 Gbps", "fast"), ("400 Mbps", "slow")):
        T = {k: [] for k in ("GPU baseline", "+SPU", "+SPU+DRU", "+SPU+co-sched", "full (DRU on)")}
        for k, m in M.items():
            f = F[k]; n = LN[mach][k]["n"]; nic = (n + 2) * (0.5 if amode == 500 else alpha_bw(k[0], k[1], BETA[amode]))
            T["GPU baseline"].append(max(m["gpu_ms"], nic)); T["+SPU"].append(m["spu_ms"] + m["ntt_ms"] + nic); T["+SPU+DRU"].append(f["spu_ms"] + f["ntt_ms"] + nic)
            T["+SPU+co-sched"].append(max(m["spu_ms"] * m["spu_slow"], m["ntt_ms"] * m["ntt_slow"], nic)); T["full (DRU on)"].append(max(f["spu_ms"] * f["spu_slow"], f["ntt_ms"] * f["ntt_slow"], nic))
        # Table IV rows: "+SPU" keeps the baseline's merge NTT (HEonGPU), "+DRU" switches to the four-step with the DRU
        # (B200; on L40S the runtime keeps merge, so the DRU rows equal the merge rows)
        MM = {(r["c"], r["t"], r["logN"]): r for r in rows(f"{W}/e2e_nom.json", mach, "merge")}
        nic_k = lambda k: (LN[mach][k]["n"] + 2) * (0.5 if amode == 500 else alpha_bw(k[0], k[1], BETA[amode]))
        T["+SPU (merge NTT)"] = [MM[k]["spu_ms"] + MM[k]["ntt_ms"] + nic_k(k) for k in M]
        T["+SPU+co-sched (merge NTT)"] = [max(MM[k]["spu_ms"] * MM[k]["spu_slow"], MM[k]["ntt_ms"] * MM[k]["ntt_slow"], nic_k(k)) for k in M]
        g = {k: gm(v) for k, v in T.items()}
        L.append(f"  {mach} {alab:14s}: " + "; ".join(f"{k} {v:.1f} ({g['GPU baseline']/v:.2f}x)" for k, v in g.items())
                 + f" | increments: SPU {g['GPU baseline']/g['+SPU']:.2f}, DRU serial {g['+SPU']/g['+SPU+DRU']:.3f}, co-scheduling {g['+SPU']/g['+SPU+co-sched']:.2f}, DRU co-scheduled {g['+SPU+co-sched']/g['full (DRU on)']:.3f}"
                 + (f"\n       Table IV rows: GPU baseline {g['GPU baseline']:.1f} | +SPU {g['+SPU (merge NTT)']:.1f} ({g['GPU baseline']/g['+SPU (merge NTT)']:.3f}) | +SPU+DRU {g['+SPU+DRU']:.1f} ({g['+SPU (merge NTT)']/g['+SPU+DRU']:.3f}) | +SPU+co-sched {g['+SPU+co-sched (merge NTT)']:.1f} ({g['+SPU (merge NTT)']/g['+SPU+co-sched (merge NTT)']:.3f}) | full {g['full (DRU on)']:.1f} ({g['+SPU+co-sched (merge NTT)']/g['full (DRU on)']:.3f})" if mach == "B200" else
                    f"\n       Table IV rows: GPU baseline {g['GPU baseline']:.1f} | +SPU {g['+SPU (merge NTT)']:.1f} ({g['GPU baseline']/g['+SPU (merge NTT)']:.3f}) | +SPU+DRU = +SPU (merge kept) | +SPU+co-sched {g['+SPU+co-sched (merge NTT)']:.1f} ({g['+SPU (merge NTT)']/g['+SPU+co-sched (merge NTT)']:.3f}) | full = +SPU+co-sched"))
L.append("\nDRU ablation inside the final design (transposes on the SMs vs on the DRU), 40 Gbps, contention:")
for mach in ("L40S", "B200"):
    M = {(r["c"], r["t"], r["logN"]): r for r in rows(f"{W}/e2e_f4sm.json", mach, "f4sm")}; F = {(r["c"], r["t"], r["logN"]): r for r in rows(f"{W}/e2e_f4sm.json", mach, "f4dru")}
    g = [M[k]["pcg_ms"] / F[k]["pcg_ms"] for k in M]; nb = [k for k in M if M[k]["ntt_ms"] * M[k]["ntt_slow"] > M[k]["spu_ms"] * M[k]["spu_slow"]]
    L.append(f"  {mach}: geomean {gm(g):.3f} (range {min(g):.2f}-{max(g):.2f}); NTT-bound cells {len(nb)}/{len(M)}: {gm(M[k]['pcg_ms']/F[k]['pcg_ms'] for k in nb) if nb else float('nan'):.3f}")
L.append("\nWindow-size check (headline under this accounting; i4 = 4 DPF instances per SPU, i8/i16 = 2x/4x, n13 = 2x leaves per instance; i16 = the main run):")
for wlab, wd in (("i4_n12", f"{WC}/i4_n12/e2e.json"), ("i8_n12", f"{WC}/i8_n12/e2e.json"), ("i16_n12 (main)", f"{W}/e2e_nom.json"), ("i4_n13", f"{WC}/i4_n13/e2e.json")):
    if not os.path.exists(wd): continue
    L.append(f"  {wlab:16s} " + " | ".join(f"{mach} {tl} {panel(wd, mach, tier)[1]:.2f}/{panel(wd, mach, tier)[2]:.2f}" for mach in ("L40S", "B200") for tier, tl in TIERS))
open(os.path.join(OUT, "results.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
