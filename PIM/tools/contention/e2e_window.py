#!/usr/bin/env python3
"""End-to-end PCG^2 per Fig. 8 cell, contention measured in a steady-state WINDOW.

e2e_cosim.py shrank every NTT kernel by the cell's time scale (as small as ~1/500 on
L40S), so fixed per-kernel DRAM latency inflated the simulated NTT lane 1.5-2.3x.
Here kernels keep a size the standalone validation reproduces (>= ~20k CK each): the
window is one run of the SPU trace, and the NTT pipeline inside it is sized so that
NTT-alone / SPU-alone matches the cell's real lane ratio r. The window gives the two
lanes' co-run slowdowns; the cell is then

   PCG^2 = max( SPU lane x SPU slowdown, NTT lane x NTT slowdown, NIC )

with the SPU lane from the paper's anchor (scaled by the SPU clock) and the NTT lane
(incl. the DPF->NTT gather) from the simulator's standalone per-mul time.
Usage: e2e_window.py --sim BIN --out DIR [--clock nom|1.0|0.5] [--gather --spu-presum]
"""
import argparse, concurrent.futures as cf, io, contextlib, json, math, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "e2e"))
from ntt_jobs import DEV, base, mul_ms, ntt_table, cheddar_table, ole_table, muls_per_output, OLE, phases, write_jobs, SPU_G_ROW
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, LOGN, alpha_bw, gm
    from fig8 import BETA
PIM = os.path.dirname(os.path.dirname(HERE))
GEN = os.path.join(PIM, "tools", "gen_dpf_tree_trace.py")
YAML = {"l40s": os.path.join(PIM, "sim/test/gddr6_dpf_24ch.yaml"), "b200": os.path.join(PIM, "sim/test/hbm3e_dpf_b200d.yaml")}
FNOM = {"l40s": 2.25, "b200": 2.0}; MACH = {"l40s": "L40S", "b200": "B200"}
NSPU = {o: 8 * d["ch"] for o, d in DEV.items()}          # 8 SPUs per channel (192 on L40S, 2048 on B200)
KERNEL_MIN_CK = 20000


def parse(p):
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9.eE+]+)", open(p).read(), re.M)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--clock", default="nom"); ap.add_argument("--orgs", default="l40s,b200")
    ap.add_argument("--designs", default="merge,f4dru"); ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--gather", action="store_true"); ap.add_argument("--spu-presum", action="store_true")
    ap.add_argument("--spu-extra", default="", help="extra arguments for gen_dpf_tree_trace.py (e.g. '--pass2-placement trickle', '--skip-pass2')")
    ap.add_argument("--spu-tag", default="", help="tag appended to the SPU trace name when --spu-extra changes it")
    ap.add_argument("--ntt-release", action="store_true",
                    help="pipeline dependency: the co-run NTT jobs are released in win_inst rounds at the SPU rounds' boundaries (round r's jobs "
                         "start at (r+1)/win_inst of the SPU window, when its polynomials exist) instead of all at cycle 0; the standalone lane stays unreleased. Adds "
                         "tail_ms = NTT work left after the SPUs finish (real ms) to every row")
    ap.add_argument("--ntt-window", type=int, default=0,
                    help="reads in flight per channel for every NTT-lane phase (0 = each design's fitted pattern): throttles the SMs' "
                         "DRAM demand, e.g. to run the NTT at the SPUs' production rate instead of full burst")
    ap.add_argument("--f4c-sm-bound", action="store_true",
                    help="four-step SM kernel time = min(measured four-step SM lane, measured Cheddar): same kernels, launches, "
                         "bytes and butterflies, so Cheddar's time bounds it; the SM transposes keep their measured extra")
    ap.add_argument("--handoff", action="store_true",
                    help="DPF -> NTT handoff at the SPUs' rows: the phase that reads g (gather, the DRU's fused gather, or the ole "
                         "column pass over g) reads rows from SPU_G_ROW in every bank, where the SPUs' pass 2 wrote g")
    ap.add_argument("--gather-dru", action="store_true",
                    help="designs with DRU transposes: the DRU's first transpose reads the SPUs' bank-local output directly "
                         "(the t/m pre-summed vectors, summed on the way) and writes the transposed polynomial, so the SM "
                         "gather phase disappears; SM-only designs keep the SM gather")
    ap.add_argument("--spu-lane", default="paper", choices=["paper", "sim"],
                    help="paper: Fig. 8 anchor (718,660 CK, --reduce mau: no per-leaf H' on the SPU) scaled by T(clock)/T(nominal); "
                         "sim: this run's SPU trace time (--reduce chacha: expand + H' + mod-p on the SPU) x ceil(I/NSPU) x leaves/instance")
    ap.add_argument("--reduce", default="chacha", choices=["chacha", "mau", "fused"],
                    help="SPU trace leaf conversion: chacha = expand + per-leaf ChaCha8 hash H' + mod-p on the SPU (Sec. IV); "
                         "mau = the paper's Fig. 8 anchor mode (conversion as a channel-level column stream, H' not charged)")
    ap.add_argument("--alu-mult", type=float, default=1.0,
                    help="SPU datapath width multiplier (2 = two VADD/VXORL pairs, or 16 lanes): divides the per-op cycle counts")
    ap.add_argument("--occupancy", default="ceil", choices=["ceil", "fractional", "range"],
                    help="instances per SPU: ceil(I/NSPU) (makespan of the static assignment) or I/NSPU (the paper's lane model); "
                         "range: every SPU holds the same leaf range (subtrees of 2^win-depth leaves, both halves of the "
                         "negacyclic fold) of all t instances of its polynomials, so the sum over the t instances is bank-local "
                         "(presum m = t) and the static assignment is ceil at subtree granularity (all SPUs busy)")
    ap.add_argument("--dpf-split", action="store_true",
                    help="scheduler also assigns DPF instances to the SMs while they are idle: the SM lane runs the NTTs "
                         "plus a (1-x) share of the DPF (the measured GPU DPF kernel), the SPUs the x share, x chosen to "
                         "balance the two lanes; both lanes' co-run slowdowns come from the window")
    ap.add_argument("--dru-clock", default="nom", help="DRU clock in GHz or 'nom' (= DRAM clock): the DRU tap takes 2*f_DRAM/f_DRU bus cycles per column")
    ap.add_argument("--win-inst", type=int, default=4, help="window: DPF instances per SPU in the simulated SPU trace")
    ap.add_argument("--win-depth", type=int, default=12, help="window: GGM tree depth of each simulated instance (2^depth leaves)")
    ap.add_argument("--sub-batch", action="store_true",
                    help="NTT lane as 2c^2/g jobs of g polynomials (g in 4, 16, 64 up to c^2, measured per-mul time at batch g), "
                         "each design at the g of its fastest co-run lane: the DRU transposes one sub-batch while the SMs run "
                         "another; default: the lane is n identical jobs sized only for kernel length (n = 1 serialises the DRU)")
    ap.add_argument("--unit-mult", type=int, default=8, help="channel unit (oleu/olepu): modular multipliers per channel")
    ap.add_argument("--unit-ghz", type=float, default=1.0, help="channel unit clock, GHz")
    ap.add_argument("--dru-stream", action="store_true",
                    help="steady state of a continuous NTT stream (every cell's and query's NTTs back to back): the DRU's "
                         "transposes run as their own pipeline beside the SM kernels (always another sub-batch to transpose), "
                         "sharing the banks with them and the SPUs; the lane ends when both pipelines have finished. "
                         "Kernel sizes stay as in the default window")
    ap.add_argument("--fpu-gate", default="true", help="fpu_gate_issue: true = campaign (serial PU, no load/compute overlap), false = LSU overlap as described in Sec. IV")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import ntt_jobs; ntt_jobs.F4C_SM_BOUND = a.f4c_sm_bound; ntt_jobs.UNIT_MULT = a.unit_mult; ntt_jobs.UNIT_GHZ = a.unit_ghz
    eoc = os.path.join(a.out, "eoc.trace"); open(eoc, "w").write("AiM EOC\n")
    ctrl = ["-p", f"MemorySystem.Controller.fpu_gate_issue={a.fpu_gate}", "-p", "MemorySystem.Controller.wr_max_age=1000",
            "-p", "MemorySystem.Controller.pim_row_wait=0", "-p", "MemorySystem.Controller.class_priority=0,1,3,2",
            "-p", "MemorySystem.Controller.class_min_run=64", "-p", "MemorySystem.DRAM.org.channel=2", "-p", "Frontend.issue_width=2"]
    def ctrl_for(org):
        fd = FNOM[base(org)] if a.dru_clock == "nom" else float(a.dru_clock)
        slot = max(2, int(round(2 * FNOM[base(org)] / fd)))
        return ctrl + ["-p", f"MemorySystem.Controller.dru_bus_slot={slot}"]

    def spu_trace(org, f, tag):
        tr = os.path.join(a.out, f"spu_{org}_{a.reduce}_x{a.alu_mult:g}_i{a.win_inst}_n{a.win_depth}{'_' + a.spu_tag if a.spu_tag else ''}_{tag}.trace"); o = tr[:-6] + ".out"
        if not os.path.exists(tr):
            subprocess.run([sys.executable, GEN, "-n", str(a.win_depth), "-C", "2", "-P", "8", "-I", str(16 * a.win_inst), "--mode", "instances", "--seed-bits", "128",
                            "--reread", "--broadcast", "--cl", str(round(155 * FNOM[base(org)] / f / a.alu_mult)), "--reduce", a.reduce,
                            "--convert-cl", str(round(465 * FNOM[base(org)] / f / a.alu_mult)),
                            "--modmul-cl", str(round(32 * FNOM[base(org)] / f / a.alu_mult)),
                            "--add64m-cl", str(max(1, round(4 * FNOM[base(org)] / f / a.alu_mult))), "-o", tr] + a.spu_extra.split(), check=True, stderr=subprocess.DEVNULL)
        if not (os.path.exists(o) and "pim_done_cycles" in open(o).read()):
            with open(o, "w") as fo: subprocess.run([a.sim, "-f", YAML[base(org)], "-t", tr] + ctrl_for(org), stdout=fo, stderr=subprocess.STDOUT)
        return tr, parse(o)["pim_done_cycles"]

    orgs = a.orgs.split(","); TR, TS, CLK = {}, {}, {}
    for org in orgs:
        f = FNOM[base(org)] if a.clock == "nom" else float(a.clock)
        TR[org], TS[org] = spu_trace(org, f, "clk")
        CLK[org] = TS[org] / spu_trace(org, FNOM[base(org)], "nom")[1]          # SPU lane scale vs the anchor clock

    def build(org, d, lg, c, t, r, g=None):
        """kernel pipeline for the window; returns (jobs file, n muls, scale). g: sub-batch (n = 2c^2/g jobs)"""
        tm = mul_ms(org, d, lg, g or c * c)
        mul_ck = tm * 1e6 / DEV[org]["tck"]                                  # real per-mul CK
        n = 2 * c * c // g if g else max(1, int(r * TS[org] / (10 * KERNEL_MIN_CK)))
        if a.ntt_release: n = max(n, a.win_inst)      # at least one job per SPU round (kernels may fall below KERNEL_MIN_CK: the tail is an upper bound)
        s = r * TS[org] / (n * mul_ck)
        ph = phases(org, d, lg, g or c * c, scale=s, window=a.ntt_window or None, handoff=a.handoff)
        hr = (SPU_G_ROW,) if a.handoff else ()   # handoff: the read side of the g-reading phase starts at the SPUs' g rows
        if a.gather and d not in OLE:   # ole: the column pass reads the SPUs' output directly (no gather pass)
            m = (t if a.occupancy == "range" else max(1, min(t, (c * c * t * t) // NSPU[org]))) if a.spu_presum else 1
            gb = int(round(8 * (2 * (1 << lg) * t) / 2 / m / DEV[org]["ch"] * s))
            if a.gather_dru and any(p[0] == 1 for p in ph):
                # DPF -> NTT layout change fused into the DRU's first transpose: its read side is the gather (8N t/m B,
                # bank-local SPU output) instead of one 8N-B polynomial; its write side (8N B, transposed) is unchanged
                i = next(k for k, p in enumerate(ph) if p[0] == 1); p = ph[i]
                wr = int(round(p[2] * (1 - p[3])))
                ph = ph[:i] + [(p[0], p[1], gb + wr, gb / (gb + wr)) + p[4:9] + hr] + ph[i + 1:]
            else:
                ph = [(0, 1, gb, 1.0, 48, 0, 16, 20000, 0) + hr] + ph
        jp = os.path.join(a.out, f"{org}_{c}_{t}_{lg}_{d}{f'_g{g}' if g else ''}.jobs")
        if a.dru_stream and any(p[0] == 1 for p in ph):
            write_jobs(jp, [(n, 0, [p for p in ph if p[0] == 0]), (n, 0, [(0,) + p[1:] for p in ph if p[0] == 1])])
        else:
            write_jobs(jp, [(n, 0, ph)])                              # NTT-only pipeline (the NTT lane)
        if a.ntt_release:   # the co-run copy: jobs released per SPU round (round r of win_inst at r/win_inst of the SPU window)
            R = a.win_inst; per = [n // R + (r < n % R) for r in range(R)]
            write_jobs(jp[:-5] + "_rel.jobs", [(k, int((r + 1) * TS[org] / R), [p for p in ph if p[0] == 0]) for r, k in enumerate(per) if k > 0])   # round r's polynomials exist at the END of round r
        if a.dpf_split:
            # SM lane = NTT kernels + a (1-x0) share of the DPF (measured GPU expand+convert kernel),
            # sequential on the SMs: appended to every job as one phase. ~88 B/leaf of DRAM traffic
            # (tree levels 48 B, two H' passes + scatter 40 B), 55% reads, issued at the kernel's own
            # bandwidth (gap-throttled: L40S ~38%, B200 ~30% of peak); compute floor = its measured time.
            L = L_(MACH[base(org)])[(c, t, lg)]; spu_ms = SPU_MS[(org, c, t, lg)]
            Tg, Tn0 = L["gpu_dpf_g"], tm * muls_per_output(d, c)
            x0 = min(1.0, (Tn0 + Tg) / (spu_ms + Tg))
            share_ms = (1 - x0) * Tg
            dpf_ck = share_ms / spu_ms * TS[org]                       # window CK for the SM's DPF share
            leaves = 2 * (1 << lg) * c * c * t
            dpf_bytes = 88 * leaves * (1 - x0) / DEV[org]["ch"] * (dpf_ck / (share_ms * 1e6 / DEV[org]["tck"])) if share_ms > 0 else 0
            gap = {"l40s": 5, "b200": 6}[base(org)]
            ph2 = ph + ([(0, 1, int(dpf_bytes / n), 0.55, 48, gap, 16, 24576, int(dpf_ck / n))] if dpf_bytes > 0 else [])
            write_jobs(jp[:-5] + "_split.jobs", [(n, 0, ph2)])
            X0[(org, c, t, lg, d)] = x0
        return jp, n, s

    def sim(tag, trace, jp, org):
        out = os.path.join(a.out, tag + ".out")
        if not (os.path.exists(out) and "memory_system_cycles" in open(out).read()):
            with open(out, "w") as fo:
                subprocess.run([a.sim, "-f", YAML[base(org)], "-t", trace, "-p", f"MemorySystem.host_jobs={jp}"] + ctrl_for(org), stdout=fo, stderr=subprocess.STDOUT)
        return parse(out)

    cells, work = {}, []
    SPU_MS, X0 = {}, {}
    for org in orgs:
        LN = L_(MACH[base(org)])
        for (c, t) in CFG:
            for lg in LOGN:
                L = LN.get((c, t, lg))
                if not L or (lg, c * c) not in ntt_table(org): continue
                if a.spu_lane == "sim":
                    I = c * c * t * t; n_leaf = 2 * (1 << lg) // t
                    if a.occupancy == "range":   # subtrees of 2^win_depth leaves per SPU, ceil
                        nsub = math.ceil(I * n_leaf / NSPU[org] / (1 << a.win_depth))
                    else:
                        per_spu = math.ceil(I / NSPU[org]) if a.occupancy == "ceil" else max(1.0, I / NSPU[org])
                        nsub = per_spu * n_leaf / (1 << a.win_depth)
                    spu_ms = TS[org] / a.win_inst * DEV[org]["tck"] / 1e6 * nsub   # window: win_inst subtrees of 2^win_depth leaves per SPU
                else:
                    spu_ms = L["spu"] * CLK[org]
                SPU_MS[(org, c, t, lg)] = spu_ms
                for d, g in [(d, g) for d in a.designs.split(",")
                             for g in ([g for g in (4, 16, 64) if g <= c * c and (lg, g) in ntt_table(org)] if a.sub_batch else [None])]:
                    tm = mul_ms(org, d, lg, g or c * c)
                    if tm >= 1e8: continue                                        # design not measured on this card (no row in its csv)
                    r = tm * muls_per_output(d, c) / spu_ms                      # real lane ratio (pre-gather); ole: tm is per output
                    jp, n, s = build(org, d, lg, c, t, r, g)
                    d = f"{d}_g{g}" if g else d
                    cells[(org, c, t, lg, d)] = dict(L=L, spu_ms=spu_ms, n=n, s=s, tm=tm, g=g)
                    work += [(f"{org}_{c}_{t}_{lg}_{d}_alone", eoc, jp, org)]
                    if a.dpf_split:
                        js = jp[:-5] + "_split.jobs"
                        work += [(f"{org}_{c}_{t}_{lg}_{d}", TR[org], js, org), (f"{org}_{c}_{t}_{lg}_{d}_splitalone", eoc, js, org)]
                    else:
                        work += [(f"{org}_{c}_{t}_{lg}_{d}", TR[org], jp[:-5] + "_rel.jobs" if a.ntt_release else jp, org)]
    R = {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for (tag, *_), d in zip(work, ex.map(lambda w: sim(*w), work)): R[tag] = d

    fin = lambda o: max(v for k, v in o.items() if re.fullmatch(r"pipe\d+_finish_max", k))   # NTT lane: all pipelines (SM, DRU, per-round releases)
    rows = []
    for (org, c, t, lg, d), cd in cells.items():
        tck = DEV[org]["tck"]; x = R[f"{org}_{c}_{t}_{lg}_{d}"]; y = R[f"{org}_{c}_{t}_{lg}_{d}_alone"]
        K = muls_per_output(d.rsplit("_g", 1)[0] if "_g" in d else d, c)
        ntt_lane = fin(y) / cd["s"] / cd["n"] * K * tck / 1e6   # real ms, incl. gather
        spu_slow = x["pim_done_cycles"] / TS[org]
        tail_ms = max(0.0, fin(x) - x["pim_done_cycles"]) / cd["s"] / cd["n"] * K * tck / 1e6 if a.ntt_release else 0.0   # NTT work after the SPUs finish
        ntt_slow = fin(x) / fin(R[f"{org}_{c}_{t}_{lg}_{d}_splitalone"] if a.dpf_split else y)
        mm = ntt_table(org)[(lg, c * c)][0]
        mc = cheddar_table(org).get((lg, c * c), (float("nan"),))[0]          # Cheddar's NTT as the GPU baseline's NTT
        xs = 1.0; sm_slow = ntt_slow
        for tier in ("fast", "slow"):
            nic = (cd["L"]["n"] + 2) * alpha_bw(c, t, BETA[tier])
            gpu = cd["L"]["gpu_dpf_g"] + mm * 2 * c * c
            gpu_ch = cd["L"]["gpu_dpf_g"] + mc * 2 * c * c
            ot = ole_table(org).get((lg, c * c), ((float("nan"), 0),) * 3)          # the GPU baseline with the PCG-output algorithm (ole / olep, same kernels)
            gpu_ole = cd["L"]["gpu_dpf_g"] + ot[0][0]; gpu_olep = cd["L"]["gpu_dpf_g"] + ot[2][0]
            if a.dpf_split:
                # window: SM lane = NTT + (1-x0) DPF share; its co-run slowdown applies to both parts
                x0 = X0[(org, c, t, lg, d)]; Tg = cd["L"]["gpu_dpf_g"]; Ts = cd["spu_ms"]
                ys = R[f"{org}_{c}_{t}_{lg}_{d}_splitalone"]
                sm_slow = x["pipe0_finish_max"] / ys["pipe0_finish_max"]
                # rebalance with the measured slowdowns: x Ts s_spu = (Tn + (1-x) Tg) s_sm
                xs = min(1.0, (ntt_lane + Tg) * sm_slow / (Ts * spu_slow + Tg * sm_slow))
                pcg = max(xs * Ts * spu_slow, (ntt_lane + (1 - xs) * Tg) * sm_slow, nic)
            else:
                pcg = max(cd["spu_ms"] * spu_slow, ntt_lane * ntt_slow, nic)
            rows.append(dict(org=org, c=c, t=t, logN=lg, tier=tier, design=d, spu_ms=cd["spu_ms"], ntt_ms=ntt_lane,
                             ntt_meas_ms=cd["tm"] * K, spu_slow=spu_slow, ntt_slow=ntt_slow, nic_ms=nic,
                             pcg_ms=pcg, pcg_nocont_ms=max(cd["spu_ms"], ntt_lane, nic), gpu_ms=gpu,
                             spu_share=xs, sm_slow=sm_slow,
                             base_serial_ms=gpu + nic, base_overlap_ms=max(gpu, nic),
                             gpu_cheddar_ms=gpu_ch, base_overlap_cheddar_ms=max(gpu_ch, nic),
                             gpu_ole_ms=gpu_ole, base_overlap_ole_ms=max(gpu_ole, nic), gpu_olep_ms=gpu_olep, tail_ms=tail_ms,
                             base_paper_ms=cd["L"]["gpu_dpf_g"] + cd["L"]["ntt_dev"] + nic, g=cd["g"], n=cd["n"]))
    if a.sub_batch:   # each design at the sub-batch of its fastest co-run NTT lane
        best = {}
        for r in rows:
            k = (r["org"], r["c"], r["t"], r["logN"], r["tier"], r["design"].rsplit("_g", 1)[0])
            if k not in best or r["ntt_ms"] * r["ntt_slow"] < best[k]["ntt_ms"] * best[k]["ntt_slow"]: best[k] = r
        rows = [dict(r, design=k[-1]) for k, r in best.items()]
    json.dump(rows, open(os.path.join(a.out, "e2e.json"), "w"), indent=1)
    print(f"clock={a.clock}  (best (c,t) / geomean over all cells; GPU baseline = DPF + merge NTT, network co-scheduled)")
    for org in orgs:
        for tier in ("fast", "slow"):
            for d in a.designs.split(","):
                rr = [r for r in rows if r["org"] == org and r["tier"] == tier and r["design"] == d]
                if not rr: continue                                           # design not measured on this card
                def hb(key):
                    per = {}
                    for r in rr: per.setdefault((r["c"], r["t"]), []).append(r[key] / r["pcg_ms"])
                    allv = [r[key] / r["pcg_ms"] for r in rr]
                    return max(gm(v) for v in per.values()), gm(allv)
                nc = [r["base_serial_ms"] / r["pcg_nocont_ms"] for r in rr]
                ntt_err = gm(r["ntt_ms"] / r["ntt_meas_ms"] for r in rr)
                print(f"  {MACH[base(org)]} {tier} {d:6s}: serial-net base {hb('base_serial_ms')[0]:.2f}/{hb('base_serial_ms')[1]:.2f}"
                      f" | overlap-net base {hb('base_overlap_ms')[0]:.2f}/{hb('base_overlap_ms')[1]:.2f}"
                      f" | paper base {hb('base_paper_ms')[0]:.2f}/{hb('base_paper_ms')[1]:.2f}"
                      f" | no-cont geo {gm(nc):.2f} | SPU slow {min(r['spu_slow'] for r in rr):.2f}-{max(r['spu_slow'] for r in rr):.2f}"
                      f" NTT slow {min(r['ntt_slow'] for r in rr):.2f}-{max(r['ntt_slow'] for r in rr):.2f} | sim NTT lane/measured {ntt_err:.2f}")


if __name__ == "__main__":
    main()
