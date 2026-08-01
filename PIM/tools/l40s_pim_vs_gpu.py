#!/usr/bin/env python3
"""L40S: PIM+GPU co-design vs the two GPU baselines.

Baselines (only two; the batched/coalesced variant is NOT a baseline):

  B0  naive NTT      the reference expansion + the naive NTT backend.
  B1  reference      the reference implementation exactly as written
                     (four-step NTT, standalone bit-reversal, serial blocks,
                     c^2*n CW exchanges).  Measured end to end.

B1 is the measured wall clock of the reference, unmodified.  It includes a
device->host->device round trip of g on every poly_mul (poly_mul_u64_gpuntt_square
takes host pointers; pcg_ole_impl.h:467-481 and pcg_ole.h:217-243 copy and repack
it, once per (i,j) at batch=1).  Keeping g on the device would remove that, but
it is a software optimisation and the baseline is defined to exclude software
optimisations -- so it stays in.  Its size is reported per row (sm_pcie) so the
reader can see how much of the gap is data placement.

B0 replaces only the device part of B1's NTT lane with the naive backend's,
measured at the same (logN, batch); everything else, PCIe included, carries
through untouched.

The PIM side pays no such round trip: g is written into DRAM by the SPU and read
from DRAM by the SM.  That is an architectural difference, not a software one,
and it is the point of near-memory expansion.

The PIM side does pay its protocol network in full.  e2e = max(DPF lane, NTT
lane, NET lane) with NET = (c^2*n + 2c^2) * alpha: the same exchange count the
reference issues, no gang and no coalescing assumed.  The co-schedule buys
overlap between the three lanes, nothing more.
"""
import csv, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "GPU_baseline")
MACHINES = {
    # name:   (csv dir, e2e file, arm filter, PIM ns/leaf, tCK ns, grade)
    "L40S": dict(dir="data_l40s", e2e="e2e_arms.csv", arm="serial",
                 pim_ns=718660 * 0.444 / (768 << 12), tck=0.444,
                 grade="measured (ramulator2 718,660 CK, GDDR6_L40S org, 192 SPU)"),
    # B200: the pass-2-inclusive L40S point scaled by the device-transfer factor
    # 0.1069, itself fixed by two DIRECT sim anchors (v5_ch465 = 692,720 CK on
    # the GDDR6 org, v9b_base = 701,600 CK on the HBM3E_B200D org).  Derived,
    # not a direct B200-with-pass-2 run.
    "B200": dict(dir="pcg_baseline_out", e2e="e2e_full.csv", arm="serial",
                 pim_ns=718660 * 0.444 / (768 << 12) * 0.1069, tck=0.500,
                 grade="derived: L40S measured x 0.1069 device-transfer factor"),
}
DATA = None

# ---- PIM anchor -----------------------------------------------------------
# ramulator2, GDDR6_L40S org (24 ch x 8 PU = 192 SPU, 2 banks/PU, all-bank
# broadcast, dual-issue ARX + LSU), chacha fused CONVERT, both output-layer
# passes.  n=12, I=768 instances -> 768*2^12 leaves.
PIM_CK      = 718660
PIM_TCK_NS  = 0.444
PIM_LEAVES  = 768 * (1 << 12)
PIM_NS_LEAF = PIM_CK * PIM_TCK_NS / PIM_LEAVES


def load(name):
    with open(os.path.join(ROOT, DATA, name)) as f:
        return list(csv.DictReader(f))


def main(machine="L40S"):
    global DATA, PIM_NS_LEAF
    M = MACHINES[machine]
    DATA, PIM_NS_LEAF = M["dir"], M["pim_ns"]
    ntt = {}
    for r in load("ntt_stages.csv"):
        if r["status"] == "FAIL":
            continue
        ntt[(int(r["logN"]), int(r["batch"]), r["backend"], r["brev_mode"])] = \
            float(r["ms_per_mul"])

    # e2e_arms (serial arm) is the campaign that passes the cross-check against the
    # independent (n,B) surface of dpf_shape.csv: 1.00-1.08x on all 17 shared
    # cells.  e2e_full.csv is 1.2-6.3x above the surface with the deviation
    # growing monotonically in block count, so it is not used for B1.
    rows = []
    for r in load(M["e2e"]):
        if r["status"] != "OK" or r.get("arm", M["arm"]) != M["arm"]:
            continue
        c, t, lg = int(r["c"]), int(r["t"]), int(r["logN"])
        b, muls = c * c, 2 * c * c
        leaves = b * 2 * (1 << lg) * t
        wall = float(r["wall_ms"])
        ntt_meas = float(r["ntt_ms"])

        dev_std = ntt.get((lg, b, "square", "standalone"))
        dev_fold = ntt.get((lg, b, "square", "fold"))
        dev_naive = ntt.get((lg, b, "naive", "-"))
        if dev_std is None:
            continue
        dev_std *= muls
        pcie = ntt_meas - dev_std                    # identical in B0 and B1

        b1 = wall
        b0 = (wall - dev_std + dev_naive * muls) if dev_naive else None

        # The NTT lane is the same work on both sides: same four-step, same
        # standalone bit-reversal, same backend, and on GDDR6 the DRU hides none
        # of it.  It is device-only, because in the PIM design g is written into
        # DRAM by the SPU and read from DRAM by the SM -- there is no bus for it
        # to cross.  The reference's device->host->device round trip (sm_pcie)
        # is a defect of the reference GPU code, recoverable in software without
        # any PIM, so it is reported separately and never charged to the
        # architecture comparison.
        sm_ntt   = dev_std
        sm_pcie  = pcie
        gpu_dpf  = float(r["expand_ms"]) + float(r["convert_ms"])
        bv       = float(r["beaver_ms"])
        pim_lane = leaves * PIM_NS_LEAF / 1e6

        # B1 is the measured wall clock, full stop.  Subtracting the PCIe round
        # trip would produce a repaired GPU -- a software optimisation the
        # baseline is defined to exclude -- not the reference.
        b1 = wall
        b0 = (wall - sm_ntt + dev_naive * muls) if dev_naive else None

        # The PIM side pays its network in full.  No coalescing is assumed: the
        # design issues the same c^2*n correction-word exchanges as the
        # reference, plus 2c^2 Beaver opens, each a party-serialized round trip
        # of alpha.  What the co-schedule buys is overlap, not fewer messages --
        # SPU expansion, SM transform and the NIC are three lanes and the
        # steady state is the slowest of them, not their sum.
        net_lane = (int(r["exchanges"]) + 2 * b) * float(r["rtt_us"]) / 1000.0
        pimgpu = max(pim_lane, sm_ntt, net_lane)
        rows.append(dict(
            lam=int(r.get("lambda", 0)), c=c, t=t, logN=lg, tier=r["tier"],
            rtt=float(r["rtt_us"]), leaves=leaves, exch=int(r["exchanges"]),
            expand=float(r["expand_ms"]), convert=float(r["convert_ms"]),
            beaver=float(r["beaver_ms"]), ntt_meas=ntt_meas,
            dev_std=dev_std, gpu_dpf=gpu_dpf, sm_ntt=sm_ntt, sm_pcie=sm_pcie,
            machine=machine, b0=b0, b1=b1,
            pim_lane=pim_lane, net_lane=net_lane, pimgpu=pimgpu,
            bound=("NET" if net_lane >= max(pim_lane, sm_ntt)
                   else ("PIM" if pim_lane > sm_ntt else "NTT")),
        ))
    return rows


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


if __name__ == "__main__":
    rows = main()
    out = sys.stdout
    out.write("L40S: PIM+GPU vs GPU baselines B0 (naive NTT) and B1 (reference)\n")
    out.write(f"PIM DPF+convert lane = {PIM_NS_LEAF:.4f} ns/leaf "
              f"(ramulator2 {PIM_CK:,} CK, tCK={PIM_TCK_NS} ns, "
              f"{PIM_LEAVES:,} leaves, both output passes)\n\n")

    for tier in ("nvlink", "loopback", "wan"):
        sel = [r for r in rows if r["tier"] == tier]
        if not sel:
            continue
        out.write(f"--- tier={tier} (rtt={sel[0]['rtt']} us) ---\n")
        out.write("  (c,t) logN | gpuDPF  NTTlane |     B0     B1 |"
                  " PIMlane  PIM+GPU bound |  vs B0   vs B1 |"
                  " of which PCIe\n")
        for r in sel:
            f = lambda x: f"{x:6.0f}" if x is not None else "  n/a "
            g = lambda x: f"{x / r['pimgpu']:6.2f}x" if x else "     - "
            out.write(f"  ({r['c']},{r['t']:>3}) {r['logN']:>4} |"
                      f" {r['gpu_dpf']:6.0f} {r['sm_ntt']:8.0f} |"
                      f"{f(r['b0'])} {f(r['b1'])} |"
                      f" {r['pim_lane']:7.1f} {r['pimgpu']:8.1f} {r['bound']:>5} |"
                      f" {g(r['b0'])} {g(r['b1'])} |"
                      f" {r['sm_pcie']:13.0f}\n")
        v0 = [r["b0"] / r["pimgpu"] for r in sel if r["b0"]]
        v1 = [r["b1"] / r["pimgpu"] for r in sel]
        nb = sum(1 for r in sel if r["bound"] == "PIM")
        out.write(f"  geomean:  vs B0 {gm(v0):.2f}x (n={len(v0)})   "
                  f"vs B1 {gm(v1):.2f}x (n={len(v1)})   "
                  f"PIM-bound {nb}/{len(sel)}\n\n")
