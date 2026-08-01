#!/usr/bin/env python3
"""Network model of the PCG-OLE protocol: calibrate from comm traces, then
answer "when does the network become the new bottleneck as PIM speeds up?".

Inputs (produced by run_comm_calib.sh with a -DPCG_ENABLE_PROFILING build):
    <dir>/n{lg}_c{c}_t{t}_w{w}.p{0,1}.comm.csv
      party,context,phase,level,batch,send_bytes,recv_bytes,
      t_begin_ns,t_mid_ns,t_end_ns,io_counter_delta

Model (per exchange, party-serialized as the code does it -- see
docs/network_model.md §5):
    T_exchange(M) = 2*(alpha + M / BW_eff)          [+ skew]
Calibration:
    alpha   : intercept of the per-exchange time vs bytes regression,
              dominated by the many small dpf_gen/beaver rows
    BW_eff  : 1/slope of the same regression (loopback here)
    skew    : |t_begin(ALICE) - t_begin(BOB)| per matched exchange

Outputs: pim/results/comm_calib_l40s.txt (measured tables + fit) and, with
--crossover, the PIM-generation x network-condition bottleneck table.
"""
import csv, math, os, re, sys, glob


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["send_bytes"] = int(r["send_bytes"])
            r["recv_bytes"] = int(r["recv_bytes"])
            r["t0"] = int(r["t_begin_ns"]); r["t1"] = int(r["t_end_ns"])
            r["level"] = int(r["level"]); r["batch"] = int(r["batch"])
            r["dur_ns"] = r["t1"] - r["t0"]
            rows.append(r)
    return rows


def fit_alpha_beta(rows):
    """Least squares dur_ns = a + b*bytes over per-exchange samples."""
    xs = [r["send_bytes"] for r in rows]
    ys = [r["dur_ns"] for r in rows]
    n = len(xs)
    if n < 2:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return my, None
    b = sxy / sxx
    a = my - b * mx
    return a, b            # a = ns intercept, b = ns per byte


def phase_summary(rows):
    agg = {}
    for r in rows:
        p = r["phase"]
        d = agg.setdefault(p, {"rows": 0, "bytes": 0, "ns": 0})
        d["rows"] += 1
        d["bytes"] += r["send_bytes"]
        d["ns"] += r["dur_ns"]
    return agg


def parse_tag(path):
    m = re.search(r"n(\d+)_c(\d+)_t(\d+)_w(\d+)", os.path.basename(path))
    return tuple(int(x) for x in m.groups()) if m else None


def main():
    S = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(S, "*.p0.comm.csv")))
    if not files:
        sys.exit(f"no comm traces in {S}")

    results = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results, exist_ok=True)
    out = os.path.join(results, "comm_calib_l40s.txt")
    with open(out, "w") as f:
        f.write("== PCG network calibration (measured, localhost 2PC)\n"
                "== source: comm-trace CSV, one row per exchange; the protocol\n"
                "== is party-serialized so each row is a full round trip.\n"
                "== see docs/network_model.md for the symbolic model.\n==\n")

        allrows = []
        f.write("---- 1. per-configuration phase breakdown ----\n")
        f.write(f"   {'cfg':>18} {'phase':>12} {'rows':>6} {'MB sent':>9} "
                f"{'ms in net':>10} {'B/exch':>9} {'us/exch':>8}\n")
        for p0 in files:
            tag = parse_tag(p0)
            if not tag:
                continue
            lg, c, t, w = tag
            rows = load(p0)
            allrows += rows
            cfg = f"2^{lg} c{c} t{t} w{w}"
            for ph, d in sorted(phase_summary(rows).items()):
                f.write(f"   {cfg:>18} {ph:>12} {d['rows']:>6} "
                        f"{d['bytes']/1e6:9.2f} {d['ns']/1e6:10.2f} "
                        f"{d['bytes']//max(1,d['rows']):9d} "
                        f"{d['ns']/max(1,d['rows'])/1e3:8.1f}\n")
            f.write("\n")

        # ---- alpha/beta fit -------------------------------------------
        a, b = fit_alpha_beta(allrows)
        small = [r for r in allrows if r["send_bytes"] <= 1024]
        big = [r for r in allrows if r["send_bytes"] > 64 * 1024]
        f.write("---- 2. alpha / beta fit (all exchanges pooled) ----\n")
        if a is not None and b:
            f.write(f"   dur_ns = {a:.0f} + {b:.4f} * bytes\n")
            f.write(f"   alpha_rt = {a/1e3:.1f} us per round trip "
                    f"(party-serialized send+recv)\n")
            f.write(f"   BW_eff   = {1e9/b/1e9:.2f} GB/s = {8/b:.2f} Gbps "
                    f"(loopback)\n")
        if small:
            med = sorted(r["dur_ns"] for r in small)[len(small)//2]
            f.write(f"   small-message median ({len(small)} rows <=1KB): "
                    f"{med/1e3:.1f} us  -> latency floor\n")
        if big:
            tot_b = sum(r['send_bytes'] for r in big)
            tot_n = sum(r['dur_ns'] for r in big)
            f.write(f"   large-message throughput ({len(big)} rows >64KB): "
                    f"{tot_b/tot_n:.2f} GB/s\n")
        f.write("\n")

        # ---- rounds vs bytes, and the w knob ---------------------------
        f.write("---- 3. latency-critical vs bandwidth-critical split ----\n")
        f.write(f"   {'cfg':>18} {'rounds':>7} {'MB':>8} {'net ms':>8} "
                f"{'dpfgen rounds':>14} {'dltsft MB':>10}\n")
        for p0 in files:
            tag = parse_tag(p0)
            if not tag:
                continue
            lg, c, t, w = tag
            rows = load(p0)
            agg = phase_summary(rows)
            gen = agg.get("dpf_gen", {"rows": 0})
            # circuit_and rows belong to KS + DltSft; DltSft dominates bytes
            cir = agg.get("circuit_and", {"bytes": 0})
            f.write(f"   {'2^'+str(lg)+' c'+str(c)+' t'+str(t)+' w'+str(w):>18} "
                    f"{len(rows):>7} {sum(r['send_bytes'] for r in rows)/1e6:8.2f} "
                    f"{sum(r['dur_ns'] for r in rows)/1e6:8.1f} "
                    f"{gen['rows']:>14} {cir['bytes']/1e6:10.2f}\n")
        f.write("\n   NOTE: dpf_gen rows are the levels x c^2 blocking round\n"
                "   trips (constant 16*B*(m-1) bytes each) -- pure latency;\n"
                "   circuit_and bytes are dominated by the delta-shift AND\n"
                "   triples -- pure bandwidth. Raising w trades the former\n"
                "   for the latter.\n")

    print(f"wrote {out}")
    if a is not None and b:
        print(f"alpha_rt={a/1e3:.1f}us  BW_eff={8/b:.2f}Gbps  rows={len(allrows)}")


if __name__ == "__main__":
    main()
