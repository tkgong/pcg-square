#!/usr/bin/env python3
"""Evaluate the four FINAL-caliber gates over pcg_baseline_out/.

Every gate prints PASS or FAIL with the offending rows quoted verbatim.
Nothing is smoothed over: a missing cell is a FAIL, not an omission.

Usage: python3 check_gates.py [pcg_baseline_out]
"""
import collections
import csv
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "pcg_baseline_out"
rc = 0


def head(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def verdict(name, ok):
    global rc
    if not ok:
        rc = 1
    print(f"\n  ==> {name}: {'PASS' if ok else 'FAIL'}")


def rows(path):
    p = os.path.join(OUT, path)
    if not os.path.exists(p):
        print(f"  MISSING FILE: {p}")
        return None
    with open(p) as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---- gate 1: square correctness ------------------------------------------
head("GATE 1 — square_verify.log ends in GATE PASS")
p = os.path.join(OUT, "square_verify.log")
if not os.path.exists(p):
    print("  MISSING FILE:", p)
    verdict("gate 1", False)
else:
    txt = [l.rstrip("\n") for l in open(p) if l.strip()]
    print("  last 3 lines:")
    for l in txt[-3:]:
        print("   ", l)
    verdict("gate 1", bool(txt) and txt[-1].strip() == "GATE PASS")

# ---- gate 2: every naive row BITEXACT ------------------------------------
head("GATE 2 — every naive row in ntt_stages.csv has BITEXACT")
nt = rows("ntt_stages.csv")
if nt is None:
    verdict("gate 2", False)
else:
    naive = [r for r in nt if r["backend"] == "naive"]
    bad = [r for r in naive if (r["status"] or "").strip() != "BITEXACT"]
    print(f"  naive rows: {len(naive)}   BITEXACT: {len(naive) - len(bad)}")
    for r in bad:
        print(f"    NOT BITEXACT: logN={r['logN']} batch={r['batch']} "
              f"ms_per_mul={r['ms_per_mul']!r} status={r['status']!r}")
    # square rows: report FAIL/empty separately (OOM candidates)
    sq = [r for r in nt if r["backend"] == "square"]
    sbad = [r for r in sq if (r["status"] or "").strip() != "ok" or not r["ms_per_mul"]]
    print(f"  square rows: {len(sq)}   ok: {len(sq) - len(sbad)}")
    for r in sbad:
        print(f"    SQUARE CELL NOT OK: logN={r['logN']} batch={r['batch']} "
              f"status={r['status']!r} ms_per_mul={r['ms_per_mul']!r}")
    verdict("gate 2 (naive BITEXACT)", not bad)

# ---- gate 3: exposed comms ~= exchanges * rtt ----------------------------
head("GATE 3 — dpf_net.csv: exposed_ms ~= exchanges * rtt (spin-wait)")
dn = rows("dpf_net.csv")
if dn is None:
    verdict("gate 3", False)
else:
    TOL = 0.05          # 'within a few percent'
    worst = []
    missing = []
    for r in dn:
        ex, rtt, nex = fnum(r["exchanges"]), fnum(r["rtt_us"]), fnum(r["net_exposed_ms"])
        if ex is None or rtt is None or nex is None:
            missing.append(r)
            continue
        pred = ex * rtt / 1000.0        # us -> ms
        if pred <= 0:
            continue
        rel = (nex - pred) / pred
        worst.append((abs(rel), rel, pred, nex, r))
    # key= on the first element only: the tuples end in a dict, which has no
    # ordering, so a tie on abs(rel) would raise TypeError.
    worst.sort(key=lambda w: w[0], reverse=True)
    off = [w for w in worst if w[0] > TOL]
    print(f"  rows compared: {len(worst)}   within {TOL*100:.0f}%: {len(worst) - len(off)}")
    if missing:
        print(f"  rows with empty exchanges/exposed: {len(missing)}")
        for r in missing[:10]:
            print(f"    EMPTY: lam={r['lambda']} c={r['c']} t={r['t']} logN={r['logN']} "
                  f"tier={r['tier']} exchanges={r['exchanges']!r} exposed={r['net_exposed_ms']!r}")
    print("  worst 12 deviations:")
    for a, rel, pred, nex, r in worst[:12]:
        print(f"    lam={r['lambda']} c={r['c']} t={r['t']} logN={r['logN']} "
              f"tier={r['tier']} rtt={r['rtt_us']} ex={r['exchanges']}  "
              f"pred={pred:.3f}ms measured={nex:.3f}ms  {rel*100:+.1f}%")
    # per-tier summary: a loaded machine shows up as one tier drifting
    bytier = collections.defaultdict(list)
    for a, rel, pred, nex, r in worst:
        bytier[r["tier"]].append(rel)
    print("  per-tier median deviation:")
    for tier, v in bytier.items():
        v.sort()
        print(f"    {tier:9s} n={len(v):4d} median={v[len(v)//2]*100:+.1f}% "
              f"max={max(v, key=abs)*100:+.1f}%")
    verdict("gate 3", not off and not missing)

# ---- gate 4: monotonicity -------------------------------------------------
head("GATE 4 — monotonicity")
ok4 = True
if dn is None or nt is None:
    verdict("gate 4", False)
else:
    print("\n  (a) DPF total_ms non-decreasing in N, per (lambda,c,t,tier)")
    g = collections.defaultdict(list)
    for r in dn:
        g[(r["lambda"], r["c"], r["t"], r["tier"])].append(r)
    viol = 0
    for k, rs in sorted(g.items()):
        rs.sort(key=lambda r: int(r["logN"]))
        seq = [(int(r["logN"]), fnum(r["total_ms"])) for r in rs if fnum(r["total_ms"]) is not None]
        for (l0, v0), (l1, v1) in zip(seq, seq[1:]):
            if v1 < v0:
                viol += 1
                pct = (v0 - v1) / v0 * 100 if v0 else 0
                print(f"    VIOLATION lam={k[0]} c={k[1]} t={k[2]} tier={k[3]}: "
                      f"logN {l0}->{l1}  {v0} -> {v1}  (-{pct:.2f}%)")
    print(f"    violations: {viol}")
    ok4 &= viol == 0

    print("\n  (b) square ms_per_mul non-decreasing in N, per batch")
    viol = 0
    gb = collections.defaultdict(list)
    for r in nt:
        if r["backend"] == "square":
            gb[r["batch"]].append(r)
    for b, rs in sorted(gb.items(), key=lambda kv: int(kv[0])):
        rs.sort(key=lambda r: int(r["logN"]))
        seq = [(int(r["logN"]), fnum(r["ms_per_mul"])) for r in rs if fnum(r["ms_per_mul"]) is not None]
        for (l0, v0), (l1, v1) in zip(seq, seq[1:]):
            if v1 < v0:
                viol += 1
                pct = (v0 - v1) / v0 * 100 if v0 else 0
                print(f"    VIOLATION batch={b}: logN {l0}->{l1}  {v0} -> {v1}  (-{pct:.2f}%)")
    print(f"    violations: {viol}")
    ok4 &= viol == 0

    print("\n  (c) for a fixed row, total_ms increases with the tier's RTT")
    order = ["nvlink", "pcie", "loopback", "dc", "wan"]
    viol = 0
    gr = collections.defaultdict(dict)
    for r in dn:
        gr[(r["lambda"], r["c"], r["t"], r["logN"])][r["tier"]] = fnum(r["total_ms"])
    for k, tv in sorted(gr.items()):
        seq = [(t, tv[t]) for t in order if tv.get(t) is not None]
        for (t0, v0), (t1, v1) in zip(seq, seq[1:]):
            if v1 < v0:
                viol += 1
                pct = (v0 - v1) / v0 * 100 if v0 else 0
                print(f"    VIOLATION lam={k[0]} c={k[1]} t={k[2]} logN={k[3]}: "
                      f"{t0} -> {t1}  {v0} -> {v1}  (-{pct:.2f}%)")
    print(f"    violations: {viol}")
    ok4 &= viol == 0
    verdict("gate 4", ok4)

# ---- inventory ------------------------------------------------------------
head("DELIVERABLE INVENTORY")
for f in ["ntt_stages.csv", "dpf_net.csv", "e2e.csv", "alpha_measured.csv",
          "topo.txt", "square_verify.log"]:
    p = os.path.join(OUT, f)
    print(f"  {'OK  ' if os.path.exists(p) and os.path.getsize(p) else 'MISS'} {f}"
          f"  {os.path.getsize(p) if os.path.exists(p) else 0} bytes")
d = os.path.join(OUT, "dpf_levels")
print(f"  dpf_levels/: {len(os.listdir(d)) if os.path.isdir(d) else 'MISSING'} files")
e = rows("e2e.csv")
print(f"  e2e.csv rows: {len(e) if e else 0}  (expect 15 logN x 6 rows x 5 tiers = 450)")
if dn is not None:
    print(f"  dpf_net.csv rows: {len(dn)}  (expect 450)")
if nt is not None:
    print(f"  ntt_stages.csv rows: {len(nt)}  (expect 15 x 3 x 2 = 90)")

print(f"\nOVERALL: {'ALL GATES PASS' if rc == 0 else 'AT LEAST ONE GATE FAILED'}")
sys.exit(rc)
