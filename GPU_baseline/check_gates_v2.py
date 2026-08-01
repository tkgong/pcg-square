#!/usr/bin/env python3
"""Evaluate the six gates of the B200 full-table package, plus the package's own
contention consistency check.

Nothing is smoothed: an empty cell is reported, never filled or inferred. OOM
and CRASH statuses are counted separately from silent blanks, because the
package treats an OOM as a result and a blank as a defect.

Usage: python3 check_gates_v2.py [pcg_baseline_out]
"""
import collections
import csv
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "pcg_baseline_out"
FAILED = []


def head(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def verdict(name, ok, note=""):
    if not ok:
        FAILED.append(name)
    print(f"\n  ==> {name}: {'PASS' if ok else 'FAIL'}{('  — ' + note) if note else ''}")


class Row(dict):
    """A CSV row that yields '' for absent columns, so a renamed or dropped
    column surfaces as an explicit gate FAIL rather than a KeyError traceback
    halfway through the report."""

    def __missing__(self, k):
        return ""


def rows(path):
    p = os.path.join(OUT, path)
    if not os.path.exists(p):
        print(f"  MISSING FILE: {p}")
        return None
    with open(p) as fh:
        return [Row(r) for r in csv.DictReader(fh)]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---- gate 1 ---------------------------------------------------------------
head("GATE 1 — square_verify.log AND square_verify_fold.log end in GATE PASS")
ok1 = True
for f in ("square_verify.log", "square_verify_fold.log"):
    p = os.path.join(OUT, f)
    if not os.path.exists(p):
        print(f"  MISSING: {f}")
        ok1 = False
        continue
    lines = [l.rstrip("\n") for l in open(p) if l.strip()]
    tail = lines[-1].strip() if lines else "(empty)"
    print(f"  {f}: last line = {tail!r}   ({len(lines)} lines)")
    ok1 &= tail == "GATE PASS"
verdict("gate 1", ok1)

# ---- gate 2 ---------------------------------------------------------------
head("GATE 2 — every naive row in ntt_stages.csv shows BITEXACT")
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
    # the package says large-N batch=64 naive failures are the library's own
    # grid limit and expected; separate them so the count is honest either way
    expected = [r for r in bad if r["batch"] == "64" and int(r["logN"]) >= 20]
    other = [r for r in bad if r not in expected]
    print(f"  of those, batch=64 & logN>=20 (package says expected): {len(expected)}")
    print(f"  unexpected naive failures: {len(other)}")
    verdict("gate 2", not bad,
            f"{len(expected)} of {len(bad)} are the expected library grid limit"
            if bad else "")

# ---- gate 3 ---------------------------------------------------------------
head("GATE 3 — brev placement: standalone => brev_ms > 0; fold => brev_ms == 0")
if nt is None:
    verdict("gate 3", False)
else:
    sa = [r for r in nt if r["backend"] == "square" and r["brev_mode"] == "standalone"
          and r["status"] == "ok"]
    fo = [r for r in nt if r["backend"] == "square" and r["brev_mode"] == "fold"
          and r["status"] == "ok"]
    sa_bad = [r for r in sa if (fnum(r["brev_ms"]) or 0.0) == 0.0]
    fo_bad = [r for r in fo if (fnum(r["brev_ms"]) or 0.0) != 0.0]
    print(f"  standalone ok-rows: {len(sa)}   with brev_ms>0: {len(sa) - len(sa_bad)}")
    print(f"  fold       ok-rows: {len(fo)}   with brev_ms==0: {len(fo) - len(fo_bad)}")
    for r in sa_bad[:10]:
        print(f"    standalone but brev_ms=0: logN={r['logN']} batch={r['batch']}")
    for r in fo_bad[:10]:
        print(f"    fold but brev_ms!=0: logN={r['logN']} batch={r['batch']} "
              f"brev_ms={r['brev_ms']}")
    verdict("gate 3", not sa_bad and not fo_bad,
            "inverted => PCG_SQUARE_FOLD_BREV did not take effect"
            if (sa_bad or fo_bad) else "")

# ---- gates 4/5/6 on e2e_full ---------------------------------------------
e2 = rows("e2e_full.csv")
head("GATE 4 — e2e_full: wall_ms >= expand + convert + beaver + ntt")
if e2 is None:
    verdict("gate 4", False)
else:
    okc = [r for r in e2 if (r["status"] or "").strip() == "OK"]
    viol = []
    gaps = []
    for r in okc:
        parts = [fnum(r[k]) for k in ("expand_ms", "convert_ms", "beaver_ms", "ntt_ms")]
        w = fnum(r["wall_ms"])
        if w is None or any(p is None for p in parts):
            continue
        s = sum(parts)
        gap = (w - s) / w if w else 0.0
        gaps.append((gap, r, s, w))
        if w < s:
            viol.append((r, s, w))
    print(f"  OK rows: {len(okc)}   comparable: {len(gaps)}   wall < sum: {len(viol)}")
    for r, s, w in viol[:10]:
        print(f"    VIOLATION lam={r['lambda']} c={r['c']} t={r['t']} logN={r['logN']} "
              f"tier={r['tier']} arm={r['arm']}: wall={w} < sum={s:.4f}")
    if gaps:
        gaps.sort(key=lambda g: g[0], reverse=True)
        print("  largest unaccounted gaps (wall - sum)/wall — big gap means contention:")
        for gap, r, s, w in gaps[:8]:
            print(f"    {gap*100:6.2f}%  lam={r['lambda']} c={r['c']} t={r['t']} "
                  f"logN={r['logN']} tier={r['tier']} arm={r['arm']} wall={w} sum={s:.4f}")
        med = sorted(g[0] for g in gaps)[len(gaps) // 2]
        print(f"  median gap: {med*100:.2f}%")
    verdict("gate 4", not viol)

head("GATE 5 — both arms present per non-OOM cell; batched expand <= serial expand")
if e2 is None:
    verdict("gate 5", False)
else:
    cells = collections.defaultdict(dict)
    for r in e2:
        cells[(r["lambda"], r["c"], r["t"], r["logN"], r["tier"])][r["arm"]] = r
    missing = [k for k, v in cells.items() if set(v) != {"serial", "batched"}]
    print(f"  cells: {len(cells)}   with both arms: {len(cells) - len(missing)}")
    for k in missing[:10]:
        print(f"    MISSING ARM: lam={k[0]} c={k[1]} t={k[2]} logN={k[3]} tier={k[4]} "
              f"has={sorted(cells[k])}")
    st = collections.Counter((r["status"] or "").strip() for r in e2)
    print(f"  status counts: {dict(st)}")
    slower = []
    for k, v in cells.items():
        s, b = v.get("serial"), v.get("batched")
        if not s or not b:
            continue
        if (s["status"] or "").strip() != "OK" or (b["status"] or "").strip() != "OK":
            continue
        se, be = fnum(s["expand_ms"]), fnum(b["expand_ms"])
        if se is None or be is None:
            continue
        if be > se:
            slower.append((k, se, be, (be - se) / se * 100))
    print(f"  cells where batched expand > serial expand: {len(slower)}")
    for k, se, be, pct in sorted(slower, key=lambda x: -x[3])[:10]:
        print(f"    lam={k[0]} c={k[1]} t={k[2]} logN={k[3]} tier={k[4]}: "
              f"serial={se} batched={be}  (+{pct:.2f}%)")
    verdict("gate 5", not missing and not slower)

head("GATE 6 — monotonicity")
ok6 = True
if e2 is not None:
    print("\n  (a) e2e wall_ms non-decreasing in N, per (lambda,c,t,tier,arm)")
    g = collections.defaultdict(list)
    for r in e2:
        if (r["status"] or "").strip() == "OK":
            g[(r["lambda"], r["c"], r["t"], r["tier"], r["arm"])].append(r)
    v = 0
    for k, rs in sorted(g.items()):
        rs.sort(key=lambda r: int(r["logN"]))
        seq = [(int(r["logN"]), fnum(r["wall_ms"])) for r in rs if fnum(r["wall_ms"])]
        for (l0, a), (l1, b) in zip(seq, seq[1:]):
            if b < a:
                v += 1
                print(f"    VIOLATION lam={k[0]} c={k[1]} t={k[2]} tier={k[3]} arm={k[4]}: "
                      f"logN {l0}->{l1}  {a} -> {b}  (-{(a-b)/a*100:.2f}%)")
    print(f"    violations: {v}")
    ok6 &= v == 0

    print("\n  (c) e2e wall_ms increases with tier RTT, per (lambda,c,t,logN,arm)")
    order = ["nvlink", "pcie", "loopback", "dc", "wan"]
    v = 0
    gr = collections.defaultdict(dict)
    for r in e2:
        if (r["status"] or "").strip() == "OK":
            gr[(r["lambda"], r["c"], r["t"], r["logN"], r["arm"])][r["tier"]] = fnum(r["wall_ms"])
    for k, tv in sorted(gr.items()):
        seq = [(t, tv[t]) for t in order if tv.get(t) is not None]
        for (t0, a), (t1, b) in zip(seq, seq[1:]):
            if b < a:
                v += 1
                print(f"    VIOLATION lam={k[0]} c={k[1]} t={k[2]} logN={k[3]} arm={k[4]}: "
                      f"{t0}->{t1}  {a} -> {b}  (-{(a-b)/a*100:.2f}%)")
    print(f"    violations: {v}")
    ok6 &= v == 0

if nt is not None:
    print("\n  (b) ntt ms_per_mul non-decreasing in N, per (backend,brev_mode,batch)")
    v = 0
    gb = collections.defaultdict(list)
    for r in nt:
        if r["ms_per_mul"]:
            gb[(r["backend"], r["brev_mode"], r["batch"])].append(r)
    for k, rs in sorted(gb.items()):
        rs.sort(key=lambda r: int(r["logN"]))
        seq = [(int(r["logN"]), fnum(r["ms_per_mul"])) for r in rs if fnum(r["ms_per_mul"])]
        for (l0, a), (l1, b) in zip(seq, seq[1:]):
            if b < a:
                v += 1
                print(f"    VIOLATION {k[0]}/{k[1]}/batch={k[2]}: logN {l0}->{l1}  "
                      f"{a} -> {b}  (-{(a-b)/a*100:.2f}%)")
    print(f"    violations: {v}")
    ok6 &= v == 0
verdict("gate 6", ok6)

# ---- package's own contention check ---------------------------------------
head("CONTENTION CHECK (package-supplied) — ms_per_mul / (sum_ms/batch) in "
     "[0.75, 1.35] for logN >= 16")
if nt is None:
    print("  no ntt_stages.csv")
else:
    outside = []
    inside = 0
    for r in nt:
        if r["backend"] != "square" or int(r["logN"]) < 16:
            continue
        m, s, b = fnum(r["ms_per_mul"]), fnum(r["sum_ms"]), fnum(r["batch"])
        if None in (m, s, b) or s == 0:
            continue
        ratio = m / (s / b)
        if 0.75 <= ratio <= 1.35:
            inside += 1
        else:
            outside.append((ratio, r))
    print(f"  square rows logN>=16 checked: {inside + len(outside)}   inside: {inside}"
          f"   OUTSIDE: {len(outside)}")
    for ratio, r in sorted(outside, key=lambda x: -abs(x[0] - 1))[:15]:
        print(f"    ratio={ratio:.3f}  logN={r['logN']} batch={r['batch']} "
              f"mode={r['brev_mode']} ms_per_mul={r['ms_per_mul']} sum_ms={r['sum_ms']}")
    if outside:
        print("  -> these cells were contended (or the two columns use different"
              " normalisations); they must be marked, not reported as clean.")

# ---- shape / grid inventory ----------------------------------------------
head("INVENTORY + blanks")
for f in ("dpf_shape.csv", "ntt_stages.csv", "dpf_grid.csv", "e2e_full.csv",
          "square_verify.log", "square_verify_fold.log", "topo.txt",
          "gpu_before.txt", "gpu_after.txt"):
    p = os.path.join(OUT, f)
    print(f"  {'OK  ' if os.path.exists(p) and os.path.getsize(p) else 'MISS'} {f}"
          f"  {os.path.getsize(p) if os.path.exists(p) else 0} B")
for name in ("dpf_shape.csv", "dpf_grid.csv"):
    rs = rows(name)
    if rs is None:
        continue
    bad = [r for r in rs if (r.get("status") or "").strip() != "ok"]
    print(f"  {name}: {len(rs)} rows, non-ok: {len(bad)}")
    seen = set()
    for r in bad:
        s = (r.get("status") or "").strip()
        if s not in seen:
            seen.add(s)
            print(f"    e.g. status={s!r}  t={r.get('t')} n={r.get('n')} "
                  f"logN={r.get('logN','-')}")

print("\n" + "=" * 74)
print("OVERALL: " + ("ALL GATES PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
sys.exit(1 if FAILED else 0)
