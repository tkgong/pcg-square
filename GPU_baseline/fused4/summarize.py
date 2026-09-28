#!/usr/bin/env python3
"""Min-over-repetitions summary of fused4_bench CSV output (one or more runs).

Columns compared (ms per negacyclic poly-mul):
  merge     GPU-NTT merge backend (no transposes)
  f4_full   fused four-step, own kernels, transposes on the SMs (GPU only)
  f4_sm     fused four-step, own kernels, SM lane only (transposes on the DRU)
  f4g_full  fused four-step on GPU-NTT's kernels, transposes on the SMs
  f4g_sm    fused four-step on GPU-NTT's kernels, SM lane only (transposes on the DRU)
"""
import csv, math, sys

COLS = ["merge", "f4_full", "f4_sm", "f4_dru", "f4g_full", "f4g_sm"]
best, chk = {}, {}
for r in csv.reader(open(sys.argv[1])):
    if len(r) < 10 or r[0] == "gpu" or not r[3]:
        continue
    k = (r[0], int(r[1]), int(r[2]))
    v = [float(x) for x in r[3:7]] + ([float(r[10]), float(r[11])] if len(r) >= 15 and r[14] != "NA" else [0.0, 0.0])
    best[k] = v if k not in best else [min(a, b) if a and b else (a or b) for a, b in zip(best[k], v)]
    c = r[9] + ("/" + r[14] if len(r) >= 15 else "")
    chk[k] = c if chk.get(k, c) == c else chk[k] + "|" + c
print("gpu logN batch | " + " ".join(f"{c:>9}" for c in COLS) +
      " | merge/f4_sm merge/f4g_sm | f4_full/f4_sm f4g_full/f4g_sm | check")
for k in sorted(best):
    v = dict(zip(COLS, best[k]))
    q = lambda a, b: f"{v[a] / v[b]:6.2f}x" if v[a] and v[b] else "     -"
    print(f"{k[0]} {k[1]} {k[2]:>3} | " + " ".join(f"{v[c]:9.4f}" for c in COLS) +
          f" | {q('merge', 'f4_sm')} {q('merge', 'f4g_sm')} | {q('f4_full', 'f4_sm')} {q('f4g_full', 'f4g_sm')} | {chk[k]}")
for b in sorted({k[2] for k in best}):
    for col in ("f4_sm", "f4g_sm"):
        vals = [best[k][0] / best[k][COLS.index(col)] for k in best if k[2] == b and best[k][COLS.index(col)]]
        if vals:
            print(f"geomean merge/{col} at batch {b}: {math.exp(sum(map(math.log, vals)) / len(vals)):.3f}x "
                  f"over {len(vals)} sizes")
