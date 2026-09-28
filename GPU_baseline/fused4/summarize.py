#!/usr/bin/env python3
"""Min-over-repetitions summary of fused4_bench CSV output (one or more runs)."""
import csv, math, sys

rows = [r for r in csv.reader(open(sys.argv[1]))
        if len(r) == 10 and r[0] != "gpu" and r[3]]
best, chk = {}, {}
for r in rows:
    k = (r[0], int(r[1]), int(r[2]))
    v = [float(x) for x in r[3:7]]
    best[k] = v if k not in best else [min(a, b) for a, b in zip(best[k], v)]
    chk[k] = r[9] if chk.get(k, "BITEXACT") == "BITEXACT" else chk[k]
print("gpu logN batch | merge  f4_full  f4_sm(SM lane, DRU)  f4_dru | "
      "merge/f4_sm  merge/f4_full | check   (ms per poly-mul)")
for k in sorted(best):
    m, f, s, d = best[k]
    print(f"{k[0]} {k[1]} {k[2]:>3} | {m:7.4f} {f:7.4f} {s:9.4f} {d:9.4f} | "
          f"{m / s:6.2f}x {m / f:6.2f}x | {chk[k]}")
for b in sorted({k[2] for k in best}):
    v = [best[k][0] / best[k][2] for k in best if k[2] == b]
    print(f"geomean merge/f4_sm at batch {b}: "
          f"{math.exp(sum(map(math.log, v)) / len(v)):.3f}x over {len(v)} sizes")
