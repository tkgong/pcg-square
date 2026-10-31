#!/usr/bin/env python3
"""Top-3 forward configs per logN (20..24) from a cheddar_ntt_bench sweep -> gen/cfgs.h for cheddar_polymul_bench.
Usage: gen_cfgs.py SWEEP_CSV OUT_H"""
import csv, sys
R = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if not l.startswith("#")) if r.get("ok") == "1" and 20 <= int(r["logN"]) <= 24]
items = []
for L in range(20, 25):
    rr = sorted((r for r in R if int(r["logN"]) == L), key=lambda r: float(r["ms_per_transform"]))[:3]
    items += [f"X({L},{r['P2']},{r['M1']},{r['B1']})" for r in rr]
open(sys.argv[2], "w").write(f"// top-3 forward configs per logN from {sys.argv[1]}: (L, P2, M1, B1)\n#define CHEDDAR_CFGS(X) \\\n" + " \\\n".join(items) + "\n")
