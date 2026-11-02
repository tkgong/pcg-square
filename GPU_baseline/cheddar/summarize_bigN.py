#!/usr/bin/env python3
"""Summary of cheddar_bigN_<tag>.csv: per (logN, batch) the best valid config of each design (0 mismatches), per multiply;
the four-step SM lane against Cheddar and merge; and the cost per butterfly (ms / (N log N), relative to 2^24) that shows
which design degrades with N.  " R64" / " R128" in a config: the build capped at that many registers per thread
(-maxrregcount; column maxrreg, absent in older CSVs = no cap).  Usage: summarize_bigN.py [TAG=l40s]"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1] if len(sys.argv) > 1 else "l40s"
RES = os.path.join(HERE, "..", f"results_{TAG}")
R = [r for r in csv.DictReader(l for l in open(f"{RES}/cheddar_bigN_{TAG}.csv") if not l.startswith("#")) if r["design"] != "OOM"]
oom = sorted({(int(r["logN"]), int(r["batch"])) for r in csv.DictReader(l for l in open(f"{RES}/cheddar_bigN_{TAG}.csv")) if r["design"] == "OOM"})
B = {}
for r in R:
    if r["mismatches"] != "0": continue
    k = (int(r["logN"]), int(r["batch"]), r["design"]); t = float(r["ms_per_poly"])
    cap = r.get("maxrreg", "0"); cfg = f"{r['P1']}+{r['P2']} M{r['M1']}/{r['M2']}" + (f" B{r['B1']}" if r["design"] == "cheddar" else "") + (f" R{cap}" if cap != "0" else "")
    if k not in B or t < B[k][0]: B[k] = (t, cfg)
bad = sum(r["mismatches"] != "0" for r in R)
out = [f"{TAG}: per negacyclic multiply, ms per polynomial, best valid config per design; {bad} of {len(R)} timed rows have mismatches (excluded)"
       + (f"; out of memory at {oom}" if oom else ""),
       "logN batch | merge | Cheddar best (col+row, M col/row, B, register cap) | four-step SM lane best | four-step, transposes on SMs | Cheddar/SM lane | merge/SM lane | Cheddar/(transposes on SMs)"]
keys = sorted({(k[0], k[1]) for k in B})
for lg, b in keys:
    g = lambda d: B.get((lg, b, d), (float("nan"), "-"))
    m, c, s, f = g("merge"), g("cheddar"), g("f4c_sm"), g("f4c_full")
    out.append(f"  {lg} {b:3d} | {m[0]:.4f} | {c[0]:.4f} ({c[1]}) | {s[0]:.4f} ({s[1]}) | {f[0]:.4f} | {c[0]/s[0]:.2f}x | {m[0]/s[0]:.2f}x | {c[0]/f[0]:.2f}x")
out += ["", "cost per butterfly, ms / (N log N), relative to 2^24 at the same batch:", "logN batch | merge | Cheddar | four-step SM lane"]
for lg, b in keys:
    ref = lambda d: B.get((24, b, d), (float("nan"),))[0] / ((1 << 24) * 24)
    rel = lambda d: B.get((lg, b, d), (float("nan"),))[0] / ((1 << lg) * lg) / ref(d)
    out.append(f"  {lg} {b:3d} | {rel('merge'):.2f} | {rel('cheddar'):.2f} | {rel('f4c_sm'):.2f}")
txt = "\n".join(out) + "\n"; open(f"{RES}/cheddar_bigN_{TAG}.txt", "w").write(txt); print(txt)
