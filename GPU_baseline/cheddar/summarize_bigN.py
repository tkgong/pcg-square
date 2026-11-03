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
    cap = r.get("maxrreg", "0"); cfg = f"{r['P1']}+{r['P2']} M{r['M1']}/{r['M2']}" + (f" B{r['B1']}" if r["design"] in ("cheddar", "ole", "ole_t", "olep", "olepb") else "") + (f" R{cap}" if cap != "0" else "")
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
if any(k[2] in ("ole", "ole_t", "olep", "olepb", "tp") for k in B):
    out += ["", "PCG output z = INTT(sum_ij a^_i a^_j NTT(g_ij)) on Cheddar's kernels (cheddar_bigN_bench.cu --designs ole), batch = c^2 polynomials g_ij:",
            "ms per g_ij (time per output / c^2, the a transforms and the inverse amortised); old algorithm = 2 multiplies per g_ij = 2 x Cheddar's per-multiply time",
            "ole = g_ij in Cheddar's natural layout; ole_t = g_ij in the column-contiguous layout the SPUs write (stores unchanged);",
            "olep = the SPUs accumulated g_ij + g_ji into one buffer: c(c+1)/2 unordered pairs instead of c^2 polynomials (time still / c^2)",
            "olepb = olep with b^_ij = a^_i . a^_j precomputed once per public a (one read, one multiply per element; a transforms in the setup)",
            "tp = PCG^2's two-pass streaming NTT (SM passes contiguous on the SPU layout, the DRU re-layout in between), paired items, b^ precomputed (config = P1+P2)",
            "logN batch c | 2 x Cheddar | ole best | ole_t best | olep best | olepb best | tp best | 2 Cheddar / ole | 2 Cheddar / ole_t | ole / ole_t | ole / olep | 2 Cheddar / olep | olep / olepb | 2 Cheddar / olepb | olepb / tp | 2 Cheddar / tp"]
    for lg, b in keys:
        g = lambda d: B.get((lg, b, d), (float("nan"), "-"))
        c, o, ot, op, ob, tpv = g("cheddar"), g("ole"), g("ole_t"), g("olep"), g("olepb"), g("tp")
        if o[1] == "-" and ot[1] == "-" and op[1] == "-": continue
        out.append(f"  {lg} {b:3d} {int(round(b ** 0.5))} | {2 * c[0]:.4f} | {o[0]:.4f} ({o[1]}) | {ot[0]:.4f} ({ot[1]}) | {op[0]:.4f} ({op[1]}) | {ob[0]:.4f} ({ob[1]}) | {tpv[0]:.4f} ({tpv[1]}) | {2 * c[0] / o[0]:.2f}x | {2 * c[0] / ot[0]:.2f}x | {o[0] / ot[0]:.3f} | {o[0] / op[0]:.3f} | {2 * c[0] / op[0]:.2f}x | {op[0] / ob[0]:.3f} | {2 * c[0] / ob[0]:.2f}x | {ob[0] / tpv[0]:.3f} | {2 * c[0] / tpv[0]:.2f}x")
txt = "\n".join(out) + "\n"; open(f"{RES}/cheddar_bigN_{TAG}.txt", "w").write(txt); print(txt)
