#!/usr/bin/env python3
"""Per-block area of the three configurations (out/*_hier_444.stat) and the increments. kGE = area / NAND2_X1 (0.798 um^2)."""
import re
NAND2 = 0.798
def parse(f):
    mods, cur, ff = {}, None, {}
    for l in open(f):
        m = re.match(r"=== (.+) ===", l)
        if m: cur = m.group(1).strip(); continue
        m = re.match(r"\s+Chip area for module '(.+)': ([\d.]+)", l)
        if m: mods[m.group(1)] = float(m.group(2))
        m = re.match(r"\s+Chip area for top module '(.+)': ([\d.]+)", l)
        if m: mods["TOTAL"] = float(m.group(2))
        m = re.match(r"\s+(\d+)\s+[\d.E+]+\s+(DFF\w*)\s*$", l)
        if m and cur and cur != "design hierarchy": ff[cur] = ff.get(cur, 0) + int(m.group(1))
    return mods, ff
def short(n):
    n = n.lstrip("\\")
    for k in ("ctrl", "rf", "srf"):
        if ("\\" + k) in n: return k
    return "glue" if (n.startswith("pu") or "\\pu\\" in n) else n
R, F = {}, {}
for t in ("aim", "spu", "uni"):
    mods, ff = parse(f"out/{t}_hier_444.stat"); R[t], F[t] = {}, {}
    for k, v in mods.items(): kk = k if k == "TOTAL" else short(k); R[t][kk] = R[t].get(kk, 0) + v
    for k, v in ff.items(): F[t][short(k)] = F[t].get(short(k), 0) + v
NAMES = [("rf", "register file, 2 groups x 8 x 256 b"), ("srf", "scalar registers"), ("ctrl", "instruction memory 32 x 32 b, PC, loop, decode"),
         ("bf16_dp", "BF16: 16 multipliers, 16 adders, MAC stage 1"), ("bf16_acc", "BF16: MAC align + sum tree, accumulator, read-out"),
         ("xadd", "ARX: 8 x 32 b adder array (+ 64 b link, 62 b mask)"), ("foldk", "ARX: sparse-prime fold constant"),
         ("vxorl", "ARX: 8 x 32 b xor + rotate array"), ("useq", "ARX: ChaCha8 EXTEND micro-sequencer"),
         ("lsu", "ARX: load skid 4 col + store queue 4 col + child pack"), ("glue", "operand / write muxes, forward register, bank latch"), ("TOTAL", "")]
print(f"{'block':9} | {'AiM-function':>12} | {'SPU':>8} | {'unified':>8} | {'uni - AiM':>9} | {'uni - SPU':>9} | flops a/s/u      | content")
for k, d in NAMES:
    a, s, u = (R[t].get(k, 0) / NAND2 / 1e3 for t in ("aim", "spu", "uni"))
    fl = "/".join(str(F[t].get(k, 0) if k != "TOTAL" else sum(F[t].values())) for t in ("aim", "spu", "uni"))
    print(f"{k:9} | {a:12.2f} | {s:8.2f} | {u:8.2f} | {u - a:9.2f} | {u - s:9.2f} | {fl:16} | {d}")
a, s, u = (R[t]["TOTAL"] for t in ("aim", "spu", "uni"))
print(f"\n(kGE; NanGate45 um^2: AiM-function {a:.0f}, SPU {s:.0f}, unified {u:.0f})")
print(f"ChaCha-specific increment over the AiM-function unit: {(u - a) / NAND2 / 1e3:.2f} kGE = +{100 * (u - a) / a:.1f}% ({100 * (u - a) / u:.1f}% of the unified unit)")
print(f"BF16 increment over the SPU:                          {(u - s) / NAND2 / 1e3:.2f} kGE = +{100 * (u - s) / s:.1f}%")
print(f"two separate units {(a + s) / NAND2 / 1e3:.2f} kGE, unified {u / NAND2 / 1e3:.2f} kGE: sharing saves {100 * (a + s - u) / (a + s):.1f}%")
