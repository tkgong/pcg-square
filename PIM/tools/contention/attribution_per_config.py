#!/usr/bin/env python3
"""Per-(c,t) path from the submission's Fig. 8 labels to the final ones (geomean over logN of each (c,t)).
Steps, cumulative: 0 paper label (reproduced: paper's square NTT on both sides, paper's SPU lane) -> 1 GPU baseline NTT = merge
(SOTA) -> 2 PCG^2 NTT = final (merge on L40S, four-step + DRU on B200), simulated incl. gather -> 3 final SPU lane (static
ceil(I/N_SPU), H' on the SPU, LSU overlap) -> 4 contention -> 5 GPU baseline co-schedules the network (= Fig. 8 final).
Usage: attribution_per_config.py FINAL_WINDOW_DIR OUT"""
import io, contextlib, json, os, sys
from math import exp, log
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "e2e"))
with contextlib.redirect_stdout(io.StringIO()):
    from reproduce import L_, CFG, LOGN, PAPER_FIG8
    from fig8 import BETA
W, OUT = sys.argv[1], sys.argv[2]
gm = lambda v: exp(sum(map(log, v)) / len(v))
R = json.load(open(f"{W}/win22/e2e_nom.json"))
DES = {"L40S": ("l40s", "merge"), "B200": ("b200", "f4dru")}
lines = [__doc__.split("Usage")[0].rstrip(), "", "(c,t): paper | 0 reproduced | 1 base merge | 2 PCG2 NTT final | 3 SPU lane final | 4 contention | 5 base co-sched net = final"]
for tier, lab in (("fast", "40 Gbps"), ("slow", "400 Mbps")):
    for mach, (org, des) in DES.items():
        LN = L_(mach); lines.append(f"{lab} {mach}")
        for i, (c, t) in enumerate(CFG):
            cols = [[] for _ in range(6)]
            for lg in LOGN:
                L = LN.get((c, t, lg))
                r = [x for x in R if x["org"] == org and x["design"] == des and x["tier"] == tier and (x["c"], x["t"], x["logN"]) == (c, t, lg)]
                if not L or not r: continue
                r = r[0]; nic = r["nic_ms"]; bs = r["base_serial_ms"]
                cols[0].append(r["base_paper_ms"] / max(L["spu"], L["sm"], nic))
                cols[1].append(bs / max(L["spu"], L["sm"], nic))
                cols[2].append(bs / max(L["spu"], r["ntt_ms"], nic))
                cols[3].append(bs / max(r["spu_ms"], r["ntt_ms"], nic))
                cols[4].append(bs / r["pcg_ms"])
                cols[5].append(r["base_overlap_ms"] / r["pcg_ms"])
            lines.append(f"  ({c},{t:>3}): {PAPER_FIG8[tier][mach][i]:4.1f} | " + " | ".join(f"{gm(v):5.2f}" for v in cols))
open(OUT, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
