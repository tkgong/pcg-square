#!/usr/bin/env python3
"""Security-calibrated GPU NTT benchmark collector (BCG+20 Table 1 sizes).

Mapping: NTT size = N (ring dimension; t never enters the NTT); the c^2 g_ij
poly-muls per expansion set the batch -> c in {2,4,8} <=> batch in {4,16,64}.
Inputs: endtoend/run_ntt_secparams.sh outputs ({be}_lg{N}_b{B}.txt + _stages).
Output: pim/results/l40s_ntt_secparams.txt
  A: raw ms/mul + stage splits per (backend, N, batch), merge-vs-4step ratio
  B: per security row (lambda, N, c, t): three NTT-lane conventions
     (i)  serial as-implemented  = 2 c^2 full muls
     (ii) Move2 streamed (merge) = c^2 x (fwd+pw) + 1 INTT
     (iii)Move2+AGU (4step)      = c^2 x (ntt+twist+pw, NO input transpose)
                                   + 1 INTT-side (inv ntt+twist+out-transpose)
  C: device extrapolation via NTT16 ratios (L40S measured; others est,
     stage-mix-invariance assumed).
"""
import os, re, sys

# (lambda, c, tbl_w, t) from the user-provided derived table (N-invariant rows)
SEC = [(80, 2, 94, 64), (80, 4, 40, 16), (80, 8, 25, 4),
       (128, 2, 150, 128), (128, 4, 64, 16), (128, 8, 41, 8)]
LGNS = [20, 21, 22, 23, 24]
BATCHES = {2: 4, 4: 16, 8: 64}
DEVR = {"L40S": (1.00, "meas"), "H200": (0.53, "est"), "B200": (0.49, "est"),
        "A100": (0.73, "est"), "RTX5000Ada": (1.35, "est"),
        "RTXPRO6000": (0.70, "est")}
AGU_SAVE = 1.0    # input transpose removed exactly (agu_layout_bench: BITEXACT)


def parse(outdir, be, lg, b):
    d = {}
    try:
        txt = open(os.path.join(outdir, f"{be}_lg{lg}_b{b}.txt")).read()
        m = re.search(r"ms_per_mul=([\d.]+)", txt)
        if m:
            d["mul"] = float(m.group(1))
    except OSError:
        return None
    try:
        st = open(os.path.join(outdir, f"{be}_lg{lg}_b{b}_stages.txt")).read()
        lines = re.findall(r"\[(?:merge|4step)-stages\][^\n]*", st)
        if lines:
            last = lines[-1]
            for k in ("fwd_a", "fwd_b", "pointwise", "intt",
                      "transpose", "ntt", "twist"):
                m = re.search(rf"{k}=([\d.]+)", last)
                if m:
                    d[k] = float(m.group(1)) / b   # per-mul ms
    except OSError:
        pass
    return d if "mul" in d else None


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    P = {}
    for be in ("merge", "4step"):
        for lg in LGNS:
            for b in (4, 16, 64):
                r = parse(outdir, be, lg, b)
                if r:
                    P[(be, lg, b)] = r

    out = os.path.join(repo, "pim", "results", "l40s_ntt_secparams.txt")
    with open(out, "w") as f:
        f.write("== GPU NTT at SECURITY-CALIBRATED sizes (BCG+20 Table 1, fully-\n"
                "== splitting ring; derive_params). NTT size = N; batch = c^2.\n"
                "== L40S silicon (CUDA 12.6, patched GPUNTT 95c739c, batch-mode\n"
                "== cudaEvent phase-wall; stage splits via PCG_{MERGE,4STEP}_STAGE_MS).\n"
                "== tier: silicon (L40S); device extrapolation in C is est.\n==\n")

        f.write("---- A. raw per-mul ms (per (N, batch)); m/4 = merge/4step ----\n")
        f.write(f"{'N':>6} {'batch':>5} | {'merge':>9} {'4step':>9} {'m/4':>6} |"
                f" merge fwd/pw/intt      | 4step trans/ntt/twist/pw\n")
        for lg in LGNS:
            for b in (4, 16, 64):
                m, s = P.get(("merge", lg, b)), P.get(("4step", lg, b))
                if not m:
                    continue
                fwd = (m.get("fwd_a", 0) + m.get("fwd_b", 0)) / 2
                mstage = f"{fwd:.3f}/{m.get('pointwise',0):.3f}/{m.get('intt',0):.3f}"
                if s:
                    sstage = (f"{s.get('transpose',0):.3f}/{s.get('ntt',0):.3f}/"
                              f"{s.get('twist',0):.3f}/{s.get('pointwise',0):.3f}")
                    f.write(f"  2^{lg:<3} {b:>5} | {m['mul']:9.4f} {s['mul']:9.4f}"
                            f" {m['mul']/s['mul']:6.2f} | {mstage:22} | {sstage}\n")
                else:
                    f.write(f"  2^{lg:<3} {b:>5} | {m['mul']:9.4f} {'-':>9} {'-':>6}"
                            f" | {mstage:22} | -\n")
        f.write("\n")

        f.write("---- B. NTT lane per expansion, security rows (ms; L40S) ----\n"
                "  (i) serial = 2c^2 full mul   (ii) Move2 merge = c^2(fwd+pw)+INTT\n"
                "  (iii) Move2+AGU 4step = c^2(ntt+twist+pw) + inv side (no input\n"
                "        transpose: agu_layout_bench BITEXACT)\n")
        f.write(f"{'lam':>4} {'N':>6} {'c':>2} {'t':>4} {'batch':>5} |"
                f" {'(i)serial':>10} {'(ii)Move2':>10} {'(iii)AGU4s':>10} | best\n")
        for lam, c, w, t in SEC:
            b = BATCHES[c]
            for lg in LGNS:
                m, s = P.get(("merge", lg, b)), P.get(("4step", lg, b))
                if not m:
                    continue
                fwd = (m.get("fwd_a", 0) + m.get("fwd_b", 0)) / 2
                i1 = 2 * c * c * m["mul"]
                i2 = c * c * (fwd + m.get("pointwise", 0)) + m.get("intt", 0)
                if s and "ntt" in s:
                    # forward side per mul without input transpose; the 4step
                    # 'ntt'/'twist' buckets lump fwd+inv -> take half as fwd
                    fwd4 = (s["ntt"] + s["twist"]) / 2 + s.get("pointwise", 0)
                    inv4 = (s["ntt"] + s["twist"]) / 2 + s.get("transpose", 0) / 3
                    i3 = c * c * fwd4 + inv4
                else:
                    i3 = float("nan")
                best = min(("serial", i1), ("move2", i2),
                           ("agu4s", i3) if i3 == i3 else ("agu4s", 1e30),
                           key=lambda kv: kv[1])[0]
                f.write(f"{lam:>4} 2^{lg:<4} {c:>2} {t:>4} {b:>5} |"
                        f" {i1:10.3f} {i2:10.3f} {i3:10.3f} | {best}\n")
            f.write("\n")

        f.write("---- C. device extrapolation (NTT16 ratios; stage-mix assumed"
                " invariant) ----\n"
                "  headline row lam=128 c=4 t=16, Move2 merge lane, ms:\n")
        f.write(f"{'N':>6} |" + "".join(f" {d:>11}" for d in DEVR) + "\n")
        for lg in LGNS:
            m = P.get(("merge", lg, 16))
            if not m:
                continue
            fwd = (m.get("fwd_a", 0) + m.get("fwd_b", 0)) / 2
            lane = 16 * (fwd + m.get("pointwise", 0)) + m.get("intt", 0)
            f.write(f"  2^{lg:<3} |" + "".join(
                f" {lane*r:9.3f}{'m' if tag=='meas' else 'e':>2}"
                for d, (r, tag) in DEVR.items()) + "\n")
    print(f"wrote {out}  ({len(P)} points)")


if __name__ == "__main__":
    main()
