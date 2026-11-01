#!/bin/bash
# Silicon interference sweep of Cheddar's NTT and the four-step on Cheddar's kernels (cheddar_interfere.cu), each design at
# its best config of cheddar_bigN_<tag>.csv (run_cheddar_bigN.sh first: same sweep for both designs, register caps included).
# Batch 16 where 6 buffers of batch x N words fit in GPU memory with 8 GB to spare, else batch 4. Each design runs in the
# build of its register cap; merge in the uncapped build. Each run gets 30 min (an aggressor of 128 blocks once hung the L40S
# at 2^20); BLOCKS 8..64 reach ~70% of L40S peak.
# Usage: ARCH=sm_100 OUT=../results_b200 TAG=b200 LOGNS="20 21 22 23 24 25 26 27 28" BLOCKS=8,16,32,64,96,128 ./run_cheddar_interfere.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ARCH="${ARCH:-sm_89}"; OUT="${OUT:-../results_l40s}"; TAG="${TAG:-l40s}"; LOGNS="${LOGNS:-20 21 22 23 24 25 26}"; BLOCKS="${BLOCKS:-8,16,32,48,64}"
NVCC="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"; GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
[ -d cheddar-fhe ] || git clone -q https://github.com/scale-snu/cheddar-fhe.git cheddar-fhe
git -C cheddar-fhe checkout -q 8df8b26
mkdir -p gen bin "$OUT"
./extract.sh cheddar-fhe gen/cheddar_ntt_kernels.cuh
./extract_contig.sh gen/cheddar_ntt_kernels.cuh gen/cheddar_ntt_contig.cuh
MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | head -1)
# plan: one line per run "logN batch cap victims cheddar_cfg f4c_cfg"
PLAN=$(python3 - "$OUT/cheddar_bigN_$TAG.csv" "$MEM_MB" $LOGNS <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["mismatches"] == "0" and r["design"] in ("cheddar", "f4c_sm")]
mem = float(sys.argv[2]) * 2**20
for L in map(int, sys.argv[3:]):
    b = 16 if 6 * 8 * 16 * 2**L + 8 * 2**30 <= mem else 4
    best = {}
    for d in ("cheddar", "f4c_sm"):
        R = [r for r in rows if int(r["logN"]) == L and int(r["batch"]) == b and r["design"] == d]
        if R: best[d] = min(R, key=lambda r: float(r["ms_per_poly"]))
    if not best: print(f"# logN {L} batch {b}: no cheddar or f4c_sm row in the big-N csv", file=sys.stderr); continue
    c, f = best.get("cheddar"), best.get("f4c_sm")
    cc = f"{c['P1']},{c['M1']},{c['B1']},{c['M2']}" if c else "-"; fc = f"{f['P1']},{f['M1']},{f['M2']}" if f else "-"
    runs = {"0": ["merge"]}
    if c: runs.setdefault(c["maxrreg"], []).append("cheddar")
    if f: runs.setdefault(f["maxrreg"], []).append("f4c")
    for cap, v in sorted(runs.items()): print(L, b, cap, ",".join(v), cc, fc)
PY
)
echo "$PLAN"
FLAGS=(-O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -diag-suppress 177 -Ishim -Icheddar-fhe/include -I. -I.. -I"$GPUNTT_INC")
while read -r L B R V C F; do   # one binary per (logN, register cap), built in parallel
  CAP=(); [ "$R" -gt 0 ] && CAP=(-maxrregcount="$R")
  [ -x "bin/cheddar_interfere_${L}_r${R}_$ARCH" ] || "$NVCC" "${FLAGS[@]}" "${CAP[@]}" -DBIGN_LOGN=$L -DBIGN_MAXRREG=$R cheddar_interfere.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/cheddar_interfere_${L}_r${R}_$ARCH" &
done <<< "$PLAN"
wait
CSV="$OUT/cheddar_interfere_$TAG.csv"; : > "$CSV"
while read -r L B R V C F; do
  timeout 1800 "./bin/cheddar_interfere_${L}_r${R}_$ARCH" --batch "$B" --victims "$V" --cheddar "$C" --f4c "$F" --blocks "$BLOCKS" --iters 10 \
    > "bin/interfere_${L}_r$R.csv" || echo "logN $L cap $R ($V): bench exited with $?" >&2
  if [ -s "$CSV" ]; then tail -n +2 "bin/interfere_${L}_r$R.csv"; else cat "bin/interfere_${L}_r$R.csv"; fi | tee -a "$CSV"
done <<< "$PLAN"
python3 interfere_summary.py "$TAG"
