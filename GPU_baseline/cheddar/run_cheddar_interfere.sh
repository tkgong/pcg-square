#!/bin/bash
# Silicon interference sweep of Cheddar's NTT and the four-step on Cheddar's kernels (cheddar_interfere.cu). Configs come
# from cheddar_bigN_<tag>.csv (run_cheddar_bigN.sh first: same sweep for both designs, register caps included): for each
# design and each column stage merging M1 in 3..5, the best config with that M1 (a config that is slower alone can be less
# sensitive to DRAM interference), so each design's best co-run time is measured, not assumed from its best standalone config.
# Batch 16 where 6 buffers of batch x N words fit in GPU memory with 8 GB to spare, else batch 4. Each config runs in the
# build of its register cap (rebuilt every run); merge in the uncapped build; the four-step's best standalone config also
# times the full four-step. Each run gets 30 min (an aggressor of 128 blocks once hung the L40S at 2^20); BLOCKS 8..64
# reach ~70% of L40S peak.
# f4b_sm (butterfly DRU: row kernel only on the SMs) gets its best config per row M2 in 3..5, checked through Cheddar's product.
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
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["mismatches"] == "0" and r["design"] in ("cheddar", "f4c_sm", "f4b_sm")]
mem = float(sys.argv[2]) * 2**20
for L in map(int, sys.argv[3:]):
    b = 16 if 6 * 8 * 16 * 2**L + 8 * 2**30 <= mem else 4
    R = [r for r in rows if int(r["logN"]) == L and int(r["batch"]) == b]
    if not R: print(f"# logN {L} batch {b}: no cheddar or f4c_sm row in the big-N csv", file=sys.stderr); continue
    print(L, b, 0, "merge", "-", "-", "-")
    for d in ("cheddar", "f4c_sm", "f4b_sm"):
        D = [r for r in R if r["design"] == d]
        if not D: continue
        top = min(D, key=lambda r: float(r["ms_per_poly"]))
        for m in ("3", "4", "5"):   # best config per column M1 (per row M2 for f4b_sm, which has no column kernel)
            M = [r for r in D if r["M2" if d == "f4b_sm" else "M1"] == m]
            if not M: continue
            r = min(M, key=lambda r: float(r["ms_per_poly"]))
            if d == "cheddar": print(L, b, r["maxrreg"], "cheddar", f"{r['P1']},{r['M1']},{r['B1']},{r['M2']}", "-", "-")
            elif d == "f4c_sm": print(L, b, r["maxrreg"], "f4c" if r is top else "f4csm", "-", f"{r['P1']},{r['M1']},{r['B1']},{r['M2']}", "-")
            else: print(L, b, r["maxrreg"], "f4b", "-", "-", f"{r['P1']},{r['M2']}")
PY
)
echo "$PLAN"
FLAGS=(-O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -diag-suppress 177 -Ishim -Icheddar-fhe/include -I. -I.. -I"$GPUNTT_INC")
while read -r L R; do   # one binary per (logN, register cap), built in parallel
  CAP=(); [ "$R" -gt 0 ] && CAP=(-maxrregcount="$R")
  "$NVCC" "${FLAGS[@]}" "${CAP[@]}" -DBIGN_LOGN=$L -DBIGN_MAXRREG=$R cheddar_interfere.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/cheddar_interfere_${L}_r${R}_$ARCH" &
done <<< "$(awk '{print $1, $3}' <<< "$PLAN" | sort -u)"
wait
CSV="$OUT/cheddar_interfere_$TAG.csv"; : > "$CSV"; I=0
while read -r L B R V C F G; do
  I=$((I + 1)); O="bin/interfere_${TAG}_${I}.csv"
  timeout 1800 "./bin/cheddar_interfere_${L}_r${R}_$ARCH" --batch "$B" --victims "$V" --cheddar "$C" --f4c "$F" --f4b "$G" --blocks "$BLOCKS" --iters 10 \
    > "$O" || echo "logN $L cap $R ($V $C $F $G): bench exited with $?" >&2
  if [ -s "$CSV" ]; then tail -n +2 "$O"; else cat "$O"; fi >> "$CSV"
done <<< "$PLAN"
python3 interfere_summary.py "$TAG"
