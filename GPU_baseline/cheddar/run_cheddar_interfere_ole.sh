#!/bin/bash
# Silicon interference sensitivity of the PCG-output NTT (cheddar_interfere.cu victims ole / ole_t): at each logN (batch 16 =
# c^2, c = 4) the best standalone ole and ole_t configs of cheddar_bigN_<tag>.csv, timed alone and beside the streaming, the
# SPU-like row-burst and the sleep-only control aggressor (same protocol and BLOCKS as run_cheddar_interfere.sh), each in its
# own invocation (ole_t's g_ij are in the SPUs' column-contiguous layout). Runs are spread over GPUS.
# Usage: GPUS="0 1 2 3" LOGNS="20 21 22 23 24" ./run_cheddar_interfere_ole.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ARCH="${ARCH:-sm_89}"; OUT="${OUT:-../results_l40s}"; TAG="${TAG:-l40s}"; LOGNS="${LOGNS:-20 21 22 23 24}"; BLOCKS="${BLOCKS:-8,16,32,48,64}"; GPUS="${GPUS:-0}"
NVCC="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"; GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
[ -d cheddar-fhe ] || git clone -q https://github.com/scale-snu/cheddar-fhe.git cheddar-fhe
git -C cheddar-fhe checkout -q 8df8b26
mkdir -p gen bin "$OUT"
./extract.sh cheddar-fhe gen/cheddar_ntt_kernels.cuh
./extract_contig.sh gen/cheddar_ntt_kernels.cuh gen/cheddar_ntt_contig.cuh
python3 extract_acc.py gen/cheddar_ntt_kernels.cuh gen/cheddar_ntt_acc.cuh
# plan: "logN cap victim P1,M1,B1,M2" = the best standalone config of each design at batch 16
PLAN=$(python3 - "$OUT/cheddar_bigN_$TAG.csv" $LOGNS <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["mismatches"] == "0" and r["batch"] == "16" and r["design"] in ("ole", "ole_t")]
for L in map(int, sys.argv[2:]):
    for d in ("ole", "ole_t"):
        D = [r for r in rows if int(r["logN"]) == L and r["design"] == d]
        if not D: print(f"# logN {L}: no {d} row", file=sys.stderr); continue
        r = min(D, key=lambda r: float(r["ms_per_poly"]))
        print(L, r["maxrreg"], d, f"{r['P1']},{r['M1']},{r['B1']},{r['M2']}")
PY
)
echo "$PLAN"
FLAGS=(-O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -diag-suppress 177 -Ishim -Icheddar-fhe/include -I. -I.. -I"$GPUNTT_INC")
while read -r L R; do
  CAP=(); [ "$R" -gt 0 ] && CAP=(-maxrregcount="$R")
  "$NVCC" "${FLAGS[@]}" "${CAP[@]}" -DBIGN_LOGN=$L -DBIGN_MAXRREG=$R cheddar_interfere.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/cheddar_interfere_${L}_r${R}_$ARCH" &
done <<< "$(awk '{print $1, $2}' <<< "$PLAN" | sort -u)"
wait
read -ra G <<< "$GPUS"; NG=${#G[@]}
worker() {
  local gi=$1 gpu=$2 I=0 L R V C
  while read -r L R V C; do
    I=$((I + 1)); [ $(( (I - 1) % NG )) -eq "$gi" ] || continue
    FLAG=--ole; [ "$V" = ole_t ] && FLAG=--olet
    CUDA_VISIBLE_DEVICES=$gpu timeout 1800 "./bin/cheddar_interfere_${L}_r${R}_$ARCH" --batch 16 --victims "$V" $FLAG "$C" --blocks "$BLOCKS" --iters 10 \
      > "bin/interfere_ole_${TAG}_${I}.csv" || echo "run $I (logN $L cap $R $V $C): bench exited with $?" >&2
    echo "gpu $gpu run $I done: logN $L cap $R $V $C" >&2
  done <<< "$PLAN"
}
for gi in "${!G[@]}"; do worker "$gi" "${G[$gi]}" & done
wait
CSV="$OUT/cheddar_interfere_ole_$TAG.csv"
echo "gpu,victim,logN,batch,P1,M1,B1,M2,maxrreg,blocks,sleep_ns,mode,agg_GBps,victim_ms,slowdown_raw,mismatches" > "$CSV"
I=0; while read -r L R V C; do I=$((I + 1)); tail -n +2 "bin/interfere_ole_${TAG}_${I}.csv" >> "$CSV"; done <<< "$PLAN"
python3 interfere_ole_summary.py "$TAG"
