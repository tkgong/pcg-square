#!/bin/bash
# PCG output on Cheddar's kernels (cheddar_bigN_bench.cu --designs ole): z = INTT( sum_ij a^_i a^_j NTT(g_ij) ) per output,
# designs ole (g_ij in Cheddar's natural layout) and ole_t (g_ij in the column-contiguous layout the SPUs write), swept over
# the same Cheddar config space (P2, M1, B1, M2, register caps) at batch c^2 = 4, 16, 64 and logN 20..24, every z checked
# against merge computed one product at a time. Rows are APPENDED to cheddar_bigN_<tag>.csv (same columns; ms_per_poly =
# time per output / c^2). Runs (logN x cap) are spread over the GPUs in GPUS, one at a time per GPU.
# Usage: GPUS="0 1 2 3" LOGNS="20 21 22 23 24" [DESIGNS=ole,olep] [EXTRA=--ext] ./run_cheddar_ole.sh   (ARCH=sm_100 OUT=../results_b200 TAG=b200 on B200)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ARCH="${ARCH:-sm_89}"; OUT="${OUT:-../results_l40s}"; TAG="${TAG:-l40s}"; LOGNS="${LOGNS:-20 21 22 23 24}"; REGCAPS="${REGCAPS:-0 128 64}"; GPUS="${GPUS:-0}"; DESIGNS="${DESIGNS:-ole}"; EXTRA="${EXTRA:-}"; SUF="${SUF:-}"   # DESIGNS=ole,olep,olepb; EXTRA=--ext (only M=6 / B1=4 configs); SUF: write to cheddar_bigN_<tag><SUF>.csv (no summary)
NVCC="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"; GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
[ -d cheddar-fhe ] || git clone -q https://github.com/scale-snu/cheddar-fhe.git cheddar-fhe
git -C cheddar-fhe checkout -q 8df8b26
mkdir -p gen bin "$OUT"
./extract.sh cheddar-fhe gen/cheddar_ntt_kernels.cuh
./extract_contig.sh gen/cheddar_ntt_kernels.cuh gen/cheddar_ntt_contig.cuh
python3 extract_acc.py gen/cheddar_ntt_kernels.cuh gen/cheddar_ntt_acc.cuh
FLAGS=(-O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -diag-suppress 177 -Ishim -Icheddar-fhe/include -I. -I.. -I"$GPUNTT_INC")
for L in $LOGNS; do for R in $REGCAPS; do
  CAP=(); [ "$R" -gt 0 ] && CAP=(-maxrregcount="$R")
  "$NVCC" "${FLAGS[@]}" "${CAP[@]}" -DBIGN_LOGN=$L -DBIGN_MAXRREG=$R cheddar_bigN_bench.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/cheddar_bigN_${L}_r${R}_$ARCH" &
done; done
wait
RUNS=(); for L in $LOGNS; do for R in $REGCAPS; do RUNS+=("$L $R"); done; done
read -ra G <<< "$GPUS"; NG=${#G[@]}
worker() {
  local gi=$1 gpu=$2 i L R
  for ((i = gi; i < ${#RUNS[@]}; i += NG)); do
    read -r L R <<< "${RUNS[$i]}"
    CUDA_VISIBLE_DEVICES=$gpu "./bin/cheddar_bigN_${L}_r${R}_$ARCH" --designs "$DESIGNS" $EXTRA --batch 4,16,64 > "bin/ole_${L}_r$R.csv" \
      || echo "logN $L maxrregcount $R: bench exited with $?" >&2
    echo "gpu $gpu: logN $L cap $R done ($(($(wc -l < "bin/ole_${L}_r$R.csv") - 1)) rows)" >&2
  done
}
for gi in "${!G[@]}"; do worker "$gi" "${G[$gi]}" & done
wait
CSV="$OUT/cheddar_bigN_$TAG$SUF.csv"
[ -n "$SUF" ] && [ ! -s "$CSV" ] && head -1 "bin/ole_${LOGNS%% *}_r${REGCAPS%% *}.csv" > "$CSV"
for L in $LOGNS; do for R in $REGCAPS; do tail -n +2 "bin/ole_${L}_r$R.csv" >> "$CSV"; done; done
[ -z "$SUF" ] && python3 summarize_bigN.py "$TAG"
true
