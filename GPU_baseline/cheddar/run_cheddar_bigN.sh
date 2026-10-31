#!/bin/bash
# Cheddar's two-kernel NTT vs the four-step on Cheddar's kernels (transposes on the DRU) vs GPU-NTT merge at logN 24..28,
# per negacyclic multiply (cheddar_bigN_bench.cu: configs swept per logN, every product checked against merge computed one
# polynomial at a time). 2^24 uses PCG's prime (overlap with cheddar_f4dru_<tag>.csv), 2^25..2^28 a 62-bit prime with 2-adicity 33.
# Usage: ARCH=sm_100 OUT=../results_b200 TAG=b200 ./run_cheddar_bigN.sh   (LOGNS="24 25 26 27 28" by default)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ARCH="${ARCH:-sm_89}"; OUT="${OUT:-../results_l40s}"; TAG="${TAG:-l40s}"; LOGNS="${LOGNS:-24 25 26 27 28}"
NVCC="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"; GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
COMMIT=8df8b26
[ -d cheddar-fhe ] || git clone -q https://github.com/scale-snu/cheddar-fhe.git cheddar-fhe
git -C cheddar-fhe checkout -q "$COMMIT"
mkdir -p gen bin "$OUT"
./extract.sh cheddar-fhe gen/cheddar_ntt_kernels.cuh
./extract_contig.sh gen/cheddar_ntt_kernels.cuh gen/cheddar_ntt_contig.cuh
FLAGS=(-O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -diag-suppress 177 -Ishim -Icheddar-fhe/include -I. -I.. -I"$GPUNTT_INC")
for L in $LOGNS; do   # one binary per logN, built in parallel
  "$NVCC" "${FLAGS[@]}" -DBIGN_LOGN=$L cheddar_bigN_bench.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/cheddar_bigN_${L}_$ARCH" &
done
wait
CSV="$OUT/cheddar_bigN_$TAG.csv"; : > "$CSV"
for L in $LOGNS; do
  B="4,16,64"; [ "$L" -ge 27 ] && B="4,16"       # batch 64 does not fit at 2^27 and 2^28 (4 arrays of batch x N words)
  # a failing size is reported and the next one still runs
  "./bin/cheddar_bigN_${L}_$ARCH" --batch "$B" > "bin/bigN_${L}.csv" || echo "logN $L: bench exited with $?" >&2
  if [ -s "$CSV" ]; then tail -n +2 "bin/bigN_${L}.csv"; else cat "bin/bigN_${L}.csv"; fi | tee -a "$CSV"
done
python3 summarize_bigN.py "$TAG"
