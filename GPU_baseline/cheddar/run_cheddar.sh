#!/bin/bash
# Cheddar (scale-snu/cheddar-fhe) NTT vs GPU-NTT merge (HEonGPU's NTT) at PCG's sizes.
#  1. clone Cheddar at a pinned commit and extract its NTT kernels (extract.sh: config as a template parameter, one twiddle table),
#  2. cheddar_ntt_bench: forward NTT, every feasible two-kernel config at logN 16..24, checked against a CPU reference, 1 GB of data,
#  3. merge_scale_bench (../fused4) at the same sizes and bytes,
#  4. cheddar_polymul_bench: whole negacyclic multiply at logN 20..24, batch 4/16/64, merge vs the 3 best Cheddar configs,
#     every Cheddar product compared element-wise with the merge product.
# Usage: ARCH=sm_89 OUT=../results_l40s TAG=l40s ./run_cheddar.sh      (needs nvcc >= 11.8 and GPU-NTT with the 62-bit Barrett fix)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ARCH="${ARCH:-sm_89}"; OUT="${OUT:-../results_l40s}"; TAG="${TAG:-l40s}"
NVCC="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"; GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
COMMIT=8df8b26
[ -d cheddar-fhe ] || git clone -q https://github.com/scale-snu/cheddar-fhe.git cheddar-fhe
git -C cheddar-fhe checkout -q "$COMMIT"
mkdir -p gen bin "$OUT"
./extract.sh cheddar-fhe gen/cheddar_ntt_kernels.cuh
FLAGS=(-O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -diag-suppress 177 -Ishim -Icheddar-fhe/include -I.)
"$NVCC" "${FLAGS[@]}" cheddar_ntt_bench.cu -o "bin/cheddar_ntt_bench_$ARCH"
"./bin/cheddar_ntt_bench_$ARCH" --logN 16,17,18,19,20,21,22,23,24 --iters 10 | tee "$OUT/cheddar_ntt_sweep_$TAG.csv"
"$NVCC" -O3 -std=c++17 -arch="$ARCH" -I"$GPUNTT_INC" ../fused4/merge_scale_bench.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/merge_scale_bench_$ARCH"
"./bin/merge_scale_bench_$ARCH" --logN 16,17,18,19,20,21,22,23,24 --iters 10 --noref | tee "$OUT/merge_scale_16_24_$TAG.csv"
python3 gen_cfgs.py "$OUT/cheddar_ntt_sweep_$TAG.csv" gen/cfgs.h
"$NVCC" "${FLAGS[@]}" -I.. -I"$GPUNTT_INC" cheddar_polymul_bench.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/cheddar_polymul_bench_$ARCH"
"./bin/cheddar_polymul_bench_$ARCH" --iters 5 | tee "$OUT/cheddar_polymul_$TAG.csv"
# 5. four-step NTT on Cheddar's kernels (column kernel with contiguous addressing, extract_contig.sh): SM lane with the six
#    transposes on the DRU, and the same with the transposes on the SMs; vs Cheddar and merge in the same run, products checked.
./extract_contig.sh gen/cheddar_ntt_kernels.cuh gen/cheddar_ntt_contig.cuh
"$NVCC" "${FLAGS[@]}" -I.. -I"$GPUNTT_INC" cheddar_f4dru_bench.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/cheddar_f4dru_bench_$ARCH"
"./bin/cheddar_f4dru_bench_$ARCH" --iters 5 | tee "$OUT/cheddar_f4dru_$TAG.csv"
