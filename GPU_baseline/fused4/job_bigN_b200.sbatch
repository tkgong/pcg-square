#!/bin/bash
# B200: (1) GPU-NTT merge forward transform 2^20..2^28 (merge_scale_bench), (2) four-step vs merge poly-mul at
# 2^24..2^28, batch 1, bit-exact, headline (GPU-NTT sub-transform kernels) and own single-kernel sub-transforms with
# 2^14 tiles (128 KB shared memory). Submit from GPU_baseline/fused4: sbatch job_bigN_b200.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -J bigN-b200
#SBATCH -o bigN_b200_%j.out
set -eu
NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
mkdir -p bin ../results_b200
"$NVCC" -O3 -std=c++17 -arch=sm_100 -I"$GPUNTT_INC" merge_scale_bench.cu -L"$GPUNTT_LIB" -lntt-1.0 -o bin/merge_scale_bench_sm_100
for r in 1 2 3; do ./bin/merge_scale_bench_sm_100 --logN 20,21,22,23,24,25,26,27,28 --iters 5; done | tee ../results_b200/merge_scale_b200.csv
"$NVCC" -O3 -std=c++17 -arch=sm_100 -IbigN -I.. -I"$GPUNTT_INC" bigN/fused4_bigN_bench.cu -L"$GPUNTT_LIB" -lntt-1.0 -o bin/fused4_bigN_bench_sm_100
for r in 1 2 3; do ./bin/fused4_bigN_bench_sm_100 --logN 24,25,26,27,28 --iters 3 --prime 4611685989973229569 --max-tile 14; done | tee ../results_b200/fused4_bigN_b200.csv
