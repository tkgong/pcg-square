#!/bin/bash
# GPU-NTT merge forward transform 2^20..2^28 on B200 (merge_scale_bench.cu): per-pass cost vs N and the
# 4-kernel step at 2^27 of the non-sm_89 GPU-NTT table. Submit from GPU_baseline/fused4: sbatch job_merge_scale_b200.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J merge-scale-b200
#SBATCH -o merge_scale_b200_%j.out
set -eu
NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
mkdir -p bin ../results_b200
"$NVCC" -O3 -std=c++17 -arch=sm_100 -I"$GPUNTT_INC" merge_scale_bench.cu -L"$GPUNTT_LIB" -lntt-1.0 -o bin/merge_scale_bench_sm_100
for r in 1 2 3; do ./bin/merge_scale_bench_sm_100 --logN 20,21,22,23,24,25,26,27,28 --iters 5; done | tee ../results_b200/merge_scale_b200.csv
