#!/bin/bash
# B200: Cheddar vs four-step on Cheddar's kernels (DRU transposes) vs GPU-NTT merge at logN 24..28 (run_cheddar_bigN.sh).
# Submit from GPU_baseline/cheddar: sbatch job_cheddar_bigN_b200.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 04:00:00
#SBATCH -J cheddar-bigN-b200
#SBATCH -o cheddar_bigN_b200_%j.out
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
ARCH=sm_100 OUT=../results_b200 TAG=b200 NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}" GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib64}" ./run_cheddar_bigN.sh
