#!/bin/bash
# B200, one job: (1) run_cheddar_bigN.sh at logN 20..28 (Cheddar and the four-step on Cheddar's kernels swept over the same
# config range, register caps 0/128/64, every product checked); (2) run_cheddar_interfere.sh at logN 20..28, each design at
# its best config from (1), beside the stream / SPU-like row-burst / sleep-only aggressors.
# Submit from GPU_baseline/cheddar: sbatch job_cheddar_b200_all.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 08:00:00
#SBATCH -J cheddar-b200-all
#SBATCH -o cheddar_b200_all_%j.out
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
export ARCH=sm_100 OUT=../results_b200 TAG=b200 NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}" GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib64}"
LOGNS="20 21 22 23 24 25 26 27 28" ./run_cheddar_bigN.sh
LOGNS="20 21 22 23 24 25 26 27 28" BLOCKS=8,16,32,64,96,128 ./run_cheddar_interfere.sh
