#!/bin/bash
# B200: Cheddar's NTT vs GPU-NTT merge (HEonGPU's NTT) at PCG's sizes (run_cheddar.sh). Submit from GPU_baseline/cheddar: sbatch job_cheddar_b200.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 01:30:00
#SBATCH -J cheddar-b200
#SBATCH -o cheddar_b200_%j.out
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
ARCH=sm_100 OUT=../results_b200 TAG=b200 NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}" ./run_cheddar.sh
