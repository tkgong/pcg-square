#!/bin/bash
# B200: silicon interference sweep of Cheddar's NTT, the four-step on Cheddar's kernels (SM lane / transposes on SMs) and
# GPU-NTT merge at logN 20..24, batch 16 (run_cheddar_interfere.sh; configs from results_b200/cheddar_f4dru_b200.csv).
# Submit from GPU_baseline/cheddar: sbatch job_cheddar_interfere_b200.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J cheddar-interfere-b200
#SBATCH -o cheddar_interfere_b200_%j.out
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
ARCH=sm_100 OUT=../results_b200 TAG=b200 BLOCKS=8,16,32,64,96,128 NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}" GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib64}" ./run_cheddar_interfere.sh
