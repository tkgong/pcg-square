#!/bin/bash
# Submit from GPU_baseline/fused4:   sbatch job_fused4_b200.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J fused4-b200
#SBATCH -o fused4_b200_%j.out
set -eu
export NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
bash ./run_fused4.sh
