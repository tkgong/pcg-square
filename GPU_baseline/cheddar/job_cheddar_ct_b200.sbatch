#!/bin/bash
# B200, one job: the PCG-output NTT designs on Cheddar's kernels (cheddar_bigN_bench.cu --designs ole,ole_r,olep,olepb):
# z = INTT( sum_ij a^_i a^_j NTT(g_ij) ), swept over Cheddar's own config space at logN 20..24, batch 4/16/64, register caps
# 0/128/64, every output checked against merge; then the interference run of the ole / ole_t victims beside the aggressors.
# Rows are appended to ../results_b200/cheddar_bigN_b200.csv (same columns as the Cheddar rows already there).
# Submit from GPU_baseline/cheddar: sbatch job_cheddar_ct_b200.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 06:00:00
#SBATCH -J cheddar-ole-b200
#SBATCH -o cheddar_ct_b200_%j.out
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
export ARCH=sm_100 OUT=../results_b200 TAG=b200 NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}" GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib64}"
LOGNS="20 21 22 23 24" DESIGNS=cheddar,cheddar_t SUF=_ct ./run_cheddar_ole.sh
