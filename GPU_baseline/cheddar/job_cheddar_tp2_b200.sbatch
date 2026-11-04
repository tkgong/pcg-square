#!/bin/bash
# B200, one job: the two-pass streaming NTT (two_pass_ntt.cuh, second version: merged twiddle slices, Montgomery b^,
# prefetch) and the Cheddar-kernel pipeline olepb at logN 20..24, batch 4/16/64, register cap 0 (tp ignores caps), every
# output checked against the paired reference.  Rows go to ../results_b200/cheddar_bigN_b200_tp2.csv.
# Submit from GPU_baseline/cheddar: sbatch job_cheddar_tp2_b200.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 04:00:00
#SBATCH -J cheddar-tp2-b200
#SBATCH -o cheddar_tp2_b200_%j.out
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
export ARCH=sm_100 OUT=../results_b200 TAG=b200 NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}" GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib64}"
LOGNS="20 21 22 23 24" REGCAPS="0" DESIGNS=tp,tpt,olepb SUF=_tp2 ./run_cheddar_ole.sh
