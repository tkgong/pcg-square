#!/bin/bash
# Submit from GPU_baseline/fused4 of the clone:   cd GPU_baseline/fused4 && sbatch job_interfere_ntt_b200.sbatch
# Victims: GPU-NTT merge, four-step SM lane (DRU design), four-step with SM transposes; aggressors: streaming,
# SPU-like row-burst, sleep-only control; blocks sweep the aggressor bandwidth up to the SPU's ~20% share.
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J interfere-ntt-b200
#SBATCH -o interfere_ntt_b200_%j.out
set -eu
NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
ls "$GPUNTT_LIB"/libntt-1.0.* >/dev/null 2>&1 || GPUNTT_LIB=$(dirname "$(find "$HOME" -maxdepth 6 -name 'libntt-1.0.*' | head -1)")
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
mkdir -p bin results
"$NVCC" -O3 -std=c++17 -arch=sm_100 -I.. -I"$GPUNTT_INC" interfere_ntt.cu -L"$GPUNTT_LIB" -lntt-1.0 -o bin/interfere_ntt_sm100
for lg in 22 24; do
  ./bin/interfere_ntt_sm100 --logN $lg --batch 16 --blocks 8,16,32,64,128,256,512 --iters 10 > results/interfere_ntt_B200_$lg.csv
done
