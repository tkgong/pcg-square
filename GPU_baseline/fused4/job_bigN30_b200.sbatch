#!/bin/bash
# B200: HEonGPU/GPU-NTT merge vs four-step (GPU-NTT sub-transform kernels + DRU transposes) at 2^24..2^30.
# GPU-NTT's own table stops at 2^28; bigN30/gpuntt_2e29_2e30.patch adds 3-pass (10+9+10, 10+10+10) configurations
# with the same 1024-element tiles and relaxes the host range check. The patched library is built into a local prefix.
# Submit from GPU_baseline/fused4:  sbatch job_bigN30_b200.sbatch
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 02:00:00
#SBATCH -J bigN30-b200
#SBATCH -o bigN30_b200_%j.out
set -eu
NVCC="${NVCC:-$HOME/opt/cuda-13.0/bin/nvcc}"
GPUNTT_SRC="${GPUNTT_SRC:-$HOME/GPU-NTT}"            # the same checkout the other jobs use (with the 62-bit Barrett shift fix)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi
EXT="$PWD/bigN30/gpuntt_ext"; rm -rf "$EXT"; cp -r "$GPUNTT_SRC" "$EXT"; rm -rf "$EXT/build" "$EXT/install"
( cd "$EXT" && patch -p0 < "$OLDPWD/bigN30/gpuntt_2e29_2e30.patch" && mkdir -p build && cd build &&
  cmake .. -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_INSTALL_PREFIX="$EXT/install" > cmake.log 2>&1 &&
  make -j16 ntt > make.log 2>&1 && make install >> make.log 2>&1 )
mkdir -p bin ../results_b200
"$NVCC" -O3 -std=c++17 -arch=sm_100 -IbigN30 -I.. -I"$EXT/install/include/GPUNTT-1.0" bigN30/fused4_bigN_bench.cu -L"$EXT/install/lib" -lntt-1.0 -o bin/fused4_bigN30_sm_100
for r in 1 2 3; do ./bin/fused4_bigN30_sm_100 --logN 24,25,26,27,28,29,30 --iters 3 --prime 4611685941117976577 --max-tile 14; done | tee ../results_b200/fused4_bigN30_b200.csv
