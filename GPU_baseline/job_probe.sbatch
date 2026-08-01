#!/bin/bash
# 15-minute reconnaissance BEFORE committing to the 4-8 h full run.
# run_pcg_baseline.sh has no resume: if it dies at hour 6, everything is lost.
# So first establish, on real hardware:
#   (a) the toolchain overrides are right (nvcc at ~/opt, libntt under lib64),
#   (b) the square correctness gate actually passes,
#   (c) whether the lambda=80/c=8/t=4 row faults. The script passes B=c^2, so
#       that row gets B=64 with t=4; out_scatter_kernel computes
#       pos_base=(b/t + b%t)*bin_sz with no bounds check, giving pos_base up to
#       18*(D/2) against gN = t*(D/2) = 2D. One X^N=-1 fold cannot bring that
#       back in range -> out-of-bounds device writes on g for 75 of the 450
#       DPF cells. Need to know if that faults or corrupts silently.
#   (d) per-cell wall time at the largest sizes, to size the real job's -t.
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:2
#SBATCH -c 32
#SBATCH --mem=300G
#SBATCH -t 0:25:00
#SBATCH -J pcg-probe
#SBATCH -o /home/gong_ten_neu/PCG-acceleration/pcg-square/GPU_baseline/job_probe_%j.out

set -u
export PATH="$HOME/opt/cuda-13.0/bin:$PATH"
export NVCC="$HOME/opt/cuda-13.0/bin/nvcc"
export GPUNTT_INC="$HOME/.local/include/GPUNTT-1.0"
export GPUNTT_LIB="$HOME/.local/lib64"
HERE=/home/gong_ten_neu/PCG-acceleration/pcg-square/GPU_baseline
cd "$HERE"
P="$HERE/probe_out"; mkdir -p "$P" "$HERE/bin"

echo "===== env ====="
hostname; date -Is; nproc
git -C "$HERE" rev-parse HEAD
squeue -w "$(hostname -s)" -o "%.10i %.14u %.20j %.8T %.4C %b" 2>&1
nvidia-smi --query-gpu=index,name,driver_version,memory.total,utilization.gpu,memory.used --format=csv
nvidia-smi topo -m 2>&1 | head -8

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d ' .')
ARCH="sm_${CAP}"
NC="-O3 -std=c++17 -arch=$ARCH -I$HERE -I$GPUNTT_INC"
SQ="$HERE/common/poly_mul_gpuntt_square.cu"

echo; echo "===== build (same flags the runner uses) ====="
set -x
$NVCC $NC "$HERE/square_verify.cu" "$SQ" -L"$GPUNTT_LIB" -lntt-1.0 -o "$HERE/bin/square_verify_$ARCH" || { echo "BUILD FAIL verify"; exit 1; }
$NVCC $NC "$HERE/dpf_real_bench.cu" "$HERE/common/dpf_gpu.cu" "$HERE/common/aes_gpu.cu" \
      "$HERE/common/leaf_convert_cuda.cu" -o "$HERE/bin/dpf_real_bench_$ARCH" || { echo "BUILD FAIL dpf"; exit 1; }
$NVCC $NC "$HERE/ntt_batch_bench.cu" "$SQ" "$HERE/common/poly_mul_cuda.cu" \
      -L"$GPUNTT_LIB" -lntt-1.0 -o "$HERE/bin/ntt_batch_bench_$ARCH" || { echo "BUILD FAIL batch"; exit 1; }
$NVCC $NC "$HERE/naive_ntt_bench.cu" "$SQ" "$HERE/common/poly_mul_cuda.cu" \
      -L"$GPUNTT_LIB" -lntt-1.0 -o "$HERE/bin/naive_ntt_bench_$ARCH" || { echo "BUILD FAIL naive"; exit 1; }
$NVCC -O3 -o "$HERE/bin/alpha_nvlink" "$HERE/alpha_bench/alpha_nvlink.cu" || echo "(alpha_nvlink build failed, non-fatal)"
set +x
echo "ALL BUILDS OK"

echo; echo "===== (b) square correctness gate ====="
"$HERE/bin/square_verify_$ARCH" 8 9 10 11 12 2>&1 | tee "$P/square_verify.log"
tail -1 "$P/square_verify.log" | grep -q "GATE PASS" && echo "PROBE: GATE PASS" || echo "PROBE: GATE FAIL"

echo; echo "===== (c) does the c=8,t=4 row fault?  (B=64 > t^2=16) ====="
for lg in 12 20; do
  N=$((1<<lg)); n=1; k=0; x=$((2*N/4)); while [ $((1<<k)) -lt "$x" ]; do k=$((k+1)); done; n=$k
  echo "--- lg=$lg  n=$n  B=64 t=4 (the suspect row) ---"
  env PCG_RTT_US=5 compute-sanitizer --tool memcheck --error-exitcode 99 \
      "$HERE/bin/dpf_real_bench_$ARCH" --n "$n" --B 64 --t 4 --iters 1 \
      > "$P/sanitize_c8t4_lg$lg.log" 2>&1
  rc=$?
  echo "compute-sanitizer rc=$rc"
  grep -aE "ERROR SUMMARY|Invalid __global__|out-of-bounds|CUDA" "$P/sanitize_c8t4_lg$lg.log" | head -6
done
echo "--- control: a row where B<=t^2 (c=4,t=16 -> B=16) ---"
env PCG_RTT_US=5 compute-sanitizer --tool memcheck --error-exitcode 99 \
    "$HERE/bin/dpf_real_bench_$ARCH" --n 13 --B 16 --t 16 --iters 1 \
    > "$P/sanitize_c4t16.log" 2>&1
echo "compute-sanitizer rc=$?"
grep -aE "ERROR SUMMARY" "$P/sanitize_c4t16.log" | head -3

echo; echo "===== (d) timing of the biggest cells (to size the real -t) ====="
echo "--- ntt square lg=24 b=64 (OOM candidate) ---"
/usr/bin/time -f "  WALL %e s  MAXRSS %M KB" env PCG_SQUARE_STAGE_MS=1 \
   "$HERE/bin/ntt_batch_bench_$ARCH" --backend square --N 16777216 --batch 64 --iters 3 \
   > "$P/ntt_lg24_b64_square.log" 2>&1
echo "  rc=$?"; tail -2 "$P/ntt_lg24_b64_square.log"
echo "--- ntt naive lg=24 b=64 ---"
/usr/bin/time -f "  WALL %e s  MAXRSS %M KB" \
   "$HERE/bin/naive_ntt_bench_$ARCH" --logN 24 --batch 64 --iters 3 \
   > "$P/ntt_lg24_b64_naive.log" 2>&1
echo "  rc=$?"; tail -2 "$P/ntt_lg24_b64_naive.log"
echo "--- dpf lg=24 headline row c=4 t=16 wan tier (2 ms x levels) ---"
/usr/bin/time -f "  WALL %e s" env PCG_RTT_US=2000 PCG_DPF_LEVEL_MS=1 \
   "$HERE/bin/dpf_real_bench_$ARCH" --n 21 --B 16 --t 16 --iters 3 \
   > "$P/dpf_lg24_wan.log" 2>&1
echo "  rc=$?"; grep -aE "TOTAL DPF|network" "$P/dpf_lg24_wan.log"

echo; echo "===== probe done ====="
date -Is
