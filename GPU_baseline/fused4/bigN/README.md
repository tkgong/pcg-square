Four-step NTT at logN 24..28 on the L40S (batch 1), against the GPU-NTT merge kernel, bit-exact.
fused4/ holds the a392780 headers (headline PlanG: row transform fixed at 2^10, column N/2^10 on GPU-NTT's kernels)
with the size limits raised to logN 28 / sub-transform 2^18 and 64 KB dynamic shared memory opted in for the
own-kernel path (2^13 tiles, so logN <= 26 on the L40S). Prime 4611685989973229569 = 2^62 - 53*2^29 + 1 (2-adicity 29).
Build and run:
  nvcc -O3 -std=c++17 -arch=sm_89 -I. -I.. -I$HOME/.local/include/GPUNTT-1.0 fused4_bigN_bench.cu -L$HOME/.local/lib -lntt-1.0 -o fused4_bigN_bench
  ./fused4_bigN_bench --logN 24,25,26,27,28 --iters 3 --prime 4611685989973229569
Result: GPU_baseline/results_l40s/fused4_bigN_l40s.csv
