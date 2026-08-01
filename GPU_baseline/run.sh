#!/usr/bin/env bash
# Reproduce the GPU baseline + roofline analysis on any CUDA GPU (B200/H200/...).
#
# Produces:
#   1. plateau.txt          DPF ns/leaf vs total leaves (must be size-flat
#                           before the roofline points mean anything)
#   2. <tag>.kern.txt       nsys per-kernel times for the 4 DPF stages
#   3. roofline_points.csv  (stage, intensity, Gblocks/s), kernel-only caliber
#                           -> feeds ../PIM/tools/plot_dpf_roofline_combined.py
#   4. ntt_naive.txt        naive NTT ms/mul at the security sizes
#   5. ntt_square.txt        4-step NTT ms/mul + stage split (transpose share)
#
# The NTT comparison world in this tree is exactly {naive, square}; the merge
# backend is deliberately absent.
#
# Usage:  ./run.sh [outdir]              (default ./roofline_out)
# Needs:  nvcc (CUDA >= 12), nsys, an idle GPU, GPU-NTT installed.
#         GPUNTT_INC / GPUNTT_LIB override the default install paths.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/roofline_out}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
mkdir -p "$OUT" "$HERE/bin"

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d ' .')
ARCH="sm_${CAP}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)

# Pick an nvcc new enough for this GPU (a distro /usr/bin/nvcc is often too
# old: CUDA 11.x rejects sm_89, 12.0-12.3 reject sm_100). NVCC=... overrides.
if [ -z "${NVCC:-}" ]; then
  NVCC=nvcc
  for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-13*/bin/nvcc \
           /usr/local/cuda-12.[6-9]/bin/nvcc /usr/local/cuda-12/bin/nvcc; do
    [ -x "$c" ] || continue
    if "$c" --list-gpu-arch 2>/dev/null | grep -qx "compute_${CAP}"; then
      NVCC="$c"; break
    fi
  done
fi
if ! "$NVCC" --list-gpu-arch 2>/dev/null | grep -qx "compute_${CAP}"; then
  echo "[fatal] $NVCC does not support $ARCH; set NVCC=/path/to/newer/nvcc" >&2
  exit 1
fi
echo "[dev  ] $GPU  arch=$ARCH  nvcc=$($NVCC --version | grep -o 'release [0-9.]*')"

# ---- build ---------------------------------------------------------------
DPF="$HERE/bin/dpf_real_bench_$ARCH"
NAIVE="$HERE/bin/naive_ntt_bench_$ARCH"
BATCH="$HERE/bin/ntt_batch_bench_$ARCH"
NVCC_COMMON="-O3 -std=c++17 -arch=$ARCH -I$HERE -I$GPUNTT_INC"

[ -x "$DPF" ] || { echo "[build] dpf_real_bench"
  "$NVCC" $NVCC_COMMON "$HERE/dpf_real_bench.cu" "$HERE/common/dpf_gpu.cu" \
      "$HERE/common/aes_gpu.cu" "$HERE/common/leaf_convert_cuda.cu" -o "$DPF"; }
[ -x "$NAIVE" ] || { echo "[build] naive_ntt_bench"
  "$NVCC" $NVCC_COMMON "$HERE/naive_ntt_bench.cu" "$HERE/common/poly_mul_gpuntt_square.cu" \
      "$HERE/common/poly_mul_cuda.cu" -L"$GPUNTT_LIB" -lntt-1.0 -o "$NAIVE"; }
[ -x "$BATCH" ] || { echo "[build] ntt_batch_bench"
  "$NVCC" $NVCC_COMMON "$HERE/ntt_batch_bench.cu" "$HERE/common/poly_mul_gpuntt_square.cu" \
      "$HERE/common/poly_mul_cuda.cu" -L"$GPUNTT_LIB" -lntt-1.0 -o "$BATCH"; }

# ---- 1. DPF plateau check (phase-wall) -----------------------------------
: > "$OUT/plateau.txt"
for cfg in "18 1024" "20 256" "22 64" "24 16"; do
  set -- $cfg; echo "[wall ] DPF n=$1 B=$2"
  "$DPF" --n "$1" --B "$2" --iters 3 | tee -a "$OUT/plateau.txt"
done

# ---- 2. DPF per-kernel profile (kernel-only = the roofline caliber) ------
# Sweep the DPF domain 2^18..2^24 at CONSTANT total leaves (268.4M) so the
# roofline gets one point per size per kernel family and only the tree depth
# (number of levels = launches + barriers) varies.
for cfg in "18 1024" "19 512" "20 256" "21 128" "22 64" "23 32" "24 16"; do
  set -- $cfg; tag="n$1_B$2"; echo "[nsys ] DPF $tag"
  nsys profile -o "$OUT/$tag" --force-overwrite true \
      "$DPF" --n "$1" --B "$2" --iters 2 > "$OUT/$tag.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum "$OUT/$tag.nsys-rep" \
      > "$OUT/$tag.kern.txt" 2>/dev/null
done

# ---- 3. roofline points ---------------------------------------------------
# ChaCha blocks and algorithmic-minimum DRAM bytes per unit:
#   tree expand : 1 block / node , 112 B   (hash 48 + CW-apply read/write 64)
#   leaf convert: 2 blocks / leaf, 40 B    (H' in sums and again in scatter)
#   full DPF    : 3 blocks / leaf, 152 B
CSV="$OUT/roofline_points.csv"
echo "gpu,tag,stage,intensity_blocks_per_byte,gblocks_per_s,ms_per_run" > "$CSV"
for cfg in "18 1024" "19 512" "20 256" "21 128" "22 64" "23 32" "24 16"; do
  set -- $cfg; tag="n$1_B$2"; leaves=$(( (1 << $1) * $2 ))
  awk -v gpu="$GPU" -v tag="$tag" -v leaves="$leaves" '
    /hash_kernel/{h=$2} /expand_kernel/{e=$2}
    /out_sum_kernel/{s=$2} /out_scatter_kernel/{c=$2}
    END{ gsub(",","",h); gsub(",","",e); gsub(",","",s); gsub(",","",c); runs=3
      ex=(h+e)/runs/1e6; cv=(s+c)/runs/1e6; fl=ex+cv
      printf "%s,%s,tree_expand,%.6f,%.3f,%.3f\n", gpu,tag,1/112,  leaves/ex/1e6, ex
      printf "%s,%s,leaf_convert,%.6f,%.3f,%.3f\n",gpu,tag,2/40, 2*leaves/cv/1e6, cv
      printf "%s,%s,full_dpf,%.6f,%.3f,%.3f\n",    gpu,tag,3/152,3*leaves/fl/1e6, fl }' \
    "$OUT/$tag.kern.txt" >> "$CSV"
done

# ---- 4/5. NTT world: naive and 4-step at the security sizes --------------
: > "$OUT/ntt_naive.txt"; : > "$OUT/ntt_square.txt"
for lg in 20 21 22 23 24; do
  N=$((1 << lg))
  echo "[ntt  ] naive 2^$lg b16"
  "$NAIVE" --logN "$lg" --batch 16 --iters 5 | tee -a "$OUT/ntt_naive.txt"
  echo "[ntt  ] square 2^$lg b16 (stage split)"
  PCG_SQUARE_STAGE_MS=1 "$BATCH" --backend square --N "$N" --batch 16 --iters 5 \
      2>&1 | tee -a "$OUT/ntt_square.txt"
done

echo
echo "[done ] $OUT"
column -s, -t "$CSV"
echo
echo "gate : every naive line must read vs_square=BITEXACT"
echo "next : feed roofline_points.csv to ../PIM/tools/plot_dpf_roofline_combined.py"
echo "       together with this device's peak DRAM GB/s and INT32 Tops."
