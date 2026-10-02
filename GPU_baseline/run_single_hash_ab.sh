#!/usr/bin/env bash
# A/B of the GPU baseline's leaf conversion, one job, one GPU, same binary:
#   twopass    the campaign's conversion: H'(leaf) computed in out_sums and again in out_scatter
#   singlehash CONV_V2=1: H' computed once (out_sums_v2 keeps a tau byte per leaf, scatter_v2
#              adds +-tau*CW after the Beaver opens)
# Every cell runs both modes back to back (serial and batched arms, --iters 3) at the paper's
# tier (nvlink, PCG_RTT_US=5) and at rtt 0 (the L40S single-hash run). The two modes must give
# the same G_FINGERPRINT (bit-exact gate); a cell that does not is marked MISMATCH.
#
#   ./run_single_hash_ab.sh                      # 6 security rows x logN 20-24
#   LGS="20 21" ROWS="80:4:16" ./run_single_hash_ab.sh
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/results_b200/single_hash_ab}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
LGS="${LGS:-20 21 22 23 24}"
ROWS="${ROWS:-80:2:64 80:4:16 80:8:4 128:2:128 128:4:16 128:8:8}"
TIERS="${TIERS:-5 0}"
ITERS="${ITERS:-3}"
mkdir -p "$OUT" "$HERE/bin"

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d ' .')
ARCH="sm_${CAP}"; GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)
if [ -z "${NVCC:-}" ]; then NVCC=nvcc
  for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-13*/bin/nvcc /usr/local/cuda-12.[6-9]/bin/nvcc; do
    [ -x "$c" ] || continue
    "$c" --list-gpu-arch 2>/dev/null | grep -qx "compute_${CAP}" && { NVCC="$c"; break; }
  done
fi
echo "[dev ] $GPU  arch=$ARCH  nvcc=$NVCC  commit=$(git -C "$HERE" rev-parse --short HEAD)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$OUT/gpu_before.txt"
echo "[hyg ] other compute processes at start:"; cat "$OUT/gpu_before.txt"

NC="-O3 -std=c++17 -arch=$ARCH -I$HERE -I$GPUNTT_INC"
CORE="$HERE/common/dpf_gpu.cu $HERE/common/aes_gpu.cu $HERE/common/leaf_convert_cuda.cu"
NTTC="$HERE/common/poly_mul_gpuntt_square.cu $HERE/common/poly_mul_cuda.cu"
E2="$HERE/bin/e2e_full_ab_$ARCH"
echo "[bld ] $E2"
$NVCC $NC "$HERE/e2e_full_bench.cu" $CORE $NTTC -L"$GPUNTT_LIB" -lntt-1.0 -o "$E2"

CSV="$OUT/e2e_single_hash_ab.csv"
echo "gpu,lambda,c,t,logN,rtt_us,arm,mode,iters,expand_ms,convert_ms,beaver_ms,ntt_ms,wall_ms,fingerprint,status" > "$CSV"
mism=0
for r in $ROWS; do
  lam=${r%%:*}; rest=${r#*:}; c=${rest%%:*}; t=${rest##*:}
  for lg in $LGS; do
    for rtt in $TIERS; do
      for arm in serial batched; do
        AF=""; [ $arm = batched ] && AF="--batch-blocks"
        declare -A FP=()
        for mode in twopass singlehash; do
          MF=""; [ $mode = singlehash ] && MF="CONV_V2=1"
          L="$OUT/e2e_${lam}_c${c}t${t}_lg${lg}_rtt${rtt}_${arm}_${mode}.log"
          env $MF PCG_RTT_US=$rtt "$E2" --c $c --t $t --logN $lg --iters $ITERS $AF > "$L" 2>&1 || true
          R=$(grep "^RESULT" "$L" | tail -1); fp=$(grep -oE "G_FINGERPRINT=[0-9a-f]+" "$L" | cut -d= -f2)
          FP[$mode]="${fp:-none}"
          if [ -n "$R" ]; then f(){ echo "$R" | grep -oE "$1=[0-9.-]+" | head -1 | cut -d= -f2; }
            st=$(echo "$R" | grep -oE 'STATUS=[A-Z_]+' | cut -d= -f2)
            echo "$GPU,$lam,$c,$t,$lg,$rtt,$arm,$mode,$ITERS,$(f expand),$(f convert),$(f beaver),$(f ntt),$(f wall),${fp:-},$st" >> "$CSV"
          else echo "$GPU,$lam,$c,$t,$lg,$rtt,$arm,$mode,$ITERS,,,,,,,CRASH" >> "$CSV"; fi
        done
        if [ "${FP[twopass]}" != none ] && [ "${FP[singlehash]}" != none ] && [ "${FP[twopass]}" != "${FP[singlehash]}" ]; then
          echo "  MISMATCH c=$c t=$t logN=$lg rtt=$rtt arm=$arm: ${FP[twopass]} vs ${FP[singlehash]}"; mism=$((mism+1))
          sed -i "\$s/\$/_MISMATCH/" "$CSV"
        fi
      done
    done
  done; echo "  [ab ] lam$lam c$c t$t done"
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$OUT/gpu_after.txt"
echo "[gate] fingerprint mismatches: $mism"
echo "[done] $CSV"
