#!/usr/bin/env bash
# Build fused4_bench for the local GPU and run the security-size sweep.
# Works on L40S (sm_89) and B200 (sm_100). Output: results/fused4_<gpu>_<stamp>.csv
# (raw runs) and a min-over-repetitions summary on stdout.
#
#   ./run_fused4.sh                          # logN 20..24 x batch 4,16,64, 3 reps
#   LOGN=22 BATCH=16 REPS=1 ./run_fused4.sh  # subset
#
# Needs GPU-NTT (Alisah-Ozcan/GPU-NTT) with the 62-bit Barrett shift fix in
# gpuntt/common/modular_arith.cuh (the same install the square backend uses).
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
LOGN="${LOGN:-20,21,22,23,24}"
BATCH="${BATCH:-4,16,64}"
ITERS="${ITERS:-5}"
REPS="${REPS:-3}"

grep -q "shift >= 64" "$GPUNTT_INC/gpuntt/common/modular_arith.cuh" || {
    echo "GPU-NTT headers at $GPUNTT_INC lack the 62-bit Barrett shift fix" >&2; exit 1; }

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d ' .')
ARCH="sm_${CAP}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1 | tr ' ' '_')
mkdir -p "$HERE/bin" "$HERE/results"
BIN="$HERE/bin/fused4_bench_${ARCH}"
"$NVCC" -O3 -std=c++17 -arch="$ARCH" -I"$ROOT" -I"$GPUNTT_INC" "$HERE/fused4_bench.cu" \
    -L"$GPUNTT_LIB" -lntt-1.0 -o "$BIN"

OUT="$HERE/results/fused4_${GPU}_$(date +%Y%m%d_%H%M%S).csv"
"$BIN" --selftest --logN 16 --batch 4 --iters 2 | tee "${OUT%.csv}.selftest.txt"
for r in $(seq 1 "$REPS"); do
    "$BIN" --logN "$LOGN" --batch "$BATCH" --iters "$ITERS"
done > "$OUT"
echo "raw runs: $OUT"
python3 "$HERE/summarize.py" "$OUT"
