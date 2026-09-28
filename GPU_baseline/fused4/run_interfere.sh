#!/usr/bin/env bash
# Build interfere_bench for the local GPU and sweep aggressor size (blocks).
# Output: results/interfere_<gpu>_<stamp>.csv
#   ./run_interfere.sh                 # logN 22 and 24, batch 16, blocks 8..128
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
LOGN="${LOGN:-22 24}"
BLOCKS="${BLOCKS:-8 16 32 64 128}"
if ! ls "$GPUNTT_LIB"/libntt-1.0.* >/dev/null 2>&1; then
    GPUNTT_LIB=$(dirname "$(find "$HOME" -maxdepth 6 -name 'libntt-1.0.*' 2>/dev/null | head -1)")
fi
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d ' .')
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1 | tr ' ' '_')
mkdir -p "$HERE/bin" "$HERE/results"
BIN="$HERE/bin/interfere_bench_sm_${CAP}"
"$NVCC" -O3 -std=c++17 -arch="sm_${CAP}" -I"$ROOT" -I"$GPUNTT_INC" "$HERE/interfere_bench.cu" \
    -L"$GPUNTT_LIB" -lntt-1.0 -o "$BIN"
OUT="$HERE/results/interfere_${GPU}_$(date +%Y%m%d_%H%M%S).csv"
for lg in $LOGN; do
    for b in $BLOCKS; do
        "$BIN" --logN "$lg" --batch 16 --blocks "$b" --iters 10
    done
done | tee "$OUT"
echo "wrote $OUT"
