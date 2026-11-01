#!/bin/bash
# Silicon interference sweep of Cheddar's NTT and the four-step on Cheddar's kernels (cheddar_interfere.cu) at logN 20..24,
# batch 16, each design at its best config in cheddar_f4dru_<tag>.csv (row M2 = 3, the config the co-simulation uses).
# Each logN gets 15 min (an aggressor of 128 blocks once hung the L40S at 2^20); BLOCKS 8..64 reach ~70% of L40S peak.
# Usage: ARCH=sm_100 OUT=../results_b200 TAG=b200 BLOCKS=8,16,32,64,96,128 ./run_cheddar_interfere.sh  (needs cheddar_f4dru_<tag>.csv)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ARCH="${ARCH:-sm_89}"; OUT="${OUT:-../results_l40s}"; TAG="${TAG:-l40s}"; LOGNS="${LOGNS:-20 21 22 23 24}"; BLOCKS="${BLOCKS:-8,16,32,48,64}"
NVCC="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"; GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
[ -d cheddar-fhe ] || git clone -q https://github.com/scale-snu/cheddar-fhe.git cheddar-fhe
git -C cheddar-fhe checkout -q 8df8b26
mkdir -p gen bin "$OUT"
./extract.sh cheddar-fhe gen/cheddar_ntt_kernels.cuh
./extract_contig.sh gen/cheddar_ntt_kernels.cuh gen/cheddar_ntt_contig.cuh
FLAGS=(-O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -diag-suppress 177 -Ishim -Icheddar-fhe/include -I. -I.. -I"$GPUNTT_INC")
for L in $LOGNS; do
  "$NVCC" "${FLAGS[@]}" -DBIGN_LOGN=$L cheddar_interfere.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/cheddar_interfere_${L}_$ARCH" &
done
wait
best() {   # best valid config of design $2 at logN $1, batch 16: "P1,M1,B1" (cheddar) or "P1,M1" (f4c)
  python3 - "$OUT/cheddar_f4dru_$TAG.csv" "$1" "$2" <<'PY'
import csv, sys
R = [r for r in csv.DictReader(open(sys.argv[1])) if r["logN"] == sys.argv[2] and r["batch"] == "16" and r["design"] == sys.argv[3] and r["mismatches"] == "0"]
r = min(R, key=lambda r: float(r["ms_per_poly"]))
print(f"{r['P1']},{r['M1']},{r['B1']}" if sys.argv[3] == "cheddar" else f"{r['P1']},{r['M1']}")
PY
}
CSV="$OUT/cheddar_interfere_$TAG.csv"; : > "$CSV"
for L in $LOGNS; do
  C=$(best $L cheddar); F=$(best $L f4c_sm)
  timeout 900 "./bin/cheddar_interfere_${L}_$ARCH" --batch 16 --cheddar "$C" --f4c "$F" --blocks "$BLOCKS" --iters 10 > "bin/interfere_$L.csv" || echo "logN $L: bench exited with $?" >&2
  if [ -s "$CSV" ]; then tail -n +2 "bin/interfere_$L.csv"; else cat "bin/interfere_$L.csv"; fi | tee -a "$CSV"
done
python3 interfere_summary.py "$TAG"
