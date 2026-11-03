#!/bin/bash
# Contention-aware configuration sweep of Cheddar's NTT on silicon (cheddar_interfere.cu): every Cheddar config of
# cheddar_bigN_<tag>.csv within THRESH of the fastest standalone config at each logN (batch 16) is timed alone and beside the
# SPU-like row-burst aggressor (plus the sleep-only control at the same blocks), so the best config UNDER the SPUs' DRAM share
# can be chosen per logN instead of the best standalone one. Same bench, aggressor and protocol as run_cheddar_interfere.sh;
# streaming aggressor skipped (--modes 1,3). Runs are spread over the GPUs listed in GPUS (one aggressor+victim pair per GPU,
# separate memory). Usage: GPUS="0 1 2 3" LOGNS="20 21 22 23 24" THRESH=1.2 ./run_cheddar_interfere_cfg.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ARCH="${ARCH:-sm_89}"; OUT="${OUT:-../results_l40s}"; TAG="${TAG:-l40s}"; LOGNS="${LOGNS:-20 21 22 23 24}"; BLOCKS="${BLOCKS:-8,16,32,48,64}"
THRESH="${THRESH:-1.2}"; GPUS="${GPUS:-0}"; EXT="${EXT:-0}"; SUF="${SUF:-}"   # EXT=1: only M1/M2 = 6 or B1 = 4 configs; SUF: output csv suffix
NVCC="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"; GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
[ -d cheddar-fhe ] || git clone -q https://github.com/scale-snu/cheddar-fhe.git cheddar-fhe
git -C cheddar-fhe checkout -q 8df8b26
mkdir -p gen bin "$OUT"
./extract.sh cheddar-fhe gen/cheddar_ntt_kernels.cuh
./extract_contig.sh gen/cheddar_ntt_kernels.cuh gen/cheddar_ntt_contig.cuh
# plan: one line per run "logN batch cap cheddar_cfg"; batch 16 (every logN here fits), configs within THRESH of the best alone
PLAN=$(python3 - "$OUT/cheddar_bigN_$TAG.csv" "$THRESH" "$EXT" $LOGNS <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["mismatches"] == "0" and r["design"] == "cheddar" and r["batch"] == "16"]
th = float(sys.argv[2]); ext = sys.argv[3] == "1"
for L in map(int, sys.argv[4:]):
    R = [r for r in rows if int(r["logN"]) == L]
    if not R: print(f"# logN {L}: no cheddar row", file=sys.stderr); continue
    best = min(float(r["ms_per_poly"]) for r in R)
    seen = set()
    for r in sorted(R, key=lambda r: float(r["ms_per_poly"])):
        if float(r["ms_per_poly"]) > best * th: break
        k = (r["P1"], r["M1"], r["B1"], r["M2"], r["maxrreg"])
        if k in seen or (ext and not (r["M1"] == "6" or r["M2"] == "6" or r["B1"] == "4")): continue
        seen.add(k); print(L, 16, r["maxrreg"], f"{r['P1']},{r['M1']},{r['B1']},{r['M2']}")
PY
)
echo "$PLAN" | tee "$OUT/cheddar_interfere_cfg_$TAG$SUF.plan"
echo "$(wc -l <<< "$PLAN") runs"
FLAGS=(-O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -diag-suppress 177 -Ishim -Icheddar-fhe/include -I. -I.. -I"$GPUNTT_INC")
while read -r L R; do   # one binary per (logN, register cap), built in parallel
  CAP=(); [ "$R" -gt 0 ] && CAP=(-maxrregcount="$R")
  "$NVCC" "${FLAGS[@]}" "${CAP[@]}" -DBIGN_LOGN=$L -DBIGN_MAXRREG=$R cheddar_interfere.cu -L"$GPUNTT_LIB" -lntt-1.0 -o "bin/cheddar_interfere_${L}_r${R}_$ARCH" &
done <<< "$(awk '{print $1, $3}' <<< "$PLAN" | sort -u)"
wait
# workers: run i goes to GPU GPUS[i mod nGPU]; each worker appends to its own part file
read -ra G <<< "$GPUS"; NG=${#G[@]}
worker() {
  local gi=$1 gpu=$2 I=0 L B R C O
  : > "bin/interfere_cfg_${TAG}_g${gi}.csv"
  while read -r L B R C; do
    I=$((I + 1)); [ $(( (I - 1) % NG )) -eq "$gi" ] || continue
    O="bin/interfere_cfg_${TAG}_${I}.csv"
    CUDA_VISIBLE_DEVICES=$gpu timeout 1800 "./bin/cheddar_interfere_${L}_r${R}_$ARCH" --batch "$B" --victims cheddar --cheddar "$C" \
      --blocks "$BLOCKS" --modes 1,3 --iters 10 > "$O" || echo "run $I (logN $L cap $R cheddar $C): bench exited with $?" >&2
    tail -n +2 "$O" >> "bin/interfere_cfg_${TAG}_g${gi}.csv"
    echo "gpu $gpu run $I/$(wc -l <<< "$PLAN") done: logN $L cap $R $C" >&2
  done <<< "$PLAN"
}
for gi in "${!G[@]}"; do worker "$gi" "${G[$gi]}" & done
wait
CSV="$OUT/cheddar_interfere_cfg_$TAG$SUF.csv"
echo "gpu,victim,logN,batch,P1,M1,B1,M2,maxrreg,blocks,sleep_ns,mode,agg_GBps,victim_ms,slowdown_raw,mismatches" > "$CSV"
for gi in "${!G[@]}"; do cat "bin/interfere_cfg_${TAG}_g${gi}.csv"; done >> "$CSV"
python3 interfere_cfg_summary.py "$TAG"
