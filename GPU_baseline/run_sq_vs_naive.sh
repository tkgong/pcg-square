#!/bin/bash
# FINAL NTT baseline sweep: balanced ("square") four-step vs the naive
# per-stage NTT, over the sizes the PCG evaluation uses.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$HERE/sq_vs_naive"; mkdir -p "$OUT"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
CSV="$OUT/final_ntt_baseline.csv"
echo "logN,N,batch,naive_ms_per_mul,square_ms_per_mul,speedup_sq_over_naive,vs_square_check" > "$CSV"
for lg in $(seq 12 24); do
  N=$((1<<lg))
  for b in 4 16 64; do
    nv=$(CUDA_VISIBLE_DEVICES=$GPU "$HERE/bin/naive_ntt_bench_sm_89" --logN $lg --batch $b --iters 3 2>/dev/null)
    nvm=$(echo "$nv" | grep -o "ms_per_mul=[0-9.]*" | cut -d= -f2)
    chk=$(echo "$nv" | grep -o "vs_square=[A-Za-z]*" | cut -d= -f2)
    sq=$(CUDA_VISIBLE_DEVICES=$GPU "$HERE/bin/ntt_bench_sq" --backend square --N $N --batch $b --iters 3 2>/dev/null \
         | grep -o "ms_per_mul=[0-9.]*" | cut -d= -f2)
    sp=""
    [ -n "${nvm:-}" ] && [ -n "${sq:-}" ] && sp=$(python3 -c "print(f'{$nvm/$sq:.2f}')")
    echo "$lg,$N,$b,${nvm:-},${sq:-},${sp:-},${chk:-}" >> "$CSV"
    echo "  lg=$lg b=$b naive=${nvm:-NA} square=${sq:-NA} speedup=${sp:-NA} ${chk:-}"
  done
done
echo "DONE -> $CSV"
