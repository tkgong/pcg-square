#!/bin/bash
# Full-grid comparison of the balanced ("square") four-step against the library's
# skewed 4-step, over the sizes the PCG evaluation actually uses:
#   logN 10..24  x  batch {4,16,64}   (batch = c^2 for c in {2,4,8})
# Records ms_per_mul for both backends plus the per-stage split, so the
# transpose-vs-core trade-off can be read per size instead of extrapolated from
# a couple of points. Failures (OOM, unsupported) are printed, never hidden.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/square_sweep}"
mkdir -p "$OUT"
BIN="$HERE/bin/ntt_bench_sq"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
CSV="$OUT/square_stages.csv"
echo "logN,N,batch,backend,ms_per_mul,transpose_ms,ntt_ms,twiddle_ms,twist_ms,pointwise_ms,status" > "$CSV"

for lg in $(seq 10 24); do
  N=$((1<<lg))
  for b in 4 16 64; do
    for be in square; do
      tag="lg${lg}_b${b}_${be}"
      ENV="PCG_SQUARE_STAGE_MS=1"; PAT="square-stages"
      log="$OUT/$tag.log"
      CUDA_VISIBLE_DEVICES=$GPU env $ENV "$BIN" --backend $be --N $N --batch $b --iters 3 \
        > "$log" 2>&1
      mpm=$(grep -o "ms_per_mul=[0-9.]*" "$log" | tail -1 | cut -d= -f2)
      if [ -z "$mpm" ]; then
        why=$(head -2 "$log" | tr '\n' ' ' | cut -c1-70)
        echo "$lg,$N,$b,$be,,,,,,,FAIL: $why" >> "$CSV"
        echo "  lg=$lg b=$b $be  FAIL ($why)"
        continue
      fi
      line=$(grep "$PAT" "$log" | tail -1)
      tr=$(echo "$line" | grep -o "transpose=[0-9.]*" | cut -d= -f2)
      nt=$(echo "$line" | grep -o " ntt=[0-9.]*" | cut -d= -f2)
      tw=$(echo "$line" | grep -o "twiddle=[0-9.]*" | cut -d= -f2)
      tx=$(echo "$line" | grep -o "twist=[0-9.]*" | cut -d= -f2)
      pw=$(echo "$line" | grep -o "pointwise=[0-9.]*" | cut -d= -f2)
      echo "$lg,$N,$b,$be,$mpm,${tr:-},${nt:-},${tw:-},${tx:-},${pw:-},ok" >> "$CSV"
      echo "  lg=$lg b=$b $be  ms_per_mul=$mpm"
    done
  done
done
echo "SWEEP DONE -> $CSV"
