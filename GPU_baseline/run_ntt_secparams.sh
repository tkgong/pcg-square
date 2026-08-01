#!/usr/bin/env bash
# GPU NTT battery at the SECURITY-CALIBRATED benchmark sizes (BCG+20 Table 1,
# fully-splitting ring; bench/derive_params.py). Mapping: NTT size = N (ring
# dimension); batch = c^2 (the c^2 g_ij poly-muls per expansion) for
# c in {2,4,8} -> batch in {4,16,64}. Grid: backend {square} x
# N {2^20..2^24, incl. odd exponents} x batch {4,16,64}, two passes per point
# (clean timing + stage-split env). L40S silicon; extrapolation to other
# devices happens in pim/tools/ntt_secparams.py.
#
# Usage: BENCH=/path/to/ntt_batch_bench OUT=/path/outdir ./run_ntt_secparams.sh
set -u
BENCH="${BENCH:?set BENCH=/path/to/ntt_batch_bench}"
OUT="${OUT:?set OUT=/path/outdir}"
ITERS="${ITERS:-5}"
mkdir -p "$OUT"

for be in square; do
  for lg in 20 21 22 23 24; do
    for b in 4 16 64; do
      tag="${be}_lg${lg}_b${b}"
      if [ -s "$OUT/$tag.txt" ] && grep -q ms_per_mul "$OUT/$tag.txt"; then
        echo "skip $tag"; continue
      fi
      echo "run $tag"
      # pass 1: clean timing
      "$BENCH" --backend "$be" --N $((1 << lg)) --batch "$b" --iters "$ITERS" \
          > "$OUT/$tag.txt" 2> "$OUT/$tag.err" || {
        echo "FAILED $tag (see $tag.err)"; mv "$OUT/$tag.txt" "$OUT/$tag.failed"
        continue
      }
      # pass 2: stage split (env per backend; stderr carries the stage lines)
      env_var=PCG_SQUARE_STAGE_MS
      env "$env_var=1" "$BENCH" --backend "$be" --N $((1 << lg)) --batch "$b" \
          --iters "$ITERS" > /dev/null 2> "$OUT/${tag}_stages.txt" || true
    done
  done
done
echo NTT-SECPARAMS-DONE
