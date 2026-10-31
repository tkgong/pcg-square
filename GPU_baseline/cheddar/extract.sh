#!/bin/bash
# Extract Cheddar's INTTPhase1/INTTPhase2/NTTPhase1/NTTPhase2 kernels verbatim (src/core/NTT.cu) with three mechanical edits:
#  (1) the launch config is a template parameter instead of the hard-coded NTTLaunchConfig<log_degree,...> (N <= 2^16 only),
#  (2) every polynomial uses one twiddle table (PCG has a single prime; Cheddar indexes a table per RNS limb),
#  (3) the per-polynomial offset y_idx << log_degree is computed in 64 bits (int overflows once batch x N > 2^31 elements;
#      identical code below that).
# Usage: extract.sh CHEDDAR_ROOT OUT
set -euo pipefail
SRC="$1/src/core/NTT.cu"; OUT="$2"
s=$(grep -n "^__global__ void INTTPhase1(" "$SRC" | cut -d: -f1); e=$(grep -n "^__global__ void NTTPhase2ForModDown(" "$SRC" | cut -d: -f1)
{ echo '#pragma once'; echo '#include "common/Basic.cuh"'; echo '#include "core/NTTUtils.cuh"'; echo 'namespace cheddar { namespace kernel {'
  sed -n "$((s-1)),$((e-2))p" "$SRC" \
  | sed 's/^template <typename word, int log_degree>$/template <typename word, int log_degree, class Cfg>/' \
  | sed 's/^template <typename word, int log_degree,$/template <typename word, int log_degree, class Cfg,/' \
  | sed 's/using Config = NTTLaunchConfig<log_degree, NTTType::[A-Z]*, Phase::Phase[12]>;/using Config = Cfg;/' \
  | sed 's/const word \*w = twiddle_factors + (tw_y_idx << log_degree);/const word *w = twiddle_factors;/' \
  | sed 's/const word \*w_msb = twiddle_factors_msb + (tw_y_idx \* kMsbSize);/const word *w_msb = twiddle_factors_msb;/' \
  | sed 's/(y_idx << log_degree)/((size_t) y_idx << log_degree)/'
  echo '} }'; } > "$OUT"
echo "kernels: $(grep -c '^__global__' "$OUT"), config edits: $(grep -c 'using Config = Cfg;' "$OUT"), twiddle edits: $(grep -c '= twiddle_factors;\|= twiddle_factors_msb;' "$OUT"), leftover per-limb twiddle indexing: $(grep -c 'tw_y_idx << log_degree\|tw_y_idx \* kMsbSize' "$OUT"), 64-bit offsets: $(grep -c '((size_t) y_idx << log_degree)' "$OUT"), leftover 32-bit offsets: $(grep -c '[^)] y_idx << log_degree\|(y_idx << log_degree)' "$OUT")"
