#!/bin/bash
# Column-pass kernels for a four-step layout (data transposed by the DRU, so each column of the R x C view is a contiguous
# vector of R = 2^P1 elements): Cheddar's NTTPhase1 / INTTPhase2 (from gen/cheddar_ntt_kernels.cuh) with only the
# global-memory address mapping changed: row stride C -> 1, column offset (adjacent columns, warp batching) -> column * R,
# column = (blockIdx.x << LogWarpBatching) + batch_idx as in Cheddar's kernel (2^B1 contiguous columns per block; a warp
# reads 32 / 2^B1 consecutive words of each of its columns).
# Butterflies, twiddle indices (independent of the column), shared-memory exchange and the 1/N constant are untouched.
# Use with gridDim.x = C >> LogWarpBatching.   Usage: extract_contig.sh IN OUT
set -euo pipefail
IN="$1"; OUT="$2"
f1=$(awk '/^template <typename word, int log_degree, class Cfg>$/{t=NR} /^__global__ void NTTPhase1\(/{print t; exit}' "$IN")
e1=$(awk -v s="$f1" 'NR>s && /^}/{print NR; exit}' "$IN")
f2=$(awk '/^template <typename word, int log_degree, class Cfg,$/{t=NR} /^__global__ void INTTPhase2\(/{print t; exit}' "$IN")
e2=$(awk -v s="$f2" 'NR>s && /^}/{print NR; exit}' "$IN")
P1=$(sed -n "${f1},${e1}p" "$IN" \
  | sed 's/NTTPhase1(/NTTPhase1Contig(/' \
  | sed 's/(stage_group_idx << (log_degree - kNumStages))/stage_group_idx/' \
  | sed 's/(blockIdx.x << kLogWarpBatching)/(blockIdx.x << (kNumStages + kLogWarpBatching))/' \
  | sed 's/batch_idx +/(batch_idx << kNumStages) +/' \
  | sed 's/(i << (log_degree - kStageMerging))/(i << (kNumStages - kStageMerging))/' \
  | sed 's/(stage_group_idx << ((log_degree - kNumStages) + kStageMerging))/(stage_group_idx << kStageMerging)/' \
  | sed 's/dst_limb\[dst_idx + (i << (log_degree - kNumStages))\]/dst_limb[dst_idx + i]/')
P2=$(sed -n "${f2},${e2}p" "$IN" \
  | sed 's/INTTPhase2(/INTTPhase2Contig(/' \
  | sed 's/(stage_group_idx << (initial_log_stride + kStageMerging))/(stage_group_idx << kStageMerging)/' \
  | sed 's/(blockIdx.x << kLogWarpBatching)/(blockIdx.x << (kNumStages + kLogWarpBatching))/' \
  | sed 's/batch_idx +/(batch_idx << kNumStages) +/' \
  | sed 's/load_ptr + (i << initial_log_stride)/load_ptr + i/' \
  | sed 's/(stage_group_idx << initial_log_stride)/stage_group_idx/' \
  | sed 's/(j << (log_degree - kStageMerging))/(j << (kNumStages - kStageMerging))/')
{ echo '#pragma once'; echo '#include "gen/cheddar_ntt_kernels.cuh"'; echo 'namespace cheddar { namespace kernel {'; echo "$P1"; echo; echo "$P2"; echo '} }'; } > "$OUT"
chk() { n=$(grep -c -- "$1" "$OUT" || true); [ "$n" = "$2" ] || { echo "edit check failed: '$1' found $n times, expected $2" >&2; exit 1; }; }
chk 'log_degree - kNumStages' 1;   # only the now-unused initial_log_stride definition chk 'initial_log_stride + kStageMerging' 0; chk 'kLogWarpBatching)' 0
chk '(blockIdx.x << (kNumStages + kLogWarpBatching))' 4; chk '(batch_idx << kNumStages) +' 4; chk 'dst_limb\[dst_idx + i\]' 1; chk 'load_ptr + i)' 1; chk '(j << (kNumStages - kStageMerging))' 1
echo "contiguous column kernels written to $OUT (all address edits applied)"
