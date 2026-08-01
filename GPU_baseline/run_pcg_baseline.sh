#!/usr/bin/env bash
# Complete GPU-side measurement for the PCG expansion. Four batteries, one
# command. This exact script was run to completion on an idle L40S, so the two
# devices are directly comparable cell for cell.
#
#   1. dpf_shape.csv   ns/leaf(n, B) surface. The cost of one DPF block is set
#                      by exactly two numbers -- tree depth n and the number of
#                      parallel trees B = t^2 -- and NOT by the leaf count
#                      alone: at 2.68e8 leaves, (n=16,B=4096) measures 0.192
#                      ns/leaf while (n=24,B=16) measures 0.267, 39% apart.
#                      Every security row reads off this surface, so a curve
#                      indexed by leaves alone (what we used before) collapses
#                      a real dimension and hands every row the same number.
#   2. dpf_grid.csv    the 30 security rows x 5 interconnect tiers, each at its
#                      OWN (n = logN+1-log2 t, B = t^2). The reference
#                      implementation (pcg_ole_impl.h:114-116) runs c^2 blocks
#                      of t^2 instances SERIALLY, so the bench measures ONE
#                      block and c^2 is a multiplier outside it. Do not batch
#                      blocks together -- the reference does not.
#   3. ntt_stages.csv  logN 10-24 x batch {4,16,64} x {square, naive}, with the
#                      square backend in BOTH bit-reversal conventions.
#                      DEFAULT = standalone brev, so whatever is written to
#                      memory between transforms is in natural order (this is
#                      what the DRU offload assumes). PCG_SQUARE_FOLD_BREV=1
#                      folds it into neighbouring kernels instead: faster up to
#                      logN 22, 1.4x SLOWER at logN 24, because bit-reversed
#                      destination rows destroy DRAM row locality at that
#                      stride. Both are reported; neither is dropped.
#   4. e2e_full.csv    TRUE end-to-end, one process, one wall clock: c^2 blocks
#                      + output layer + per-block Beaver opens + 2c^2 poly_muls.
#                      Every earlier "e2e" number was arithmetic
#                      (c^2*dpf + 2c^2*ntt) and was never an actual run.
#                      TWO ARMS per cell, because the baseline must be bounded
#                      from both sides rather than assumed:
#                        serial  -- what pcg_ole_impl.h:114-116 actually does,
#                                   c^2 blocks one at a time (one block resident)
#                        batched -- all c^2*t^2 instances in one expansion, then
#                                   scatter per block by pointer offset. Legal
#                                   (instances are independent; only the scatter
#                                   is per-block) and c^2 times the memory.
#                      On L40S at c=4,t=16,logN=20 the expansion differs by only
#                      8.7% -- one block already saturates the device -- so the
#                      serial arm is not a strawman. Note the batched arm also
#                      coalesces the per-level CW exchanges (272 -> 17), which is
#                      protocol-level ganging, not just a memory layout choice.
#
# Usage:  ./run_pcg_baseline.sh                 # full grid
#         LGS="20 21 22" ./run_pcg_baseline.sh  # subset
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/pcg_baseline_out}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
LGS="${LGS:-10 11 12 13 14 15 16 17 18 19 20 21 22 23 24}"
ROWS="${ROWS:-80:2:64 80:4:16 80:8:4 128:2:128 128:4:16 128:8:8}"
declare -A TIER=( [nvlink]=5 [pcie]=16 [loopback]=25.2 [dc]=50 [wan]=2000 )
mkdir -p "$OUT" "$HERE/bin"

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d ' .')
ARCH="sm_${CAP}"; GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)
if [ -z "${NVCC:-}" ]; then NVCC=nvcc
  for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-13*/bin/nvcc /usr/local/cuda-12.[6-9]/bin/nvcc; do
    [ -x "$c" ] || continue
    "$c" --list-gpu-arch 2>/dev/null | grep -qx "compute_${CAP}" && { NVCC="$c"; break; }
  done
fi
echo "[dev ] $GPU  arch=$ARCH  nvcc=$NVCC"
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1 || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> "$OUT/topo.txt"
# MACHINE HYGIENE. Our first L40S campaign was polluted by other users' training
# jobs holding 167 GB across four GPUs; several cells came back 3-6x their
# neighbours and had to be thrown away. Record the state at both ends.
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$OUT/gpu_before.txt"
echo "[hyg ] other compute processes at start:"; cat "$OUT/gpu_before.txt"

NC="-O3 -std=c++17 -arch=$ARCH -I$HERE -I$GPUNTT_INC"
CORE="$HERE/common/dpf_gpu.cu $HERE/common/aes_gpu.cu $HERE/common/leaf_convert_cuda.cu"
NTTC="$HERE/common/poly_mul_gpuntt_square.cu $HERE/common/poly_mul_cuda.cu"
echo "[bld ] building 5 binaries..."
$NVCC $NC "$HERE/square_verify.cu"   $NTTC       -L"$GPUNTT_LIB" -lntt-1.0 -o "$HERE/bin/square_verify_$ARCH"
$NVCC $NC "$HERE/ntt_batch_bench.cu" $NTTC       -L"$GPUNTT_LIB" -lntt-1.0 -o "$HERE/bin/ntt_batch_bench_$ARCH"
$NVCC $NC "$HERE/naive_ntt_bench.cu" $NTTC       -L"$GPUNTT_LIB" -lntt-1.0 -o "$HERE/bin/naive_ntt_bench_$ARCH"
$NVCC $NC "$HERE/dpf_real_bench.cu"  $CORE                                  -o "$HERE/bin/dpf_real_bench_$ARCH"
$NVCC $NC "$HERE/e2e_full_bench.cu"  $CORE $NTTC -L"$GPUNTT_LIB" -lntt-1.0 -o "$HERE/bin/e2e_full_$ARCH"
V="$HERE/bin/square_verify_$ARCH"; NB="$HERE/bin/ntt_batch_bench_$ARCH"
NV="$HERE/bin/naive_ntt_bench_$ARCH"; DP="$HERE/bin/dpf_real_bench_$ARCH"
E2="$HERE/bin/e2e_full_$ARCH"

# ---- GATE: no timing is reported for a backend that is not bit-exact -------
echo "[gate] correctness, both bit-reversal conventions"
"$V" > "$OUT/square_verify.log" 2>&1
PCG_SQUARE_FOLD_BREV=1 "$V" > "$OUT/square_verify_fold.log" 2>&1
grep -q "GATE PASS" "$OUT/square_verify.log"      || { echo "GATE FAIL (standalone)"; exit 1; }
grep -q "GATE PASS" "$OUT/square_verify_fold.log" || { echo "GATE FAIL (fold)"; exit 1; }
echo "[gate] PASS both"

lg2(){ local n=$1 k=0; while [ $((1<<k)) -lt "$n" ]; do k=$((k+1)); done; echo $k; }

# ---- 1. DPF shape surface -------------------------------------------------
C1="$OUT/dpf_shape.csv"
echo "gpu,t,B,n,leaves,expand_ms,sums_ms,scatter_ms,total_ms,ns_per_leaf,status" > "$C1"
for t in 4 8 16 32 64 128; do B=$((t*t))
  for n in $(seq 8 24); do
    lv=$((B*(1<<n))); [ $((lv*16)) -gt 34000000000 ] && continue
    L="$OUT/shape_t${t}_n${n}.log"
    PCG_RTT_US=0 "$DP" --n $n --B $B --t $t --iters 3 > "$L" 2>&1 || true
    if grep -q "TOTAL DPF" "$L"; then
      g(){ grep "$1" "$L" | grep -oE "[0-9.]+ ms" | head -1 | cut -d' ' -f1; }
      echo "$GPU,$t,$B,$n,$lv,$(g 'expand '),$(g out_sums),$(g out_scatter),$(g 'TOTAL DPF'),$(grep 'TOTAL DPF' "$L"|grep -oE '[0-9.]+ ns/leaf'|cut -d' ' -f1),ok" >> "$C1"
    else echo "$GPU,$t,$B,$n,$lv,,,,,,$(grep -oE 'what\(\):.*' "$L"|head -1|cut -c1-30)" >> "$C1"; fi
  done; echo "  [1] t=$t done"
done

# ---- 2. NTT: both backends, both brev conventions -------------------------
C2="$OUT/ntt_stages.csv"
echo "gpu,logN,N,batch,backend,brev_mode,ms_per_mul,transpose_ms,brev_ms,ntt_ms,twiddle_ms,twist_ms,pointwise_ms,sum_ms,status" > "$C2"
for lg in $LGS; do N=$((1<<lg))
  for b in 4 16 64; do
    for mode in standalone fold; do
      EF=""; [ $mode = fold ] && EF="PCG_SQUARE_FOLD_BREV=1"
      L="$OUT/ntt_lg${lg}_b${b}_${mode}.log"
      env $EF PCG_SQUARE_STAGE_MS=1 "$NB" --backend square --N $N --batch $b --iters 3 > "$L" 2>&1 || true
      m=$(grep -oE "ms_per_mul=[0-9.]+" "$L"|tail -1|cut -d= -f2)
      if [ -z "$m" ]; then echo "$GPU,$lg,$N,$b,square,$mode,,,,,,,,,FAIL" >> "$C2"; continue; fi
      ln=$(grep "square-stages" "$L"|tail -1); f(){ echo "$ln"|grep -oE "$1=[0-9.]+"|cut -d= -f2; }
      echo "$GPU,$lg,$N,$b,square,$mode,$m,$(f transpose),$(f brev),$(echo "$ln"|grep -oE ' ntt=[0-9.]+'|cut -d= -f2),$(f twiddle),$(f twist),$(f pointwise),$(f sum),ok" >> "$C2"
    done
    L="$OUT/ntt_lg${lg}_b${b}_naive.log"
    "$NV" --logN $lg --batch $b --iters 3 > "$L" 2>&1 || true
    m=$(grep -oE "ms_per_mul=[0-9.]+" "$L"|tail -1|cut -d= -f2)
    st=$(grep -oE "vs_square=[A-Z]+" "$L"|cut -d= -f2)
    echo "$GPU,$lg,$N,$b,naive,-,${m:-},,,,,,,,${st:-FAIL}" >> "$C2"
  done; echo "  [2] logN=$lg done"
done

# ---- 3. DPF benchmark grid + 4. TRUE end-to-end ---------------------------
C3="$OUT/dpf_grid.csv"; C4="$OUT/e2e_full.csv"
echo "gpu,lambda,c,t,logN,n,B,leaves,tier,rtt_us,expand_ms,sums_ms,scatter_ms,total_ms,ns_per_leaf,exchanges,exposed_ms,status" > "$C3"
echo "gpu,lambda,c,t,logN,tier,rtt_us,arm,blocks,B_per_block,n,leaves,expand_ms,convert_ms,beaver_ms,ntt_ms,wall_ms,ns_per_leaf,exchanges,status" > "$C4"
for r in $ROWS; do
  lam=${r%%:*}; rest=${r#*:}; c=${rest%%:*}; t=${rest##*:}
  tl=$(lg2 $t); B=$((t*t))
  for lg in $LGS; do
    [ "$lg" -lt 20 ] && continue           # the security rows are logN 20-24
    n=$((lg+1-tl))
    for tier in nvlink pcie loopback dc wan; do
      L="$OUT/dpf_${lam}_c${c}t${t}_lg${lg}_${tier}.log"
      PCG_RTT_US=${TIER[$tier]} "$DP" --n $n --B $B --t $t --iters 3 > "$L" 2>&1 || true
      if grep -q "TOTAL DPF" "$L"; then
        g(){ grep "$1" "$L"|grep -oE "[0-9.]+ ms"|head -1|cut -d' ' -f1; }
        echo "$GPU,$lam,$c,$t,$lg,$n,$B,$((B*(1<<n))),$tier,${TIER[$tier]},$(g 'expand '),$(g out_sums),$(g out_scatter),$(g 'TOTAL DPF'),$(grep 'TOTAL DPF' "$L"|grep -oE '[0-9.]+ ns/leaf'|cut -d' ' -f1),$(grep network: "$L"|grep -oE 'exchanges=[0-9]+'|cut -d= -f2),$(grep network: "$L"|grep -oE 'exposed=[0-9.]+'|cut -d= -f2),ok" >> "$C3"
      else echo "$GPU,$lam,$c,$t,$lg,$n,$B,,$tier,${TIER[$tier]},,,,,,,,$(grep -oE 'what\(\):.*' "$L"|head -1|cut -c1-28)" >> "$C3"; fi
      for arm in serial batched; do
        AF=""; [ $arm = batched ] && AF="--batch-blocks"
        E="$OUT/e2e_${lam}_c${c}t${t}_lg${lg}_${tier}_${arm}.log"
        PCG_RTT_US=${TIER[$tier]} "$E2" --c $c --t $t --logN $lg --iters 1 $AF > "$E" 2>&1 || true
        R=$(grep "^RESULT" "$E"|tail -1)
        if [ -n "$R" ]; then f(){ echo "$R"|grep -oE "$1=[0-9.-]+"|head -1|cut -d= -f2; }
          echo "$GPU,$lam,$c,$t,$lg,$tier,${TIER[$tier]},$arm,$((c*c)),$B,$(f n),$(f leaves),$(f expand),$(f convert),$(f beaver),$(f ntt),$(f wall),$(f ns_per_leaf),$(f exch),$(echo "$R"|grep -oE 'STATUS=[A-Z_]+'|cut -d= -f2)" >> "$C4"
        else echo "$GPU,$lam,$c,$t,$lg,$tier,${TIER[$tier]},$arm,$((c*c)),$B,,,,,,,,,,CRASH" >> "$C4"; fi
      done
    done
  done; echo "  [3+4] lam$lam c$c t$t done"
done

nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$OUT/gpu_after.txt"
echo; echo "[done] $OUT"
echo "send back: dpf_shape.csv ntt_stages.csv dpf_grid.csv e2e_full.csv \\"
echo "           square_verify*.log topo.txt gpu_before.txt gpu_after.txt + tar of *.log"
