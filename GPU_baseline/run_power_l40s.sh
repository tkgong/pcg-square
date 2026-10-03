#!/usr/bin/env bash
# GPU board power per phase on one idle GPU, for the system-energy estimate (reviewer D, Q14):
#   idle (no context), idle (CUDA context held), the DPF phase per (c,t) at logN 22
#   (dpf_real_bench: expand + conversion, one block), and the GPU-NTT merge poly-mul at
#   logN 20/22/24 x batch 4/16/64 (+ the four-step SM lane at 22/16).
# nvidia-smi samples power.draw every 100 ms for the whole session; each phase records its
# start/end so the average over [start+1 s, end-0.5 s] can be taken afterwards.
#
#   GPU=2 ./run_power_l40s.sh            # -> results_l40s/power/{power_trace.csv,phases.csv,*.log}
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-0}"
OUT="${OUT:-$HERE/results_l40s/power}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
SECS="${SECS:-10}"
mkdir -p "$OUT" "$HERE/bin" "$HERE/fused4/bin"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" | head -1 | tr -d ' .'); ARCH="sm_${CAP}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
DP="$HERE/bin/dpf_real_bench_$ARCH"; NP="$HERE/fused4/bin/ntt_power_bench_$ARCH"
[ -x "$DP" ] || $NVCC -O3 -std=c++17 -arch=$ARCH -I"$HERE" "$HERE/dpf_real_bench.cu" "$HERE/common/dpf_gpu.cu" "$HERE/common/aes_gpu.cu" "$HERE/common/leaf_convert_cuda.cu" -o "$DP"
[ -x "$NP" ] || $NVCC -O3 -std=c++17 -arch=$ARCH -I"$HERE" -I"$GPUNTT_INC" "$HERE/fused4/ntt_power_bench.cu" -L"$GPUNTT_LIB" -lntt-1.0 -o "$NP"
export CUDA_VISIBLE_DEVICES=$GPU
nvidia-smi -i "$GPU" --query-gpu=name,driver_version,power.limit,clocks.max.sm,clocks.max.mem --format=csv > "$OUT/gpu.txt"
echo "other processes on GPU $GPU at start:" >> "$OUT/gpu.txt"; nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv >> "$OUT/gpu.txt"

nvidia-smi -i "$GPU" --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu --format=csv,noheader -lms 100 > "$OUT/power_trace.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null' EXIT
PH="$OUT/phases.csv"; echo "phase,start,end,note" > "$PH"
phase() {   # phase <name> <note> <cmd...>
  local name=$1 note=$2; shift 2
  local s=$(date +%s.%N); "$@" > "$OUT/$name.log" 2>&1 || echo "FAILED $name" >> "$OUT/$name.log"; local e=$(date +%s.%N)
  echo "$name,$s,$e,$note" >> "$PH"; echo "  [$name] $(python3 -c "print(f'{$e-$s:.1f} s')")"
}
sleep 3
phase idle_nocontext "no process" sleep "$SECS"
phase idle_context "CUDA context held, no work" "$NP" --what sleep --secs "$SECS"
lg2(){ local n=$1 k=0; while [ $((1<<k)) -lt "$n" ]; do k=$((k+1)); done; echo $k; }
for ct in 8:4 4:16 8:8 2:64 2:128; do
  c=${ct%%:*}; t=${ct##*:}; n=$((22 + 1 - $(lg2 $t))); B=$((t*t))
  # size the run: 2 iterations to get the per-iteration time, then enough for SECS
  PCG_RTT_US=0 "$DP" --n $n --B $B --t $t --iters 2 > "$OUT/dpf_c${c}t${t}_probe.log" 2>&1
  ms=$(grep "TOTAL DPF" "$OUT/dpf_c${c}t${t}_probe.log" | grep -oE "[0-9.]+ ms" | head -1 | cut -d' ' -f1)
  it=$(python3 -c "import math; print(max(3, math.ceil($SECS*1000/$ms)))")
  phase "dpf_c${c}t${t}" "logN 22, n=$n B=$B, $it iters" env PCG_RTT_US=0 "$DP" --n $n --B $B --t $t --iters $it
done
for lg in 20 22 24; do for b in 4 16 64; do
  phase "ntt_merge_lg${lg}_b${b}" "GPU-NTT merge poly-mul" "$NP" --what merge --logN $lg --batch $b --secs "$SECS"
done; done
phase "ntt_f4gsm_lg22_b16" "four-step SM lane (DRU design)" "$NP" --what f4g_sm --logN 22 --batch 16 --secs "$SECS"
sleep 2; kill $SMI; wait $SMI 2>/dev/null || true
echo "[done] $OUT"
