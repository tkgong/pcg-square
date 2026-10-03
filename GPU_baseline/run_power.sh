#!/usr/bin/env bash
# GPU board power per phase on one idle GPU (L40S or B200), for the system-energy estimate (reviewer D):
#   idle (no process), idle (CUDA context held), the DPF phase per (c,t) at logN 22 (dpf_real_bench: expand +
#   conversion, one block), the GPU-NTT merge poly-mul (baseline NTT) and the four-step SM lane f4g_sm (PCG^2's
#   NTT on B200) at logN 20/22/24 x batch 4/16/64.
# Two independent readings per phase: (1) nvidia-smi samples power.draw (1 s average) and, where the driver has
# it, power.draw.instant every 100 ms for the whole session -> mean over [start+1 s, end-0.5 s]; (2) the NVML
# cumulative energy counter (nvmlDeviceGetTotalEnergyConsumption, mJ) read before/after each phase when pynvml
# is installed. Every phase runs >= SECS seconds (default 10), so both agree; the counter is the reference.
#
#   GPU=0 ./run_power.sh                 # -> results_<l40s|b200>/power/{power_trace.csv,phases.csv,*.log}
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-0}"
SECS="${SECS:-10}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
ls "$GPUNTT_LIB"/libntt-1.0.* >/dev/null 2>&1 || GPUNTT_LIB=$(dirname "$(find "$HOME" -maxdepth 6 -name 'libntt-1.0.*' 2>/dev/null | head -1)")
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" | head -1)
case "$NAME" in *B200*) TAG=b200;; *L40S*) TAG=l40s;; *) TAG=$(echo "$NAME" | tr -c 'A-Za-z0-9\n' '_' | tr 'A-Z' 'a-z');; esac
OUT="${OUT:-$HERE/results_${TAG}/power}"
mkdir -p "$OUT" "$HERE/bin" "$HERE/fused4/bin"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" | head -1 | tr -d ' .'); ARCH="sm_${CAP}"
if [ -z "${NVCC:-}" ]; then NVCC=nvcc
  for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-13*/bin/nvcc /usr/local/cuda-12.[6-9]/bin/nvcc "$HOME"/opt/cuda-13*/bin/nvcc; do
    [ -x "$c" ] || continue; "$c" --list-gpu-arch 2>/dev/null | grep -qx "compute_${CAP}" && { NVCC="$c"; break; }
  done
fi
DP="$HERE/bin/dpf_real_bench_$ARCH"; NP="$HERE/fused4/bin/ntt_power_bench_$ARCH"
[ -x "$DP" ] || "$NVCC" -O3 -std=c++17 -arch=$ARCH -I"$HERE" "$HERE/dpf_real_bench.cu" "$HERE/common/dpf_gpu.cu" "$HERE/common/aes_gpu.cu" "$HERE/common/leaf_convert_cuda.cu" -o "$DP"
[ -x "$NP" ] || "$NVCC" -O3 -std=c++17 -arch=$ARCH -I"$HERE" -I"$GPUNTT_INC" "$HERE/fused4/ntt_power_bench.cu" -L"$GPUNTT_LIB" -lntt-1.0 -o "$NP"
export CUDA_VISIBLE_DEVICES=$GPU
echo "[dev ] $NAME  arch=$ARCH  nvcc=$NVCC  commit=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi -i "$GPU" --query-gpu=name,driver_version,power.limit,power.max_limit,clocks.max.sm,clocks.max.mem --format=csv > "$OUT/gpu.txt"
echo "other processes on GPU $GPU at start:" >> "$OUT/gpu.txt"; nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv >> "$OUT/gpu.txt"
ENERGY_OK=$(python3 "$HERE/nvml_energy.py" "$GPU"); echo "NVML energy counter at start (mJ): $ENERGY_OK" >> "$OUT/gpu.txt"

# sampler: power.draw.instant exists on newer drivers only
FIELDS="timestamp,power.draw,power.draw.instant,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu"
nvidia-smi -i "$GPU" --query-gpu=$FIELDS --format=csv,noheader >/dev/null 2>&1 || FIELDS="timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu"
echo "$FIELDS" > "$OUT/power_trace.csv"
nvidia-smi -i "$GPU" --query-gpu=$FIELDS --format=csv,noheader -lms 100 >> "$OUT/power_trace.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
PH="$OUT/phases.csv"; echo "phase,start,end,energy_mJ,note" > "$PH"
phase() {   # phase <name> <note> <cmd...>
  local name=$1 note=$2; shift 2
  local e0=$(python3 "$HERE/nvml_energy.py" "$GPU"); local s=$(date +%s.%N)
  "$@" > "$OUT/$name.log" 2>&1 || echo "FAILED $name" >> "$OUT/$name.log"
  local e=$(date +%s.%N); local e1=$(python3 "$HERE/nvml_energy.py" "$GPU")
  local dE="NA"; [ "$e0" != NA ] && [ "$e1" != NA ] && dE=$((e1 - e0))
  echo "$name,$s,$e,$dE,$note" >> "$PH"; echo "  [$name] $(python3 -c "print(f'{$e-$s:.1f} s')") energy ${dE} mJ"
}
sleep 3
phase idle_nocontext "no process" sleep "$SECS"
phase idle_context "CUDA context held, no work" "$NP" --what sleep --secs "$SECS"
lg2(){ local n=$1 k=0; while [ $((1<<k)) -lt "$n" ]; do k=$((k+1)); done; echo $k; }
for ct in 8:4 4:16 8:8 2:64 2:128; do
  c=${ct%%:*}; t=${ct##*:}; n=$((22 + 1 - $(lg2 $t))); B=$((t*t))
  PCG_RTT_US=0 "$DP" --n $n --B $B --t $t --iters 2 > "$OUT/dpf_c${c}t${t}_probe.log" 2>&1
  ms=$(grep "TOTAL DPF" "$OUT/dpf_c${c}t${t}_probe.log" | grep -oE "[0-9.]+ ms" | head -1 | cut -d' ' -f1)
  it=$(python3 -c "import math; print(max(3, math.ceil($SECS*1000/$ms)))")
  phase "dpf_c${c}t${t}" "logN 22 n=$n B=$B $it iters" env PCG_RTT_US=0 "$DP" --n $n --B $B --t $t --iters $it
done
for what in merge f4g_sm; do for lg in 20 22 24; do for b in 4 16 64; do
  phase "ntt_${what}_lg${lg}_b${b}" "$what poly-mul" "$NP" --what $what --logN $lg --batch $b --secs "$SECS"
done; done; done
sleep 2; kill $SMI; wait $SMI 2>/dev/null || true
echo "[done] $OUT"
