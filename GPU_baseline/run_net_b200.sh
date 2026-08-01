#!/usr/bin/env bash
# B200 re-measurement package, network-inclusive edition.
#
# Produces (all in $OUT):
#   e2e_gpu_v2.csv       per security row: dpf_ms (ONE t^2 block), naive/square
#                        per-mul, transpose PER-MUL, and CORRECT e2e columns:
#                            e2e_* = c^2*dpf_ms + 2*c^2*ntt_per_mul
#                        (fixes run_e2e.sh which summed ONE dpf block with c^2
#                        NTT blocks, and reported transpose per-batch)
#   alpha_measured.csv   measured RTT tiers on THIS machine:
#                            alpha_nvlink : GPU0<->GPU1 4KB p2p ping-pong
#                            alpha_tcp    : loopback (and inter-node if HOST2
#                                           env is set to a peer running the
#                                           alpha_tcp.py server)
#   topo.txt             nvidia-smi topo -m + driver/GPU inventory
#
# Usage: ./run_net_b200.sh [outdir]     (default ./net_out)
#        HOST2=<peer-ip> ./run_net_b200.sh   # adds inter-node DC alpha
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/net_out}"
GPUNTT_INC="${GPUNTT_INC:-$HOME/.local/include/GPUNTT-1.0}"
GPUNTT_LIB="${GPUNTT_LIB:-$HOME/.local/lib}"
mkdir -p "$OUT" "$HERE/bin"

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d ' .')
ARCH="sm_${CAP}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)
if [ -z "${NVCC:-}" ]; then
  NVCC=nvcc
  for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-13*/bin/nvcc \
           /usr/local/cuda-12.[6-9]/bin/nvcc /usr/local/cuda-12/bin/nvcc; do
    [ -x "$c" ] || continue
    "$c" --list-gpu-arch 2>/dev/null | grep -qx "compute_${CAP}" && { NVCC="$c"; break; }
  done
fi
echo "[dev  ] $GPU arch=$ARCH"
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1 || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> "$OUT/topo.txt"

# ---- build ---------------------------------------------------------------
DPF="$HERE/bin/dpf_real_bench_$ARCH"
NAIVE="$HERE/bin/naive_ntt_bench_$ARCH"
BATCH="$HERE/bin/ntt_batch_bench_$ARCH"
NC="-O3 -std=c++17 -arch=$ARCH -I$HERE -I$GPUNTT_INC"
[ -x "$DPF" ]   || "$NVCC" $NC "$HERE/dpf_real_bench.cu" "$HERE/common/dpf_gpu.cu" \
    "$HERE/common/aes_gpu.cu" "$HERE/common/leaf_convert_cuda.cu" -o "$DPF"
[ -x "$NAIVE" ] || "$NVCC" $NC "$HERE/naive_ntt_bench.cu" "$HERE/common/poly_mul_gpuntt_square.cu" \
    "$HERE/common/poly_mul_cuda.cu" -L"$GPUNTT_LIB" -lntt-1.0 -o "$NAIVE"
[ -x "$BATCH" ] || "$NVCC" $NC "$HERE/ntt_batch_bench.cu" "$HERE/common/poly_mul_gpuntt_square.cu" \
    "$HERE/common/poly_mul_cuda.cu" -L"$GPUNTT_LIB" -lntt-1.0 -o "$BATCH"
"$NVCC" -O3 -o "$HERE/bin/alpha_nvlink" "$HERE/alpha_bench/alpha_nvlink.cu"

# ---- alpha tiers ---------------------------------------------------------
ACSV="$OUT/alpha_measured.csv"
echo "bench,peer,msg_bytes,rtt_p50_us,rtt_p10_us,rtt_p90_us" > "$ACSV"
for m in 4096 2048 65536; do "$HERE/bin/alpha_nvlink" $m 1000 >> "$ACSV"; done
( python3 "$HERE/alpha_bench/alpha_tcp.py" server 7911 & ) ; sleep 1
for m in 4096 2048 65536; do
  python3 "$HERE/alpha_bench/alpha_tcp.py" client 127.0.0.1 7911 $m 2000 >> "$ACSV" || true
  ( python3 "$HERE/alpha_bench/alpha_tcp.py" server 7911 & ) ; sleep 1
done
pkill -f "alpha_tcp.py server" 2>/dev/null || true
if [ -n "${HOST2:-}" ]; then
  echo "[alpha] inter-node vs $HOST2 (server must be running there: alpha_tcp.py server 7911)"
  for m in 4096 2048 65536; do
    python3 "$HERE/alpha_bench/alpha_tcp.py" client "$HOST2" 7911 $m 2000 >> "$ACSV" || true
  done
fi

# ---- e2e battery with CORRECT accounting ---------------------------------
log2() { local n=$1 k=0; while [ $((1<<k)) -lt "$n" ]; do k=$((k+1)); done; echo $k; }
grab() { grep -oE "$2=[0-9.]+" "$1" | tail -1 | cut -d= -f2; }
CSV="$OUT/e2e_gpu_v2.csv"
echo "gpu,lambda,N,c,t,batch,dpf_blk_ms,naive_ms_per_mul,square_ms_per_mul,transpose_ms_per_mul,e2e_naive_ms,e2e_square_ms,e2e_square_dru_ms" > "$CSV"
ROWS="80 2 64  80 4 16  80 8 4  128 2 128  128 4 16  128 8 8"
set -- $ROWS
while [ $# -ge 3 ]; do
  lam=$1 c=$2 t=$3; shift 3
  b=$((c*c))
  for lg in ${LGS:-20 21 22 23 24}; do   # LGS="$(seq 10 24)" = full grid
    N=$((1<<lg)); n=$(log2 $((2*N/t)))
    tag="l${lam}_n${lg}_c${c}_t${t}"
    echo "[e2e ] lam=$lam N=2^$lg c=$c t=$t"
    "$DPF"   --n "$n" --B "$b" --t "$t" --iters 3 > "$OUT/$tag.dpf.log"   2>&1 || true
    "$NAIVE" --logN "$lg" --batch "$b" --iters 5 > "$OUT/$tag.naive.log" 2>&1 || true
    PCG_SQUARE_STAGE_MS=1 "$BATCH" --backend square --N "$N" --batch "$b" --iters 5 \
        > "$OUT/$tag.square.log" 2>&1 || true
    dpf=$(grep -oE "TOTAL[^0-9]*[0-9.]+ ms" "$OUT/$tag.dpf.log" | grep -oE "[0-9.]+ ms" | head -1 | cut -d' ' -f1)
    nv=$(grab "$OUT/$tag.naive.log"  ms_per_mul)
    fs=$(grab "$OUT/$tag.square.log"  ms_per_mul)
    trb=$(grep -oE "transpose=[0-9.]+" "$OUT/$tag.square.log" | tail -1 | cut -d= -f2)
    awk -v g="$GPU" -v lam=$lam -v N=$N -v c=$c -v t=$t -v b=$b \
        -v dpf="${dpf:-0}" -v nv="${nv:-0}" -v fs="${fs:-0}" -v trb="${trb:-0}" 'BEGIN{
        tr = trb / b;                                   # per-batch -> PER-MUL
        en = c*c*dpf + 2*c*c*nv;                        # c^2 DPF blocks (fix)
        e4 = c*c*dpf + 2*c*c*fs;
        ed = c*c*dpf + 2*c*c*(fs-tr);
        printf "%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
               g,lam,N,c,t,b,dpf,nv,fs,tr,en,e4,ed }' >> "$CSV"
  done
done
echo; echo "[done ] $OUT"; column -s, -t "$CSV" | head -8
echo "send back: e2e_gpu_v2.csv alpha_measured.csv topo.txt"
