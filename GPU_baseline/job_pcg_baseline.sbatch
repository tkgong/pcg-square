#!/bin/bash
# FULL PCG GPU baseline (FINAL caliber) on B200.
#   LGS="$(seq 10 24)" ./run_pcg_baseline.sh     <- exactly as specified
# Tier list and size grid are NOT touched.
#
# Wall time: run_pcg_baseline.sh has NO resume -- every one of the 90 NTT cells
# and 450 DPF cells re-runs from scratch on restart. A wall-clock kill at hour 6
# loses everything, so we ask for 20 h against a stated 4-8 h estimate rather
# than trying to cut it fine. b200-batch allows 1 day; b200-devel (4 h cap,
# QOS gres/gpu=2) cannot host this run at all.
#
# Two nodes: node[0] runs the baseline, node[1] serves the measured datacenter
# alpha tier. The peer server is run in a RESTART LOOP on purpose --
# alpha_bench/alpha_tcp.py server accepts exactly one connection and exits,
# while the runner fires three client invocations at HOST2 (msg 4096/2048/65536).
# A single server would answer only the first and the other two rows would be
# silently dropped by the runner's `|| true`. (The local loopback section of the
# runner already restarts its server between sizes -- same intent.)
#
# Nodes are deliberately NOT pinned with -w: hand-picking a quiet node put the
# job behind ~628 queued jobs with a >2 h estimated start ("Priority"), while an
# unpinned request schedules in seconds. Co-tenancy is recorded before and after
# instead, so any contention is visible in the record rather than assumed away.
#SBATCH -p b200-batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:b200:2
#SBATCH -c 32
#SBATCH --mem=400G
#SBATCH -t 20:00:00
#SBATCH -J pcg-baseline-final
#SBATCH -o /home/gong_ten_neu/PCG-acceleration/pcg-square/GPU_baseline/job_pcg_baseline_%j.out

set -u
export PATH="$HOME/opt/cuda-13.0/bin:$PATH"
export NVCC="$HOME/opt/cuda-13.0/bin/nvcc"
export GPUNTT_INC="$HOME/.local/include/GPUNTT-1.0"
export GPUNTT_LIB="$HOME/.local/lib64"
HERE=/home/gong_ten_neu/PCG-acceleration/pcg-square/GPU_baseline
OUT="$HERE/pcg_baseline_out"
cd "$HERE"

WALL_T0=$(date +%s)
echo "################ PCG BASELINE FINAL ################"
echo "job=$SLURM_JOB_ID nodelist=$SLURM_JOB_NODELIST"
hostname; date -Is
echo "commit: $(git -C "$HERE" rev-parse HEAD)"
echo "dirty:"; git -C "$HERE" status --porcelain | head -20
echo
echo "---- co-tenancy BEFORE (contention record) ----"
squeue -w "$(hostname -s)" -o "%.10i %.14u %.22j %.8T %.4C %b" 2>&1
cat /proc/loadavg; echo "nproc=$(nproc) SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-?}"
echo
echo "---- GPU inventory / interconnect ----"
nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total,utilization.gpu,memory.used --format=csv
nvidia-smi topo -m 2>&1 | head -12
nvidia-smi -q -d CLOCK | grep -E "SM +:" | head -1
echo

# ---- peer node: restart-looping alpha server -------------------------------
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "nodes: ${NODES[*]}"
SRV_PID=""
if [ "${#NODES[@]}" -ge 2 ]; then
  PEER="${NODES[1]}"
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$PEER" \
    bash -c 'for i in $(seq 1 40); do python3 '"$HERE"'/alpha_bench/alpha_tcp.py server 7911 || true; done' \
    > "$OUT.peer.log" 2>&1 &
  SRV_PID=$!
  PEER_IP=$(getent hosts "$PEER" | awk '{print $1}' | head -1)
  [ -z "$PEER_IP" ] && PEER_IP="$PEER"
  for _ in $(seq 1 30); do
    if python3 - "$PEER_IP" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(1.0)
sys.exit(0 if s.connect_ex((sys.argv[1], 7911)) == 0 else 1)
PY
    then echo "peer alpha server up at $PEER_IP:7911"; export HOST2="$PEER_IP"; break; fi
    sleep 2
  done
  if [ -z "${HOST2:-}" ]; then
    echo "[warn] peer server never answered; measured DC alpha will be absent"
  fi
else
  echo "[warn] single node allocated; measured DC alpha will be absent"
fi
echo

# background GPU memory sampler, for the OOM report
( while true; do date +%s | tr '\n' ' '; \
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ';'; \
    echo; sleep 10; done ) > "$OUT.mem.txt" 2>/dev/null &
SAMPLER=$!

echo "################ run_pcg_baseline.sh ################"
chmod +x ./run_pcg_baseline.sh
LGS="$(seq 10 24)" ./run_pcg_baseline.sh
RC=$?
echo "################ runner exit=$RC ################"

kill $SAMPLER 2>/dev/null
[ -n "$SRV_PID" ] && kill $SRV_PID 2>/dev/null

echo
echo "---- co-tenancy AFTER ----"
squeue -w "$(hostname -s)" -o "%.10i %.14u %.22j %.8T %.4C %b" 2>&1
cat /proc/loadavg
nvidia-smi -q -d CLOCK | grep -E "SM +:" | head -1
nvidia-smi -q | grep -A4 "ECC Errors" | head -12
echo "peak GPU MiB seen: $(tr ';' '\n' < "$OUT.mem.txt" | awk -F', ' 'NF==2{if($2+0>m)m=$2+0}END{print m+0}')"

echo
echo "---- deliverable inventory ----"
for f in ntt_stages.csv dpf_net.csv e2e.csv alpha_measured.csv topo.txt square_verify.log; do
  if [ -s "$OUT/$f" ]; then echo "OK   $f  ($(wc -l < "$OUT/$f") lines)"; else echo "MISSING/EMPTY  $f"; fi
done
echo "dpf_levels/: $(ls "$OUT/dpf_levels" 2>/dev/null | wc -l) files"
echo "logs: $(ls "$OUT"/*.log 2>/dev/null | wc -l)"
tar -czf "$OUT/pcg_baseline_logs_${SLURM_JOB_ID}.tar.gz" -C "$OUT" \
    $(cd "$OUT" && ls *.log 2>/dev/null) 2>/dev/null
ls -l "$OUT/pcg_baseline_logs_${SLURM_JOB_ID}.tar.gz" 2>&1

echo
echo "---- crash / OOM scan over all cell logs (verbatim) ----"
grep -alE "out of memory|OOM|CUDA error|illegal memory|an illegal|cudaError|terminate called|Aborted|std::bad_alloc" \
    "$OUT"/*.log 2>/dev/null | while read -r f; do
  echo "=== $f ==="
  grep -aE "out of memory|OOM|CUDA error|illegal memory|an illegal|cudaError|terminate called|Aborted|std::bad_alloc" "$f" | head -4
done
echo "(end of crash scan)"

WALL_T1=$(date +%s)
echo
echo "TOTAL WALL CLOCK: $((WALL_T1-WALL_T0)) s = $(( (WALL_T1-WALL_T0)/3600 ))h $(( ((WALL_T1-WALL_T0)%3600)/60 ))m"
echo "runner exit code: $RC"
exit $RC
