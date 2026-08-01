#!/bin/bash
# B200 GPU baseline for PCG expansion, full table. HEAD = 45b5f8e.
#   ./run_pcg_baseline.sh   (no args -- full grid, exactly as specified)
#
# One GPU: the runner only ever touches device 0, and the batched e2e arm is
# meant to be bounded by one device's 183 GB (its OOM at the large corners is a
# reported result, not a failure to avoid). Asking for 1 GPU also schedules
# faster on a partition with ~630 queued jobs.
#
# Wall: the runner has no resume -- one abort loses all four batteries. 12 h
# against a stated 4-8 h. (b200-devel cannot host this: 4 h cap, QOS
# gres/gpu=2.)
#
# MACHINE HYGIENE. The package asks whether the B200 is shared. The runner's own
# gpu_before/after come from `nvidia-smi --query-compute-apps`, which inside a
# Slurm cgroup can only ever see THIS job's processes on THIS job's GPUs -- it
# is blind to another user's job on a different GPU of the same physical node,
# which is exactly the pollution mode described in the package. So we also
# record node-level co-tenancy from squeue and the load average at both ends,
# and the report quotes those, not just the nvidia-smi files.
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 32
#SBATCH --mem=400G
#SBATCH -t 12:00:00
#SBATCH -J pcg-baseline-v2
#SBATCH -o /home/gong_ten_neu/PCG-acceleration/pcg-square/GPU_baseline/job_baseline_v2_%j.out

set -u
export PATH="$HOME/opt/cuda-13.0/bin:$PATH"
export NVCC="$HOME/opt/cuda-13.0/bin/nvcc"
export GPUNTT_INC="$HOME/.local/include/GPUNTT-1.0"
export GPUNTT_LIB="$HOME/.local/lib64"      # not ~/.local/lib on this host
HERE=/home/gong_ten_neu/PCG-acceleration/pcg-square/GPU_baseline
cd "$HERE"
T0=$(date +%s)

echo "################ PCG BASELINE v2 (HEAD 45b5f8e) ################"
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST"; hostname; date -Is
echo "commit: $(git -C "$HERE" rev-parse HEAD)"
echo
echo "---- NODE-LEVEL co-tenancy BEFORE (what nvidia-smi cannot see) ----"
squeue -w "$(hostname -s)" -o "%.10i %.14u %.24j %.8T %.4C %b" 2>&1
echo "loadavg: $(cat /proc/loadavg)  nproc=$(nproc)  cpus_alloc=${SLURM_CPUS_ON_NODE:-?}"
echo "GPUs visible to this job:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
echo

chmod +x run_pcg_baseline.sh
./run_pcg_baseline.sh
RC=$?

echo
echo "################ runner exit=$RC ################"
echo "---- NODE-LEVEL co-tenancy AFTER ----"
squeue -w "$(hostname -s)" -o "%.10i %.14u %.24j %.8T %.4C %b" 2>&1
echo "loadavg: $(cat /proc/loadavg)"
nvidia-smi -q | grep -A4 "ECC Errors" | head -12

OUT="$HERE/pcg_baseline_out"
echo
echo "---- deliverable inventory ----"
for f in dpf_shape.csv ntt_stages.csv dpf_grid.csv e2e_full.csv \
         square_verify.log square_verify_fold.log topo.txt gpu_before.txt gpu_after.txt; do
  if [ -s "$OUT/$f" ]; then echo "OK    $f  ($(wc -l < "$OUT/$f") lines)"
  else echo "MISSING/EMPTY  $f"; fi
done
echo "logs: $(ls "$OUT"/*.log 2>/dev/null | wc -l)"
tar -czf "$OUT/pcg_baseline_logs_${SLURM_JOB_ID}.tar.gz" -C "$OUT" \
    $(cd "$OUT" && ls *.log 2>/dev/null) 2>/dev/null
ls -l "$OUT/pcg_baseline_logs_${SLURM_JOB_ID}.tar.gz" 2>&1

echo
echo "---- OOM / crash scan over every cell log (verbatim) ----"
grep -alE "out of memory|OOM|illegal memory|CUDA Error|CudaException|terminate called|std::bad_alloc|Aborted" \
    "$OUT"/*.log 2>/dev/null | while read -r f; do
  echo "=== $(basename "$f") ==="
  grep -aE "out of memory|OOM|illegal memory|CUDA Error|CudaException|terminate called|std::bad_alloc|Aborted" "$f" | head -3
done
echo "(end scan)"

T1=$(date +%s)
echo
echo "TOTAL WALL CLOCK: $((T1-T0)) s = $(( (T1-T0)/3600 ))h $(( ((T1-T0)%3600)/60 ))m $(( (T1-T0)%60 ))s"
exit $RC
