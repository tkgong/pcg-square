#!/bin/bash
# B200: A/B of the GPU baseline's leaf conversion, two-pass H' (the campaign) vs single-hash
# (CONV_V2=1). One GPU, one job; both modes per cell back to back. ~10-15 min on a B200
# (the whole serial e2e battery at logN 20-24 is ~25 s per iteration).
#   sbatch GPU_baseline/job_single_hash_b200.sbatch
# Deliverables: results_b200/single_hash_ab/e2e_single_hash_ab.csv + this job's .out
# (the "[gate] fingerprint mismatches" line must say 0).
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -J pcg-single-hash-ab
#SBATCH -o GPU_baseline/job_single_hash_%j.out

# Submit from the repo root of the clone (the B200 host runs it from
# /home/gong_ten_neu/pcg-square-cosim); paths follow the submit directory.
set -u
export PATH="$HOME/opt/cuda-13.0/bin:$PATH"
export NVCC="$HOME/opt/cuda-13.0/bin/nvcc"
export GPUNTT_INC="$HOME/.local/include/GPUNTT-1.0"
export GPUNTT_LIB="$HOME/.local/lib64"      # not ~/.local/lib on this host
HERE="${SLURM_SUBMIT_DIR:-$PWD}/GPU_baseline"
cd "$HERE"
T0=$(date +%s)

echo "################ SINGLE-HASH A/B ################"
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST"; hostname; date -Is
echo "commit: $(git -C "$HERE" rev-parse HEAD)  branch: $(git -C "$HERE" rev-parse --abbrev-ref HEAD)"
echo "---- NODE-LEVEL co-tenancy BEFORE ----"
squeue -w "$(hostname -s)" -o "%.10i %.14u %.24j %.8T %.4C %b" 2>&1
echo "loadavg: $(cat /proc/loadavg)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
echo

chmod +x run_single_hash_ab.sh
./run_single_hash_ab.sh
RC=$?

echo
echo "################ runner exit=$RC ################"
echo "---- NODE-LEVEL co-tenancy AFTER ----"
squeue -w "$(hostname -s)" -o "%.10i %.14u %.24j %.8T %.4C %b" 2>&1
echo "loadavg: $(cat /proc/loadavg)"
OUT="$HERE/results_b200/single_hash_ab"
echo "---- deliverable ----"
[ -s "$OUT/e2e_single_hash_ab.csv" ] && echo "OK  e2e_single_hash_ab.csv ($(wc -l < "$OUT/e2e_single_hash_ab.csv") lines)" || echo "MISSING e2e_single_hash_ab.csv"
grep -c MISMATCH "$OUT/e2e_single_hash_ab.csv" | sed 's/^/MISMATCH rows: /'
grep -alE "out of memory|illegal memory|CUDA Error|terminate called|Aborted" "$OUT"/*.log 2>/dev/null | while read -r f; do
  echo "=== $(basename "$f") ==="; grep -aE "out of memory|illegal memory|CUDA Error|terminate called|Aborted" "$f" | head -2; done
T1=$(date +%s); echo "TOTAL WALL CLOCK: $((T1-T0)) s"
exit $RC
