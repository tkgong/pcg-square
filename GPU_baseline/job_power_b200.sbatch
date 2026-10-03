#!/bin/bash
# B200 board power per phase (idle, DPF, merge NTT, four-step SM lane) for the system-energy estimate.
# Submit from the repo root of the clone:   sbatch GPU_baseline/job_power_b200.sbatch
# ~6 min of measurement (25 phases x 10 s) plus the build. Deliverables: GPU_baseline/results_b200/power/
# {power_trace.csv, phases.csv, gpu.txt, *.log} and this job's .out.
#SBATCH -p b200-batch
#SBATCH -N 1
#SBATCH --gres=gpu:b200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 00:40:00
#SBATCH -J pcg-power-b200
#SBATCH -o GPU_baseline/job_power_%j.out
set -u
export PATH="$HOME/opt/cuda-13.0/bin:$PATH"
export NVCC="$HOME/opt/cuda-13.0/bin/nvcc"
export GPUNTT_INC="$HOME/.local/include/GPUNTT-1.0"
export GPUNTT_LIB="$HOME/.local/lib64"      # not ~/.local/lib on this host
HERE="${SLURM_SUBMIT_DIR:-$PWD}/GPU_baseline"
cd "$HERE"
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST"; hostname; date -Is
echo "commit: $(git -C "$HERE" rev-parse HEAD)  branch: $(git -C "$HERE" rev-parse --abbrev-ref HEAD)"
squeue -w "$(hostname -s)" -o "%.10i %.14u %.24j %.8T %.4C %b" 2>&1
nvidia-smi --query-gpu=index,name,driver_version,power.draw,power.limit,memory.used,utilization.gpu --format=csv
python3 -c "import pynvml; print('pynvml OK')" 2>&1 || echo "pynvml missing: energy_mJ will be NA (nvidia-smi trace still taken)"
which dcgmi 2>/dev/null && echo "dcgmi present" || echo "no dcgmi"
chmod +x run_power.sh
GPU=0 SECS=10 ./run_power.sh
RC=$?
echo "runner exit=$RC"; ls -l results_b200/power | head -40
exit $RC
