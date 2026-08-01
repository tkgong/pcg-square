#!/usr/bin/env bash
# Baseline benchmark grid v2 — runnable skeleton (spec: bench_baseline_spec.md).
#
# Runs the (N x t x path) e2e grid with one GPU per party, records all three
# timing tiers (e2e / wall phases / nsys kernel sums) under one CSV schema,
# is resumable per point, and refuses to run on a busy GPU.
#
# Usage:
#   GPU_A=0 GPU_B=1 ./endtoend/run_baseline_grid.sh [--platform b200|l40s]
#       [--build-dir build-cuda] [--dry-run] [--ntt-sweep]
#
# Notes:
#   * gpu-dpf-copyout points need the PCG_DPF_DEVICE_LEAVES=0 env gate in
#     pcg_ole_impl.h (spec section 3); until it exists they are SKIP(no-toggle).
#   * No sudo assumed: clocks are RECORDED (not locked) at session start/end.
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="b200"
BUILD_DIR="${REPO_ROOT}/build-cuda"
DRY_RUN=0
DO_NTT_SWEEP=0
GPU_A="${GPU_A:-0}"          # party 1 (ALICE)
GPU_B="${GPU_B:-1}"          # party 2 (BOB) — MUST differ from GPU_A
PORT_BASE=7800
NSYS="${NSYS:-nsys}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)  PLATFORM="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --dry-run)   DRY_RUN=1; shift;;
    --ntt-sweep) DO_NTT_SWEEP=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

BIN="${BUILD_DIR}/bin/bench_pcg_ole_2pc"
POLY_BIN="${BUILD_DIR}/bin/bench_poly_mul"
OUT="${REPO_ROOT}/data/bench/baseline_v2/${PLATFORM}"
DONE_DIR="${OUT}/done"
NSYS_DIR="${OUT}/nsys"
MANIFEST="${OUT}/manifest.tsv"

# Grid (spec section 1)
NLOGS=(16 18 20 22 24)
TS=(4 8 16 32)
PATHS=(cpu-dpf gpu-dpf-copyout gpu-dpf-device-resident)
PRG=chacha8; W=1; C=2
NTT_BATCHES=(1 4 16)

# ── Defensive checks ─────────────────────────────────────────────────────────
die() { echo "FATAL: $*" >&2; exit 1; }

[[ -x "$BIN" ]] || die "missing binary $BIN (build with -DPCG_ENABLE_CUDA=ON -DPCG_ENABLE_PROFILING=ON)"
[[ "$GPU_A" != "$GPU_B" ]] || die "GPU_A == GPU_B ($GPU_A): one GPU per party is MANDATORY (spec 4 / caliber D3)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
nvidia-smi -i "$GPU_A" >/dev/null 2>&1 || die "GPU index $GPU_A not visible"
nvidia-smi -i "$GPU_B" >/dev/null 2>&1 || die "GPU index $GPU_B not visible"

gpu_idle_or_die() {
  local idx="$1"
  local util mem
  util=$(nvidia-smi -i "$idx" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  mem=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits)
  if (( util > 5 )) || (( mem > 2048 )); then
    die "GPU $idx busy (util=${util}%, mem=${mem}MiB) — refusing to bench (spec 4)"
  fi
}
gpu_idle_or_die "$GPU_A"
gpu_idle_or_die "$GPU_B"

mkdir -p "$OUT" "$DONE_DIR" "$NSYS_DIR"

# ── Environment / clocks header (spec 4, caliber D12) ────────────────────────
record_env() {
  local tag="$1"
  {
    echo "== ${tag} $(date -Is) =="
    echo "commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "host: $(hostname)"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    nvidia-smi -q -d CLOCK
    nvidia-smi -q -d POWER | grep -E "Power Limit|Power Draw" || true
  } >> "${OUT}/env.txt"
}
record_env "session-start"

env_header() {  # one-line CSV header comment for a point CSV
  local sm mem
  sm=$(nvidia-smi -i "$GPU_A" --query-gpu=clocks.sm --format=csv,noheader,nounits)
  mem=$(nvidia-smi -i "$GPU_A" --query-gpu=clocks.mem --format=csv,noheader,nounits)
  echo "# ENV,host=$(hostname),gpu=$(nvidia-smi -i "$GPU_A" --query-gpu=name --format=csv,noheader),driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1),commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown),date=$(date -Is),sm_clock_mhz=${sm},mem_clock_mhz=${mem},platform=${PLATFORM}"
}

manifest() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date -Is)" >> "$MANIFEST"; }

# ── Per-point runner ─────────────────────────────────────────────────────────
# iters convention (spec 2, caliber D11): warm >=1; iters >=5 small N, >=3 large.
iters_for() { local nlog="$1"; if (( nlog <= 20 )); then echo 5; else echo 3; fi; }

# Launch both parties with per-party CUDA_VISIBLE_DEVICES (run_2pc can't).
run_2pc_pinned() {
  local port="$1"; shift
  local extra_env="$1"; shift          # e.g. "PCG_DPF_DEVICE_LEAVES=0" or ""
  local logdir="$1"; shift
  local prefix=("$@")                  # e.g. nsys wrapper for party 1, or empty

  ( cd "$REPO_ROOT" && mkdir -p data
    env CUDA_VISIBLE_DEVICES="$GPU_A" ${extra_env:+$extra_env} \
      "${prefix[@]}" "$BIN" 1 "$port" "${ARGS[@]}" \
      >"$logdir/p0.log" 2>&1 ) &
  local p0=$!
  sleep 0.5
  ( cd "$REPO_ROOT" && \
    env CUDA_VISIBLE_DEVICES="$GPU_B" ${extra_env:+$extra_env} \
      "$BIN" 2 127.0.0.1 "$port" "${ARGS[@]}" \
      >"$logdir/p1.log" 2>&1 )
  local s1=$?
  wait "$p0"; local s0=$?
  return $(( s0 || s1 ))
}

port=$PORT_BASE
for nlog in "${NLOGS[@]}"; do
  for t in "${TS[@]}"; do
    for path in "${PATHS[@]}"; do
      point="N$(printf '%02d' "$nlog")_t${t}_${path}"
      donef="${DONE_DIR}/${point}.ok"
      [[ -e "$donef" ]] && { echo "SKIP (done): $point"; continue; }
      port=$(( port + 1 ))

      # ---- feasibility guards (spec 1) ----
      # device leaf tensor ~= 2*N*t*16 * 2.5 bytes per party
      N=$(( 1 << nlog ))
      need_gib=$(( (2 * N * t * 16 * 5 / 2) >> 30 ))
      free_mib=$(nvidia-smi -i "$GPU_A" --query-gpu=memory.free --format=csv,noheader,nounits)
      if [[ "$path" != "cpu-dpf" ]] && (( need_gib * 1024 > free_mib )); then
        manifest "$point" "SKIP(mem:${need_gib}GiB>free)"; continue
      fi
      if [[ "$path" == "gpu-dpf-copyout" ]] && \
         ! grep -q "PCG_DPF_DEVICE_LEAVES" "${REPO_ROOT}/pcg_ole_2pc/pcg_ole_impl.h"; then
        manifest "$point" "SKIP(no-toggle)"; continue   # spec 3 prerequisite
      fi

      iters=$(iters_for "$nlog"); warm=1
      case "$path" in
        cpu-dpf)                  dpf=cpu; extra_env="";;
        gpu-dpf-copyout)          dpf=gpu; extra_env="PCG_DPF_DEVICE_LEAVES=0";;
        gpu-dpf-device-resident)  dpf=gpu; extra_env="";;
      esac
      logdir="${OUT}/${point}"; mkdir -p "$logdir"
      csv="${logdir}/profile"   # bench appends .p0.csv/.p1.csv itself via --profile-csv

      ARGS=(--N "$N" --c "$C" --t "$t" --w "$W" --prg "$PRG"
            --dpf "$dpf" --poly-mul cuda-gpuntt-merge
            --warmup "$warm" --iters "$iters"
            --profile-csv "${csv}.csv")

      echo "== $point (iters=$iters, port=$port) =="
      if (( DRY_RUN )); then manifest "$point" "DRYRUN"; continue; fi

      gpu_idle_or_die "$GPU_A"; gpu_idle_or_die "$GPU_B"

      # ---- pass 1: unprofiled — e2e + wall-phase tiers ----
      if ! run_2pc_pinned "$port" "$extra_env" "$logdir"; then
        manifest "$point" "FAIL(run)"; continue
      fi
      grep -qi "verify.*pass\|PASS" "$logdir/p0.log" || { manifest "$point" "FAIL(verify)"; continue; }

      # ---- pass 2: nsys on party 1 only — kernel tier ----
      port=$(( port + 1 ))
      if command -v "$NSYS" >/dev/null; then
        run_2pc_pinned "$port" "$extra_env" "$logdir" \
          "$NSYS" profile -o "${NSYS_DIR}/${point}" --force-overwrite=true \
          || echo "WARN: nsys pass failed for $point" >&2
        "$NSYS" stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
          --format csv --output "${NSYS_DIR}/${point}" \
          "${NSYS_DIR}/${point}.nsys-rep" >/dev/null 2>&1 || true
      else
        echo "WARN: nsys not found — kernel tier missing for $point" >&2
      fi

      # ---- merge tiers into the point CSV (schema: spec section 7) ----
      { env_header
        echo "run_id,platform,path,prg,c,t,w,N_log2,batch,party,tier,name,value_ms,iters,warmup,phase_schema,verify,notes"
        # e2e tier from bench_result line; wall tier from profile CSVs; kernel
        # tier from nsys stats CSVs — assembled by the aggregator:
      } > "${logdir}/${point}.csv"
      python3 "${REPO_ROOT}/scripts/aggregate_profile.py" \
          "${csv}.p0.csv" >> "${logdir}/${point}.csv" 2>/dev/null \
        || echo "# TODO: extend scripts/aggregate_profile.py for schema v2" >> "${logdir}/${point}.csv"

      touch "$donef"; manifest "$point" "OK"
    done
  done
done

# ── NTT batch sweep (separate microbench, spec 8) ────────────────────────────
if (( DO_NTT_SWEEP )); then
  [[ -x "$POLY_BIN" ]] || die "missing $POLY_BIN"
  sweep_csv="${OUT}/ntt_batch_sweep.csv"
  if [[ ! -e "${DONE_DIR}/ntt_sweep.ok" ]]; then
    env_header > "$sweep_csv"
    CUDA_VISIBLE_DEVICES="$GPU_A" "$POLY_BIN" --N 4096 --batch 1 --csv-header --iters 1 --warmup 0 \
      | head -1 >> "$sweep_csv" || true
    for nlog in "${NLOGS[@]}"; do
      for b in "${NTT_BATCHES[@]}"; do
        N=$(( 1 << nlog ))
        skip_cpu=""; (( nlog >= 22 )) && skip_cpu="PCG_BENCH_SKIP_CPU=1"   # verify=SKIP rows
        echo "== ntt N=2^$nlog batch=$b =="
        env CUDA_VISIBLE_DEVICES="$GPU_A" ${skip_cpu:+$skip_cpu} \
          "$POLY_BIN" --N "$N" --batch "$b" --iters 20 --warmup 5 >> "$sweep_csv"
        if command -v "$NSYS" >/dev/null; then
          env CUDA_VISIBLE_DEVICES="$GPU_A" ${skip_cpu:+$skip_cpu} \
            "$NSYS" profile -o "${NSYS_DIR}/ntt_N${nlog}_b${b}" --force-overwrite=true \
            "$POLY_BIN" --N "$N" --batch "$b" --iters 5 --warmup 2 >/dev/null 2>&1 || true
        fi
      done
    done
    touch "${DONE_DIR}/ntt_sweep.ok"; manifest "ntt_sweep" "OK"
  fi
fi

record_env "session-end"
# Clock-drift flag (spec 4): compare first/last SM clock in env.txt manually or:
first=$(grep -m1 "SM *:" "${OUT}/env.txt" | grep -o '[0-9]*' | head -1 || echo 0)
last=$(grep "SM *:" "${OUT}/env.txt" | tail -1 | grep -o '[0-9]*' | head -1 || echo 0)
if (( first > 0 && last > 0 )) && \
   (( 100 * (first > last ? first - last : last - first) / first > 5 )); then
  manifest "SESSION" "CLOCK_DRIFT(${first}->${last}MHz)"
  echo "WARNING: SM clock drifted ${first} -> ${last} MHz across the session" >&2
fi

echo "Done. Manifest: $MANIFEST"
