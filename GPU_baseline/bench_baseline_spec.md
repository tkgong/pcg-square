# Baseline benchmark spec v2 — THE reference protocol for all PCG/PIM claims

Status: spec (2026-07-11). Fixes every caliber issue in
`docs/caliber_review_b200.md` (D1–D14). Runner skeleton:
`endtoend/run_baseline_grid.sh`. Results land under `data/bench/baseline_v2/`
(gitignored; summary tables go into docs with CSV provenance lines).

Primary platform: **B200 cluster** (sm_100, one GPU per party — mandatory).
Secondary: **L40S** (sm_89, one GPU per party). No shared-GPU numbers are
publishable under this spec (fixes D3).

## 1. Grid

| axis | values | notes |
|---|---|---|
| N | 2¹⁶, 2¹⁸, 2²⁰, 2²², 2²⁴ | ring degree |
| t | 4, 8, 16, 32 | fixes D7 (B200 had only t=8) |
| path | `cpu-dpf`, `gpu-dpf-copyout`, `gpu-dpf-device-resident` | see §3 |
| PRG | chacha8 (fixed) | production PRG; AES only as a labeled side-study |
| w, c | 1, 2 (fixed) | binary tree, protocol standard |
| NTT batch sweep | batch ∈ {1, 4, 16} × same N | **separate** microbench (`bench_poly_mul`), not inside the e2e grid |

Constraints/skips (runner enforces):
- `w=1` requires `2N/t` a power of two — always true on this grid.
- Device leaf tensor per party = `2Nt·16 B` (+ ping-pong + hashed ≈ 2.5×):
  N=2²⁴,t=32 → ~43 GB/party. Fits B200 (183 GB); **skip on L40S (48 GB)** —
  runner checks free memory and writes a `SKIP(mem)` row instead of dying.
- `cpu-dpf` at N=2²⁴ is minutes/iter: iters may drop to 3 but never below.

## 2. Timing tiers — recorded TOGETHER per grid point (fixes D5, D13)

Every point produces rows at all three tiers in ONE schema; a `tier` column
disambiguates. No number may be quoted without its tier.

| tier | definition | source |
|---|---|---|
| `kernel` | per-kernel duration sums over the measured iters, divided by iters | `nsys profile` + `nsys stats --report cuda_gpu_kern_sum` (one row per kernel name) + `cuda_gpu_mem_time_sum` (memcpy rows, tier `memcpy`) |
| `wall` | per-phase host span (chrono/cudaEvent incl. copies & packing) | `--profile-csv` phase timers (`PCG_ENABLE_PROFILING`), phase schema v2 (see §5) |
| `e2e` | mean ms/iter over measured iters | `bench_result` line of `bench_pcg_ole_2pc` |

Run discipline: **warmup ≥ 1 always; iters ≥ 5 for N ≤ 2²⁰, ≥ 3 for
N ≥ 2²²** (fixes D11). Warmup and iters are CSV columns. The nsys pass is a
*separate invocation* of the identical command (profiling perturbs walls);
the wall/e2e pass runs unprofiled. Both passes share one `run_id`.

## 3. Paths — REAL kernels only (fixes D1, D10, D14)

All three paths run the production protocol binary `bench_pcg_ole_2pc`:

- `cpu-dpf` — `--dpf cpu --poly-mul cuda-gpuntt-merge` (CPU DPF reference,
  GPU NTT; isolates the DPF contribution).
- `gpu-dpf-copyout` — `--dpf gpu` with the device-leaves fast path DISABLED
  (leaves D2H + host leaf_conversion). **Prerequisite code change:** add env
  gate `PCG_DPF_DEVICE_LEAVES=0` in `pcg_ole_impl.h` (the fast path is
  currently unconditional when CUDA is on). Until that lands, the runner
  marks these points `SKIP(no-toggle)`.
- `gpu-dpf-device-resident` — `--dpf gpu` default fast path
  (`dpf_gpu_batch_full_eval_device` + `dpf_leaf_sums`/`dpf_leaf_scatter_g`).

Kernels that define the baseline: `hash_kernel_chacha8`, `expand_kernel`,
`leaf_sum_kernel`, `partial_fold_kernel`, `leaf_scatter_kernel`, GPU-NTT
merge Forward/InverseCore + pointwise, 4-step cores/transpose/twist.

**The timing-only stand-ins are RETIRED for baseline purposes**:
`common/dpf_gpu_chacha.cu` (tkgong branch) and
`endtoend/dpf_trace_bench.cu`'s single-bin `leaf_convert_kernel` are kept
ONLY as Accel-Sim/NVBit trace-capture vehicles (small, closed-form traces).
Any ns/leaf they produce must be labeled `trace-capture, not baseline` and
never enter a model constant. The timing PRG must not be used at w>1 (it
returns identical children — caliber review D10).

## 4. Run conventions (fixes D3, D9, D12)

- **One GPU per party**: party 1 gets `CUDA_VISIBLE_DEVICES=$GPU_A`, party 2
  `CUDA_VISIBLE_DEVICES=$GPU_B`, launched as separate processes (the stock
  `run_2pc` cannot set per-party env — the runner inlines the launch).
- **Header row per CSV** records: hostname, GPU name, driver, CUDA runtime,
  arch flags, git commit, date, and the full `nvidia-smi -q -d CLOCK` dump
  (SM/mem clocks at run start), plus `nvidia-smi -q -d POWER` cap. No sudo →
  we cannot lock clocks; we RECORD them and re-record after the last point;
  if start/end SM clock differs >5%, the runner flags the whole session
  `CLOCK_DRIFT` in the manifest.
- **GPU idle check** before each point: utilization ≤ 5% and memory used
  ≤ 2 GiB on both assigned GPUs, else wait/abort.
- **Peak-BW denominators**: every BW% is reported against BOTH the nominal
  spec (B200: 8.0 TB/s; L40S: 0.864 TB/s) and the measured streaming roof
  (pointwise-mul plateau from the NTT sweep) — two columns, never a single
  ambiguous "of peak".
- NCU: if unlocked, one NCU pass per kernel at N=2²⁰ to calibrate the
  analytic byte formulas (§6); if locked, note `ncu=locked` in the manifest.

## 5. Phase schema v2 (fixes D2, D8)

Wall-tier rows use these phase names, with `phase_schema=v2` stamped in every
row. Scope definitions (device-resident path):

| phase | scope |
|---|---|
| `dpf_delta_shift_setup` | interactive DltSft 2-PC (host) |
| `dpf_batch_eval` | kernels + per-level CW exchange; NO leaf copy-out |
| `leaf_convert_kernels` | leaf_sum + fold + scatter kernel spans (cudaEvent) |
| `leaf_convert_net` | S/CW exchange rounds inside leaf_conversion |
| `leaf_g_copyout` | g D2H (N×8 B) |
| `output_pack` | host pack/copy inside poly_mul wrappers |
| `output_kernels` | NTT kernel spans (cudaEvent) |
| `net_wait` | blocking recv time outside the above |
| `other` | total − sum(above) — MUST be emitted so rows close to 100% |

Prerequisite code change: split today's `leaf_conversion_cw` and
`output_compute` scopes and add `net_wait`/`other`. Until it lands, the
runner emits the v1 phases with `phase_schema=v1-legacy` — comparable only
within themselves. Old (pre-device-resident) CSVs are schema `v0` and must
never be column-aligned with v1/v2 (caliber D2).

## 6. Byte-accounting appendix (fixes D4, D6, D9)

Traffic formulas per kernel — the ONLY formulas allowed in BW% claims.
Convention: DRAM-visible bytes = reads + writes (+ 2× per atomic RMW that
misses L2); element = 8 B coefficient; leaf block = 16 B; transform size = N
(negacyclic ψ-twist, never 2N).

| kernel | reads | writes | atomics | total/launch |
|---|---|---|---|---|
| `hash_kernel_chacha8` (level i, F=B·2^i, w=1) | F·16 (parents) | F·16 (hashed) | ~0 (1 atomicXor/block) | 32·F; Σ over levels ≈ 32·B·D |
| `expand_kernel` (level i) | F·16 + F·16 (parent + hashed) | 2F·16 (children) | 0 | 64·F; Σ ≈ 64·B·D |
| DPF total | | | | ≈ 96 B per leaf (= 192 B/parent/level, matches doc) |
| `leaf_sum_kernel` | B·D·16 | B·nparts·8 | 0 | ≈ 16 B/leaf |
| `leaf_scatter_kernel` | B·D·16 + B·8 (cws) | — | ≤ B·D hits × 16 (CAS RMW) | ≤ 32 B/leaf + N·8 memset |
| NTT merge Fwd/InvCore | batch·N·8 | batch·N·8 | 0 | batch·N·16 |
| 4-step transpose / ψ-twist | batch·N·8 | batch·N·8 | 0 | batch·N·16 |
| **pointwise mul** | **2·batch·N·8** | batch·N·8 | 0 | **batch·N·24 — NOT 16** (caliber D4: the blanket 16 B formula understates pointwise BW 1.5×) |
| PCIe (copyout path) | leaves D2H = B·D·16 | — | — | per pair |

Poly-mul count per iteration (c=2): `output_compute` runs **10** full
negacyclic poly-muls (x: c=2, a·a: c²=4, z: c²=4) = 30 transforms + 10
pointwise (caliber D6). Models quoting "8 muls" or "2 fwd/pair" must label
that as an *optimized/resident* design assumption, not the measured baseline.

## 7. CSV schema (one file per point + merged `baseline_v2.csv`)

```
# ENV,host=...,gpu=...,driver=...,cuda=...,arch=...,commit=...,date=...,sm_clock_mhz=...,mem_clock_mhz=...,power_cap_w=...,ncu=locked|ok
run_id,platform,path,prg,c,t,w,N_log2,batch,party,tier,name,value_ms,iters,warmup,phase_schema,verify,notes
```

- `tier` ∈ {e2e, wall, kernel, memcpy}; `name` = phase name (wall), kernel
  name (kernel), `total` (e2e), memcpy direction (memcpy).
- `verify` = PASS/FAIL/SKIP from the protocol's built-in check — a point with
  FAIL is discarded, never averaged around.
- `party` ∈ {p0, p1}; publish p0 (ALICE) rows, keep both.
- Aggregation: mean over iters; per-row `notes` may carry `std=`.

## 8. NTT batch sweep (separate microbench)

`bench_poly_mul --N {2^16..2^24} --batch {1,4,16}`, warmup 5 / iters 20
(defaults), `PCG_BENCH_SKIP_CPU=1` for N ≥ 2²² (the `*_correct` columns are
then meaningless — recorded as `verify=SKIP`). nsys pass on top for
kernel-tier rows. Purpose: (a) streaming roof (pointwise, with the CORRECT
24 B formula), (b) NTT core compute-bound plateau, (c) the L2→DRAM regime
break location per device.

## 9. Deliverables & acceptance

- `data/bench/baseline_v2/<platform>/<point>.{p0,p1}.csv` + `nsys/` reports +
  `manifest.tsv` (point → status: OK/SKIP(reason)/FAIL) + `env.txt`.
- Docs get summary tables ONLY with the provenance line
  `source: data/bench/baseline_v2/... run_id=...`.
- Acceptance: (1) every published constant traceable to a CSV row with a tier
  tag; (2) wall phases close to 100±2% of e2e; (3) DPF ns/leaf recomputed
  from real kernels replaces 0.744/0.912 in
  `pim/tools/{superiority_map,coexec_model,e2e_stitched}.py`; (4) all BW%
  double-denominated.
