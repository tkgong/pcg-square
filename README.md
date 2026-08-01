# pcg-square

Open-source staging tree for the PCG² artifact: accelerating ring-LPN
PCG (pseudorandom correlation generation) on a PIM-coupled GPU
architecture. Three self-contained components:

## Layout

```
CPU_baseline/   Reference 2PC protocol (real two-party, verified)
  pcg_ole_2pc/      PCG-OLE protocol (Beaver-corrected output layer)
  half_tree_dpf/    half-tree DPF: fused Gen + full-domain eval (ChaCha8)
  bool_circuit/     Kogge-Stone / Gilboa / delta-shift circuits over FerretCOT
  test/             correctness tests + bench_pcg_ole_2pc (2PC wall, verify-gated)
  chacha8.h ffp.h prof.h timing.h comm_trace.h   shared headers
  run_2pc CMakeLists.txt

GPU_baseline/   GPU comparison world: naive NTT + 4-step NTT ONLY
  run.sh                  ** one-shot reproduction on any CUDA GPU (B200/
                          H200/L40S): builds, plateau-checks, nsys per-kernel
                          profile, emits roofline_points.csv + NTT tables **
  naive_ntt_bench.cu      textbook per-stage radix-2 NTT, BITEXACT-gated
                          against the 4-step backend (vs_4step=BITEXACT)
  ntt_batch_bench.cu      batched NTT bench, --backend {4step,v1}
  common/poly_mul_gpuntt_4step.cu   4-step backend (PCG_4STEP_STAGE_MS stages)
  dpf_real_bench.cu       GPU DPF+convert baseline (production kernels)
  common/                 dpf_gpu / aes_gpu / leaf_convert_cuda / poly_mul_cuda
  test/                   out_hash_cuda_bitexact + GPU micro-benches
  run_ntt_secparams.sh run_baseline_grid.sh
  The merge backend is intentionally NOT part of this tree: the paper's GPU
  comparison ladder is naive -> 4-step -> 4-step+DRU.

PIM/        Proposed design: in-bank SPU (ChaCha DPF) + channel DRU
  sim/            Ramulator2 with GGM_EXTEND / fused CONVERT PU timing model
  tools/          trace generator (word-major, ping-pong, broadcast),
                  collectors (e2e_secparams, sched_timeline, net_model),
                  figure scripts
  results/        measured campaign outputs (L40S anchor + 5 memory gens)
  docs/           ISA + datapath + DRU + co-exec + network model
```

## Notes for packaging

- Include paths still reference the original repo layout (`common/...`,
  `pim/...`); adjust before standalone release.
- `GPU_baseline` deliberately excludes the merge backend
  (`poly_mul_gpuntt.cu` stays out of the comparison world).
- `CPU_baseline`'s optional CUDA hooks (`PCG_ENABLE_CUDA`) reference
  files now under `GPU_baseline/common/`.
