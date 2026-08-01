# New-algorithm GPU baseline — DPF + ChaCha8 convert on L40S silicon

Production kernels (caliber D1), single party, no network. Measures the
all-GPU cost of the **new Beaver-corrected output layer**: `dpf_gpu_batch_full_eval_device`
(ChaCha8 tree PRG) → `dpf_out_sums` (per-leaf ChaCha8 out-hash H′ + sparse
mod-P convert + control bit) → `dpf_out_scatter_g` (y = C + τ?CW, negacyclic
fold). **No modular multiply on the device.** This is the GPU baseline the PIM
speedup divides into (SM does DPF+convert+NTT; PIM does DPF+convert, NTT on SM).

- Bench: `endtoend/dpf_real_bench.cu` (adapted to `dpf_out_sums`/`dpf_out_scatter_g`).
- Host: NVIDIA L40S (AD102, SM89), CUDA 12.6, patched GPUNTT 95c739c.
- tier: **silicon phase-wall** (cudaEvent; kernel-only via nsys), 5 iters, 1 warm.
- prg = ChaCha8 (matches the PIM tree PRG and the CPU protocol default).

## Regime curve (total leaves → DPF+convert ns/leaf)

| total leaves | shape (n,B) | ns/leaf | regime |
|---|---|---|---|
| 0.52 M | 12, 128 | 0.720 | launch-dominated |
| 1.05 M | 14, 64  | 0.470 | launch→bandwidth |
| 4.2 M  | 16, 64  | 0.247 | bandwidth |
| 16.8 M | 16, 256 | 0.204 | plateau |
| 16.8 M | 18, 64  | 0.229 | plateau |
| 67 M   | 18, 256 | 0.204 | plateau |
| 67 M   | 20, 64  | 0.228 | plateau |
| 268 M  | 20, 256 | 0.210 | plateau |
| 268 M  | 22, 64  | 0.226 | plateau |

Shape-consistency (same total leaves, different n,B) holds to ±6%, confirming
ns/leaf is total-leaf-driven (not shape-driven) → the interp curve is valid.
Plateau ≈ **0.216 ns/leaf**.

## Comparison to the OLD algorithm (per-leaf modmul leaf-convert)

The old DPF+lc L40S plateau was ~0.19 ns/leaf; the new DPF+convert plateau is
~0.216 — **+14% on the GPU** because the per-leaf 62-bit modmul was replaced by
a per-leaf ChaCha8 out-hash (more ARX work, but no multiplier). The same hash
lands on the PIM side, so both baselines move together; the speedup ratio is
what the campaign reports.

## Curve used by the collector (`pim/tools/newalgo_campaign.py`, `GPU_BC`)

```
[(0.52e6, .720), (1.05e6, .470), (4.2e6, .247), (16.8e6, .216),
 (67.1e6, .216), (268e6, .218), (1.07e9, .220)]
```

5 non-L40S devices: scaled from this L40S absolute by the per-device
old-silicon ratio (h200/b200 measured; a100/ada5k/gddr7 modeled) — flagged
`est` in the speedup table (new-algorithm silicon exists only on L40S).
