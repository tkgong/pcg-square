# Task: B200 GPU baseline for PCG expansion — full table

Cluster node with NVIDIA B200(s). Check out branch `final`, work in
`pcg-square/GPU_baseline/`. GPU-NTT must be installed (override `GPUNTT_INC` /
`GPUNTT_LIB` if it is not under `~/.local`).

One command produces everything:

```bash
cd pcg-square/GPU_baseline
git fetch && git reset --hard origin/final     # this package is only valid at HEAD
chmod +x run_pcg_baseline.sh
./run_pcg_baseline.sh
```

The identical script has already been run to completion on an idle L40S, so
every cell you return is directly comparable to ours. Expect 4–8 h.

## What it measures, and why each battery exists

**1. `dpf_shape.csv` — the ns/leaf(n, B) surface.** A DPF block's cost is set by
exactly two numbers: the tree depth `n` and the number of parallel trees
`B = t²`. It is *not* a function of the leaf count alone. Measured on L40S at
2.68e8 leaves: `(n=16, B=4096)` = 0.192 ns/leaf, `(n=24, B=16)` = 0.267 — 39%
apart at identical leaf counts, because 16 trees cannot fill the machine. Every
security row reads off this surface. Our previous baseline used a curve indexed
by leaf count alone, which collapsed this dimension and handed all 30 rows the
same number; that is the error this battery exists to prevent.

**2. `ntt_stages.csv` — NTT, both backends, both bit-reversal conventions.**
The library is natural-in / bit-reversed-out, so a four-step composition must
undo the permutation somewhere. Two placements are measured:
- **standalone (default)** — a separate kernel, so whatever is written to memory
  between transforms is in natural order. This is the convention the DRU offload
  assumes and the one the paper uses.
- **fold** (`PCG_SQUARE_FOLD_BREV=1`) — folded into neighbouring kernels'
  indices. On L40S this is 1.1–1.4× faster up to logN 22 but **1.4× slower at
  logN 24**, because bit-reversed destination rows destroy DRAM row locality at
  that stride. Report both; do not drop either.

**3. `dpf_grid.csv` — the 30 security rows × 5 tiers**, each at its own real
`(n = logN+1−log₂t, B = t²)`.

**4. `e2e_full.csv` — the true end-to-end, two arms per cell.** One process,
one wall clock: `c²` blocks × `t²` instances + output layer + per-block Beaver
opens + `2c²` poly_muls. Every earlier "e2e" figure in this project was
arithmetic (`c²·dpf + 2c²·ntt`) and was never an actual run.

The GPU baseline is bounded from **both** sides rather than assumed:

- **`serial`** — what `pcg_ole_impl.h:114-116` actually does: `c²` blocks one at
  a time, one block resident.
- **`batched`** — all `c²·t²` instances in a single expansion, then scatter per
  block by pointer offset. This is a legal reordering (the instances are
  independent; only the scatter is per-block) and costs `c²` times the memory.

Measured on L40S at `c=4, t=16, logN=20`: the expansion differs by only **8.7%**
(84.9 → 78.1 ms) — one block of 256 trees already saturates the device, so the
serial arm is not a strawman. The batched arm additionally coalesces the
per-level CW exchanges (**272 → 17**), which is protocol-level ganging rather
than a memory-layout choice, so read its network column with that in mind.

Both arms are run automatically; report both. Do not drop the slower one.

## Things you must not "improve"

**Do not substitute for OOM or crashed cells.** Report them as `OOM_SKIP` /
`CRASH` and leave the row empty. Nothing is extrapolated into a missing cell.
The `batched` arm will OOM at the large `(c, t, N)` corners on a 180 GB device —
that is expected and is itself a result.

## Gates — report PASS/FAIL for each, never silently continue

- `square_verify.log` **and** `square_verify_fold.log` both end in `GATE PASS`.
  The script aborts otherwise, so a completed run implies both passed.
- Every `naive` row in `ntt_stages.csv` shows `BITEXACT` (a few large-`N`,
  `batch=64` naive cells fail on the library's own grid limit — that is expected
  and is not our bug; report them as-is).
- Every `square` row with `brev_mode=standalone` has a **non-zero** `brev_ms`;
  every `fold` row has `brev_ms = 0.0000`. If this is inverted, the env flag did
  not take effect.
- `e2e_full.csv`: `wall_ms ≥ expand + convert + beaver + ntt` (it is the
  enclosing clock). A gap beyond a few percent means contention.
- `e2e_full.csv`: both `serial` and `batched` rows present for every cell that
  did not OOM; `batched` never slower than `serial` on the expansion phase.
- Monotonicity: within one `(c,t)` row, `wall_ms` and `ms_per_mul` are
  non-decreasing in `N`, and `wall_ms` increases with the tier's RTT.

## Machine hygiene — this is not optional

The script records `nvidia-smi --query-compute-apps` before and after; both
files come back. Our first L40S campaign was polluted by another user's training
jobs holding 167 GB across four GPUs, and several cells came back 3–6× their
neighbours before we caught it. **If the B200 is shared, say so explicitly and
mark the affected cells rather than reporting them as clean.** A consistency
check you can run yourself: within `ntt_stages.csv`, `ms_per_mul` divided by
`sum_ms/batch` should sit in [0.75, 1.35] for logN ≥ 16; anything outside that
was contended.

## Deliverables — verbatim, no post-processing

1. `pcg_baseline_out/dpf_shape.csv`
2. `pcg_baseline_out/ntt_stages.csv`
3. `pcg_baseline_out/dpf_grid.csv`
4. `pcg_baseline_out/e2e_full.csv`
5. `pcg_baseline_out/square_verify.log`, `square_verify_fold.log`
6. `pcg_baseline_out/topo.txt`, `gpu_before.txt`, `gpu_after.txt`
7. A tar of `pcg_baseline_out/*.log` — we parse them ourselves.

Also report: GPU count, interconnect class from `topo.txt` (NV# vs PIX/PHB),
driver version, total wall-clock, and any OOM or crash verbatim.
