# B200 GPU baseline for PCG expansion — full table

Companion to `results_newalgo_gpu_l40s.md`. Same script, same grid, so every
cell here is directly comparable to the L40S campaign.

**Raw data is committed alongside this file** in `pcg_baseline_out/` (677 files,
612 KB): the four CSVs, both verify logs, `topo.txt`, `gpu_before.txt`,
`gpu_after.txt`, and all 669 per-cell `*.log`. It is force-added past
`.gitignore` on purpose — every number below is traceable to a file in that
directory rather than to this summary.

---

## 1. Provenance

| | |
|---|---|
| Commit | `45b5f8ee8a1643435dbed0b2a820a650b54f17a2` (branch `final`) |
| Command | `LGS` unset (full grid) → `./run_pcg_baseline.sh` |
| Slurm job | 239193, partition `b200-batch`, node **b0010** |
| GPU | **1 × NVIDIA B200**, 183359 MiB, sm_100 (compute cap 10.0) |
| Driver | **595.71.05** |
| Toolchain | CUDA 13.0 at `~/opt/cuda-13.0`, GPU-NTT at `~/.local` with `GPUNTT_LIB=~/.local/lib64` |
| **Total wall clock** | **1973 s = 32 min 53 s** |
| Date | 2026-07-30 |

`GPUNTT_LIB` and `NVCC` both need overriding on this host: the script defaults
to `~/.local/lib` (the install is under `lib64`) and searches only
`/usr/local/cuda*` for nvcc.

### Interconnect class

`topo.txt` from this run contains **only `GPU0`** — one GPU was allocated, so
there is no GPU↔GPU pair and **no `NV#` classification can be read off it**.
The only non-`SYS` relation present is `GPU0 ↔ NIC5 = PIX`. (The node's
GPU-to-GPU fabric is NVLink; an earlier 2-GPU allocation on this cluster showed
`NV18`. That is a different allocation and is *not* evidence from this
`topo.txt`.)

### Machine sharing — the node was NOT idle

`gpu_before.txt` and `gpu_after.txt` both contain only a header:

```
pid, used_gpu_memory [MiB]
```

That reads as a clean machine and **it is misleading**. Inside a Slurm cgroup,
`nvidia-smi --query-compute-apps` can only ever see this job's own processes on
this job's own GPU; it is structurally blind to another user's job on a
different GPU of the same physical node — which is exactly the pollution mode
the L40S campaign hit. Node-level `squeue`, recorded before *and* after the run:

| | this job (239193) | co-tenant (238644, `scale-4b34-full`) |
|---|---|---|
| GPUs on b0010 | **1** | **6** |
| CPU cores on b0010 | 32 | 96 |
| memory | 400 GB requested (24.5 GB peak RSS) | 600 GB |

b0010 has 128 cores and 8 B200s, so **7 of 8 GPUs and 100 % of the CPU cores
were allocated** for the whole run. `loadavg` 8.79 → 8.98.

**What this does and does not contaminate.** The GPU die itself was exclusively
ours — cgroup-isolated, 0 MiB / 0 % utilisation at start, no foreign process on
it at either end. Device-resident kernel work is therefore clean, and three
independent checks agree (gate 4 accounts the wall clock to 0.01 %; the fixed
overhead below is constant across a 40× work range; the shape surface
reproduces L40S's ratio to 0.5 %). What is *not* protected is everything
host-mediated: CPU, PCIe root complex, host memory bandwidth, NUMA fabric. The
two anomalies in this run are both host-side (§7).

---

## 2. Gates

| # | Gate | Verdict |
|---|---|---|
| 1 | `square_verify.log` **and** `square_verify_fold.log` end in `GATE PASS` | **PASS** |
| 2 | every `naive` row in `ntt_stages.csv` is `BITEXACT` | **PASS — 45/45** |
| 3 | `standalone` ⇒ `brev_ms > 0`; `fold` ⇒ `brev_ms = 0.0000` | **PASS — 45/45 and 45/45** |
| 4 | `wall_ms ≥ expand + convert + beaver + ntt` | **PASS — 247 rows, 0 violations** |
| 5 | both arms per non-OOM cell; `batched` expand ≤ `serial` expand | **FAIL** (see §6) |
| 6 | monotonicity in `N` and in tier RTT | **FAIL** (see §6) |

Two notes where B200 differs from the package's expectation:

- Gate 2 — the package expects "a few large-`N`, `batch=64` naive cells fail on
  the library's own grid limit". **On B200 none failed.** logN 20–24 × batch
  4/16/64 are all `BITEXACT` (logN 24 / batch 64 = 5.0226 ms/mul).
- Gate 4 — the unaccounted gap `(wall − Σphases)/wall` has **median and maximum
  both 0.01 %**. The enclosing clock is fully explained by the phases.

---

## 3. Battery 1 — `dpf_shape.csv`: the ns/leaf(n, B) surface

82 rows, **0 non-ok**. The surface is real and it reproduces L40S almost
exactly. Along the 2.68 × 10⁸-leaf contour:

| t | B = t² | n | ns/leaf |
|---|---|---|---|
| 4 | 16 | 24 | **0.1118** |
| 16 | 256 | 20 | 0.0820 |
| 32 | 1024 | 18 | 0.0820 |
| 64 | 4096 | 16 | **0.0807** |
| 128 | 16384 | 14 | 0.0822 |

Spread at identical leaf count: **0.1118 / 0.0807 = 1.385, i.e. 38.5 % apart**.
L40S measured 0.267 / 0.192 = **39 %** at the same two endpoints. Same ratio,
same endpoints, ~2.4× faster in absolute terms.

That agreement is the point of the battery: the two-dimensional dependence is a
structural property of the kernel, not an L40S artifact, so indexing DPF cost by
leaf count alone is wrong on both machines.

Deep-and-narrow is the expensive corner throughout: at every leaf count the
`t=4, B=16` row is the slowest, because 16 trees cannot fill the device. The
surface flattens once `B ≥ 4096` — `t=64` and `t=128` agree to within 2 % at
matched leaf counts, i.e. the machine is saturated and only depth still matters.

---

## 4. Battery 2 — `ntt_stages.csv`: both backends, both bit-reversal conventions

90 rows. Gate 3 confirms the env flag took effect: all 45 `standalone` rows have
`brev_ms` ≈ 0.036–0.040, all 45 `fold` rows have exactly `0.0000`.

### fold vs standalone — the L40S regression does NOT reproduce

`fold_speedup = standalone / fold`; `> 1` means fold is faster.

| logN | b=4 | b=16 | b=64 |
|---|---|---|---|
| 10 | 1.300× | 1.293× | 1.271× |
| 14 | 1.274× | 1.273× | 1.246× |
| 18 | 1.225× | 1.165× | 1.153× |
| 20 | 1.181× | 1.155× | 1.155× |
| 22 | 1.126× | 1.116× | 1.113× |
| 23 | 1.119× | 1.113× | 1.112× |
| **24** | **1.009×** | **1.005×** | **1.004×** |

**On B200, fold is faster at every single point of the grid and never inverts.**
The advantage decays monotonically and converges to parity (~1.005×) at
logN 24, but does not become a penalty.

L40S reported fold **1.4× slower at logN 24**, attributed to bit-reversed
destination rows destroying DRAM row locality at that stride. B200's HBM3e does
not show that collapse — the stride costs the advantage, not the throughput.
Any model that carries the L40S logN-24 penalty forward to B200 is wrong.

### square vs naive — crossover at logN ≈ 19

`naive / square-fold`; `> 1` means the square backend wins.

| logN | b=4 | b=16 | b=64 |
|---|---|---|---|
| 16 | 0.461× | 0.689× | 0.966× |
| 18 | 0.731× | 1.011× | 1.235× |
| 19 | 0.929× | 1.225× | 1.367× |
| 20 | 1.136× | 1.390× | 1.482× |
| 22 | 1.360× | 1.433× | 1.451× |
| 24 | 1.361× | 1.375× | 1.385× |

Below logN ≈ 18 the textbook naive backend is *faster* — up to 2.2× at
logN 16 / batch 4 — because the four-step composition's fixed passes dominate
when there is little work. The square backend takes over from logN ≈ 19–20 and
settles at **1.36–1.48×**. Larger batch moves the crossover earlier.

---

## 5. Battery 3 — `dpf_grid.csv`: 30 security rows × 5 tiers

150 rows, **0 non-ok**. Each row at its own real `(n = logN+1−log₂t, B = t²)`.
At logN 24, `nvlink` tier:

| λ | c | t | n | B | leaves | total ms | ns/leaf | exchanges | exposed ms |
|---|---|---|---|---|---|---|---|---|---|
| 80 | 2 | 64 | 19 | 4096 | 2.15e9 | 159.279 | 0.0742 | 19 | 0.096 |
| 80 | 4 | 16 | 21 | 256 | 5.37e8 | 43.397 | 0.0808 | 21 | 0.106 |
| 80 | 8 | 4 | 23 | 16 | 1.34e8 | 15.557 | 0.1159 | 23 | 0.116 |
| 128 | 2 | 128 | 18 | 16384 | 4.29e9 | 326.864 | 0.0761 | 18 | 0.091 |
| 128 | 4 | 16 | 21 | 256 | 5.37e8 | 43.407 | 0.0809 | 21 | 0.106 |
| 128 | 8 | 8 | 22 | 64 | 2.68e8 | 24.599 | 0.0916 | 22 | 0.111 |

The ns/leaf column reads straight off the §3 surface — `c=8,t=4` is the
expensive row (0.1159) for the same reason it is on the surface: B = 16.

---

## 6. Battery 4 — `e2e_full.csv`: the true end-to-end, both arms

300 rows. Status tally: **247 OK / 40 OOM_SKIP / 13 CRASH**.

One process, one wall clock, at logN 24 / `dc` tier / `serial` arm:

| λ | c | t | leaves | expand | convert | beaver | ntt | **wall** |
|---|---|---|---|---|---|---|---|---|
| 80 | 2 | 64 | 8.59e9 | 315.4 | 333.5 | 0.40 | 130.4 | **779.6** |
| 80 | 4 | 16 | 8.59e9 | 327.8 | 383.5 | 1.60 | 622.2 | **1335.2** |
| 80 | 8 | 4 | 8.59e9 | 411.2 | 656.3 | 6.41 | 1988.7 | **3062.9** |
| 128 | 2 | 128 | 1.72e10 | 629.2 | 692.5 | 0.40 | 150.4 | **1472.6** |
| 128 | 4 | 16 | 8.59e9 | 327.8 | 384.2 | 1.60 | 523.6 | **1237.3** |
| 128 | 8 | 8 | 1.72e10 | 705.9 | 937.5 | 6.41 | 1988.2 | **3638.4** |

All times in ms. Beaver opens are negligible (0.4–6.4 ms, ≤ 0.2 % of wall). The
`c=8` rows are NTT-dominated (2c² = 128 poly_muls); the `c=2` rows are
expansion-dominated.

### Gate 5 FAIL — `batched` is not always faster on the expansion

All 150 cells have both arms. But **14 cells have `batched` expand > `serial`
expand**, against the package's expectation that batching never loses:

```
lam=80  c=4 t=16  logN=20 nvlink    serial=28.6704  batched=36.9647  (+28.93%)
lam=80  c=2 t=64  logN=23 loopback  serial=160.5676 batched=202.9065 (+26.37%)
lam=80  c=2 t=64  logN=22 nvlink    serial=82.1992  batched=87.2541  (+6.15%)
... 11 more between +0.22% and +4.12%
```

The +28.93 % entry is a **one-off transient, not a property**: for that same
cell the batched expand is 22.26 / 22.43 / 22.92 ms at pcie / loopback / dc and
only 36.96 at nvlink. A first-touch/allocator-warm-up explanation was tested
and refuted — only 1 of 12 comparable cells shows it, so it is not run-order.

The remaining 12 violations are concentrated in the two `c=2` rows (t=64 and
t=128) at +0.22 % to +6.15 %, which is consistent in magnitude across cells and
should **not** be dismissed as noise. **Open question: re-measure the `c=2` rows
before claiming the batched arm dominates on B200.**

Note that the batched arm also coalesces the per-level CW exchanges, so its
network column is not comparable to serial's on a like-for-like basis.

### OOM — expected, 40 cells

All in the `batched` arm at the large `(c, t, N)` corners, verbatim form:

```
RESULT c=2 t=128 logN=23 mode=batched STATUS=OOM_SKIP
```

The bench's own device-memory check works correctly. Nothing was substituted.

### CRASH — 13 cells, no error text exists

All at **λ=128, c=2, t=128, `batched`, logN 20–22**. Their logs are **0 bytes**
— the process emitted nothing at all (successful peers are 442–445 B), so
**there is no verbatim message to quote**. Evidence gathered:

- **Non-deterministic.** Of the 15 tier-instances in that range, logN 21/pcie
  and logN 22/loopback returned `STATUS=OK`; the other 13 died.
- The device-OOM path demonstrably works — logN 23/24 give clean `OOM_SKIP`.
- `MaxRSS` for the whole job was **24.5 GB against a 400 GB request**, so this
  was not a Slurm host-memory kill.

Cause is **not determinable** from the available evidence. Given the co-tenant
job (§1), these 15 cells should be treated as contaminated/indeterminate and
re-measured in isolation, not recorded as a reproducible property.

### Gate 6 FAIL — decomposed

- **(a) `wall_ms` non-decreasing in N: 0 violations.**
- **(b) `ms_per_mul` non-decreasing in N: 8 violations**, all at logN 10–15,
  all ≤ 3.78 % — the launch-bound plateau.
- **(c) `wall_ms` increasing with tier RTT: 47 violations.** This gate cannot
  resolve what it is asking on B200. For `c=4,t=16,logN=20,serial`
  (272 exchanges) the injected RTT term is:

  | tier | rtt | RTT term | measured wall |
  |---|---|---|---|
  | nvlink | 5 µs | 1.36 ms | 95.62 |
  | pcie | 16 µs | 4.35 ms | 98.99 |
  | loopback | 25.2 µs | 6.85 ms | 121.33 |
  | dc | 50 µs | 13.60 ms | 109.45 |
  | wan | 2000 µs | **544.00 ms** | 702.22 |

  Between pcie and loopback the injected difference is 2.5 ms while the measured
  walls differ by 22.3 ms. The four low tiers are separated by 1.4–13.6 ms on
  walls of 100–3700 ms, i.e. at or below run-to-run variation.
  **Decisive: none of the 47 violations involves the `wan` tier.** Where the RTT
  step is large (dc → wan, +530 ms) the ordering holds in 100 % of cells. The
  gate is only meaningful across that one step at these wall times.

---

## 7. Contention check — 12 cells outside the band, but not from contention

The package's check (`ms_per_mul / (sum_ms/batch)` ∈ [0.75, 1.35] for
logN ≥ 16) flags **12 of 54** square rows, worst `ratio = 2.018` at
logN 16 / batch 4 / fold. They are marked, as required — but the cause is a
**constant per-call overhead that `sum_ms` does not account for**, not the
neighbouring job:

```
implied overhead = batch × ms_per_mul − sum_ms
                 = 0.2035 … 0.2502 ms across the whole grid (≈0.22 ms ±10%)
while sum_ms itself ranges 0.17 … 6.46 ms (a 40× span)
```

Three independent signatures rule out contention:

1. the ratio decreases **monotonically** with logN, converging to 1.001;
2. it is strictly ordered by batch — b=4 worst, b=64 never out of band;
3. `standalone` and `fold` are **two independent runs**, yet their ratios agree
   to within 1–4 % at all 27 points. Contention is time-varying and cannot
   reproduce like that.

| logN | b=4 | b=16 | b=64 |
|---|---|---|---|
| 16 | 1.980 | 1.653 | 1.250 |
| 18 | 1.598 | 1.235 | 1.074 |
| 20 | 1.242 | 1.073 | 1.019 |
| 22 | 1.068 | 1.018 | 1.005 |
| 24 | 1.017 | 1.004 | 1.001 |

**The [0.75, 1.35] @ logN ≥ 16 threshold is calibrated for L40S.** B200 kernels
are much faster, so the same fixed ~0.22 ms stays significant longer; the band
only becomes valid from **logN ≈ 20** at batch 4. Recalibrate before using this
check on B200, or subtract the fixed term first.

---

## 8. Open items

1. **Re-measure the 15 λ=128/c=2/t=128/batched cells** at logN 20–22 in
   isolation, to separate the 13 zero-byte CRASHes from co-tenant interference.
2. **Re-measure the two `c=2` rows** for gate 5 — 12 small but consistent
   batched-slower-than-serial violations remain unexplained.
3. **Recalibrate the contention band** for B200 (§7), or subtract the ~0.22 ms
   fixed per-call term before applying it.
4. An `--exclusive` allocation was attempted and **could not be scheduled** on
   this cluster (≈630 queued jobs, all 28 `b200-batch` nodes permanently
   `mixed`, so no node ever fully drains). Hand-pinning a quiet node with `-w`
   pushed the estimated start out by >2 h. Everything here was therefore run on
   a shared node with the co-tenancy recorded rather than assumed away.

---

## 9. File map — `pcg_baseline_out/`

| File | Rows / size | Note |
|---|---|---|
| `dpf_shape.csv` | 82 | ns/leaf(n, B) surface; 0 non-ok |
| `ntt_stages.csv` | 90 | both backends × both brev conventions |
| `dpf_grid.csv` | 150 | 30 security rows × 5 tiers; 0 non-ok |
| `e2e_full.csv` | 300 | true e2e, serial + batched; 247 OK / 40 OOM_SKIP / 13 CRASH |
| `square_verify.log` | 11 | ends `GATE PASS` |
| `square_verify_fold.log` | 11 | ends `GATE PASS` |
| `topo.txt` | — | single GPU; no NV# pair present |
| `gpu_before.txt` / `gpu_after.txt` | — | header only; see §1 for why that is not "clean" |
| `*.log` | 669 files, 490 KB | one per cell |

Reproduce the verdicts in §2 with `python3 check_gates_v2.py pcg_baseline_out`.
The Slurm submission used is `job_baseline_v2.sbatch`.

### Superseded — `pcg_baseline_out_235933_prev/`

Output of the **previous** HEAD (`213383e`), kept only for traceability.
**Do not use its numbers.** That runner passed `--B $((c*c))` where the protocol
requires `t²` (`CPU_baseline/pcg_ole_2pc/pcg_ole_impl.h:114-116` loops
`ii<c, jj<c` with `const int B = t*t` inside, and allocates `g_dev(c*c)`), which
(a) made `λ=80,c=8,t=4` do out-of-bounds writes in `dpf_out_scatter_g` —
confirmed under `compute-sanitizer` — and (b) understated the DPF workload by up
to 256×, which is why that run finished in 16 min. Both are fixed at `45b5f8e`.
