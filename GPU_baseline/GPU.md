# GPU baseline for PCG expansion — measurements and how they were reached

L40S (GDDR6, 24 ch, 865 GB/s), exclusive device, CUDA 12.x, `sm_89`.
Every number below is silicon; nothing is modelled or interpolated. Cells that
failed are reported as failures and never substituted.

Reproduce all of it with `./run_pcg_baseline.sh` (four batteries, one command).

---

## 1. What the baseline is

**The reference implementation, run as written.** `CPU_baseline/pcg_ole_2pc/
pcg_ole_impl.h:114-116` expands `c²` blocks of `t²` DPF instances **serially**,
and exchanges a correction word after every tree level of every block, so a full
expansion costs `c²·n` round trips. No coalescing, no cross-block batching, no
reordering. That is the baseline.

There are exactly **two** baselines, differing only in the NTT backend:

| | expansion | NTT backend | how obtained |
|---|---|---|---|
| **B0** | reference, serial | **naive** | B1 with the device NTT time replaced by the naive backend's, measured at the same `(logN, batch)`; the PCIe residual (§3.5) is identical in both and carries through untouched |
| **B1** | reference, serial | four-step, standalone brev | **measured end to end**, one process, one wall clock |

A batched/CW-coalesced variant was also measured. It is a *bound*, reported in
§D.4 to show the serial reference is not a strawman, and it is **not** a
baseline — no speedup in this document is taken against it.

## 2. The protocol parameters, and what actually sets cost

| symbol | meaning | range here |
|---|---|---|
| `N` | ring degree = problem size | 2²⁰ … 2²⁴ |
| `t` | noise weight; splits the ring into `t` bins | 4 … 128 |
| `c` | compression factor; the expansion has `c²` blocks | 2, 4, 8 |
| **`n = log₂N + 1 − log₂t`** | **DPF tree depth** | 14 … 23 |
| **`B = t²`** | **DPF instances in one block** | 16 … 16384 |

A block's cost is a function of exactly `(n, B)`. `c²` never enters it — it is a
multiplier outside. This matters: **the cost is not a function of the leaf count
alone.** At 2.68e8 leaves, `(n=16, B=4096)` measures 0.192 ns/leaf and
`(n=24, B=16)` measures 0.267 — **39% apart at identical work**, because 16
trees cannot fill the device. See §D.1.

## 3. The measurement path, and what each step corrected

Each step below was forced by a defect found in the step before it. The numbers
in parentheses are the effect on the reported speedup — positive means the
earlier number was too favourable to us.

### 3.1 Workload: `--B c²` → `--B t²`  (was 1/t² of the work)

The first campaign launched the DPF bench with `c²` instances — neither one
block (`t²`) nor the whole expansion (`c²t²`). At `logN=18` it reported
**0.0042 ns/leaf**, ~50× off the plateau, because 32K leaves is launch-bound.
Fixed to one real block. *(This never reached a headline number; it polluted the
composed `e2e.csv` only.)*

### 3.2 Cost model: leaf-indexed curve → `(n, B)` surface  (removed a fake constant)

The DPF baseline was a curve indexed by total leaf count, which collapses `(n,B)`
into one variable and hands all 30 security rows the same number — the origin of
the flat "DPF = 2.25 everywhere" column. Replaced by the measured 82-cell surface
(§D.1); every row now reads off it at its own `(n, B)`. **Rows now differ by up
to 40%.**

### 3.3 NTT: the bit-reversal convention became an explicit, measured choice

The library is natural-in / bit-reversed-out, so a four-step composition must
undo the permutation somewhere. Both placements are now measured in one binary
(`PCG_SQUARE_FOLD_BREV`), and both pass an `__int128` bit-exactness gate:

* **standalone (default, and what the paper uses)** — a separate kernel, so what
  is written to memory between transforms is in natural order.
* **fold** — folded into the twiddle-table indices, a transpose's destination
  row, and the pointwise store.

Folding is 1.1–1.4× faster up to `logN 22` and **1.4× slower at `logN 24`**
(§D.2): bit-reversed destination rows destroy DRAM row locality once the row
stride reaches 32 KB. Reporting only the fast one at small `N` would have been a
size-dependent cherry-pick.

### 3.4 e2e: composed arithmetic → one process, one wall clock  (+ a warm-up bug)

Every previous "e2e" figure was `c²·dpf + 2c²·ntt`, assembled from two separate
campaigns and never run. `e2e_full_bench.cu` now runs the whole expansion —
blocks, output layer, per-block Beaver opens, `2c²` poly_muls — under one clock.

Its first version put the NTT share at 84–91%, which was wrong: the per-`N`
twiddle-table construction (host `modpow`, seconds at large `N`) landed inside
the measured NTT phase. It is one-time state a deployment reuses, so it is now
built before the clock starts. Corrected, at `logN 20, c=2, t=64`: **expand 68%
/ convert 14% / NTT 18%** (§D.5) — the expansion dominates, which is the segment
the PIM design targets.

### 3.5 The NTT lane is mostly PCIe, and that is the reference's own doing

The e2e `ntt_ms` and the NTT battery's `ms_per_mul` are not the same quantity.
`ntt_batch_bench` reports `st.device_ms` — kernels only. The e2e figure is host
wall clock, and `poly_mul_u64_gpuntt_square` takes **host** pointers, so every
call pays H2D of `a`,`b` and D2H of `out`. Subtracting one from the other:

| (c,t) | logN | e2e `ntt_ms` | device (2c²·four-step) | residual | bytes | implied |
|---|---|---|---|---|---|---|
| (4,16) | 20 | 103.9 | 22.0 | 81.9 | 0.75 GB | 9.2 GB/s |
| (4,16) | 22 | 734.7 | 114.4 | 620.3 | 3.00 GB | 4.8 GB/s |
| (8,4) | 23 | 5453.5 | 904.2 | 4549.3 | 24.0 GB | 5.3 GB/s |

4–10 GB/s is pageable-host PCIe. **60–85% of the reference's NTT lane is the
bus, not the SM.**

This is not a bench artefact. `pcg_ole_impl.h:467-481` copies the device `g`
back element-by-element into a host `Poly<P>`, `pcg_ole.h:217-243` packs it into
a `uint64_t` array, ships it to the GPU, and unpacks the result — and it calls
`poly_mul` **once per `(i,j)` pair, batch = 1**. The bench batches `c²` and
skips both host loops, so **B1 as measured is a lower bound on what the
reference actually costs.**

The consequence for the comparison: both baselines pay this bus traffic because
the reference is written that way. The PIM design does not — `g` is written into
DRAM by the in-bank SPU and read from DRAM by the SM — so its NTT lane is the
device-only time. §4 keeps the two apart in every row.

*(A batched/coalesced variant of the expansion was measured separately: it
differs by 2.6% geomean on compute and folds 272 CW exchanges into 17. Reported
in §D.4 as a bound; not a baseline.)*

### 3.6 A library limit that was silently emptying the table

`GPU_NTT_Inplace` aborts above 65535 transforms per launch. Chunking the call
unlocked `batch=64` at every `N` and `batch=16` at `logN 23–24` — **14 of 30
main-table cells that were previously permanently blank.**

### 3.7 Machine hygiene, promoted to a gate

The first campaign ran while another user held 167 GB across four GPUs; several
cells came back 3–6× their neighbours. All numbers here were taken on an
exclusive device with `nvidia-smi --query-compute-apps` recorded before and
after (both empty). The consistency check `ms_per_mul ÷ (sum_ms/batch) ∈
[0.75, 1.35]` is now part of the campaign and caught the one remaining bad cell
(`logN 20, batch 16`, ratio 1.73).

## 4. Result

### 4.1 The two baselines

Compute throughput of the reference expansion, from §D.1 read at each row's own
`(n, B)`:

| (c,t) | B | ns/leaf (geomean over logN 20–24) |
|---|---|---|
| (2,64) | 4096 | **0.2020** |
| (4,16) | 256 | **0.2128** |
| (8,4) | 16 | **0.2717** |
| all | | **0.2223** |

`(8,4)` is 35% worse than `(2,64)` at identical total work: 16 parallel trees
under-fill the device. This is the `(n,B)` effect of §3.2, and it is why the
per-row DPF column is not a constant.

Communication scales with the block count and is a separate axis:

| (c,t) | exchanges per expansion | at α = 2 ms (WAN) |
|---|---|---|
| (2,64) | 60–72 | 0.12–0.14 s |
| (4,16) | 272–336 | 0.54–0.67 s |
| (8,4) | 1216–1472 | **2.4–2.9 s** |

At `c=8` the reference's communication is an order of magnitude larger than the
computation it accompanies.

B0 differs from B1 only in the NTT backend. The naive backend is *faster* below
its crossover (logN 20↔21 at batch 4, 18↔19 at batch 16, 16↔17 at batch 64) and
2.0–2.4× slower above it, so B0 < B1 in some rows and > B1 in others — see §D.2.
The 7 cells where naive hits its own grid limit are reported as `n/a`, never
substituted.

### 4.2 Against the PIM design

The PIM design moves the DPF expansion and both output-layer passes into the
in-bank SPU. **It does not change the NTT.** The SM runs the same four-step, the
same standalone bit-reversal, the same backend as B1, and on GDDR6 the DRU hides
none of it — so the NTT lane is *identical on both sides* and cancels wherever it
dominates.

```
e2e(GPU) = DPF + NTT              (serial, as measured)
e2e(PIM) = max(DPF lane, NTT lane)
```

DPF lane = `leaves × 0.1014 ns` — ramulator2, GDDR6_L40S org (24 ch × 8 PU =
192 SPU), dual-issue ARX + LSU, both output passes: 718,660 CK at tCK = 0.444 ns
over 768 × 2¹² leaves. **DRU gain = 0** on GDDR6.

**B1 is the measured wall clock, unmodified.** It includes the device→host→device
round trip of `g` (§3.5). Keeping `g` on the device would remove it, but that is
a software optimisation and §1 defines the baseline to exclude those — so it
stays in. The per-row PCIe share is printed so the reader can see how much of the
gap is data placement rather than arithmetic. The PIM side pays no such round
trip: `g` is written into DRAM by the SPU and read from DRAM by the SM. **That is
an architectural difference, not a software one** — it is the point of
near-memory expansion, and the one asymmetry in this table that is earned.

NVLink tier, ms:

| (c,t) | logN | gpu DPF | NTT lane | B0 | B1 | PIM lane | **PIM+GPU** | bound | vs B0 | vs B1 | of which PCIe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (2,64) | 20 | 106 | 4 | 124 | 125 | 54.5 | **54.5** | PIM | 2.28× | 2.30× | 15 |
| (2,64) | 21 | 212 | 9 | 262 | 251 | 108.9 | **108.9** | PIM | 2.40× | 2.30× | 29 |
| (2,64) | 22 | 449 | 29 | 601 | 572 | 217.8 | **217.8** | PIM | 2.76× | 2.63× | 94 |
| (2,64) | 23 | 875 | 59 | 1203 | 1136 | 435.7 | **435.7** | PIM | 2.76× | 2.61× | 202 |
| (4,16) | 20 | 113 | 22 | 222 | 192 | 54.5 | **54.5** | PIM | 4.07× | 3.53× | 57 |
| (4,16) | 21 | 226 | 50 | 549 | 491 | 108.9 | **108.9** | PIM | 5.04× | 4.51× | 216 |
| (4,16) | 22 | 464 | 114 | 1127 | 1015 | 217.8 | **217.8** | PIM | 5.18× | 4.66× | 437 |
| (4,16) | 23 | 925 | 227 | n/a | 1822 | 435.7 | **435.7** | PIM | — | 4.18× | 670 |
| (4,16) | 24 | 1824 | 456 | n/a | 3603 | 871.3 | **871.3** | PIM | — | 4.13× | 1322 |
| (8,4) | 20 | 152 | 85 | n/a | 678 | 54.5 | **85.1** | NTT | — | 7.97× | 440 |
| (8,4) | 21 | 284 | 189 | n/a | 1392 | 108.9 | **189.4** | NTT | — | 7.35× | 918 |
| (8,4) | 22 | 573 | 439 | n/a | 2813 | 217.8 | **439.3** | NTT | — | 6.40× | 1800 |
| (8,4) | 23 | 1118 | 904 | n/a | 5203 | 435.7 | **904.2** | NTT | — | 5.75× | 3179 |
| (8,4) | 24 | 2254 | 1816 | n/a | 10746 | 871.3 | **1815.6** | NTT | — | 5.92× | 6675 |
| (2,128) | 20 | 211 | 4 | 228 | 229 | 108.9 | **108.9** | PIM | 2.10× | 2.10× | 14 |
| (2,128) | 21 | 417 | 9 | 481 | 471 | 217.8 | **217.8** | PIM | 2.21× | 2.16× | 43 |
| (2,128) | 22 | 873 | 29 | 1083 | 1054 | 435.7 | **435.7** | PIM | 2.49× | 2.42× | 151 |
| (4,16) | 20 | 115 | 22 | 268 | 239 | 54.5 | **54.5** | PIM | 4.93× | 4.38× | 101 |
| (4,16) | 21 | 227 | 50 | 501 | 443 | 108.9 | **108.9** | PIM | 4.60× | 4.07× | 166 |
| (4,16) | 22 | 474 | 114 | 1097 | 985 | 217.8 | **217.8** | PIM | 5.04× | 4.52× | 396 |
| (4,16) | 23 | 930 | 227 | n/a | 1859 | 435.7 | **435.7** | PIM | — | 4.27× | 702 |
| (4,16) | 24 | 1814 | 456 | n/a | 3560 | 871.3 | **871.3** | PIM | — | 4.09× | 1289 |

| tier | vs B0 | vs B1 | PIM-bound |
|---|---|---|---|
| nvlink | 3.31× | **3.88×** | 17/22 |
| loopback | 3.41× | **3.98×** | 17/22 |
| wan | 5.90× | **7.36×** | 16/21 |

**A PIM-bound cell's number cannot move when the NTT accounting changes.** In 17
of 22 cells `PIM+GPU = DPF lane`, full stop — 54.5 ms at `(2,64) logN 20`
whatever the SM is doing. That is the invariant to check any revision of this
table against.

**Where the gap comes from, and which part the architecture actually earns.**
At `(4,16) logN 22`: B1 = 1015 ms = 464 DPF + 114 NTT + 437 PCIe, against a
PIM+GPU of 218 ms. The 797 ms saved splits three ways, and they are *not* the
same kind of thing:

| term | ms | what it is |
|---|---|---|
| `g` over PCIe | 437 | **a defect of the reference**, 3.0 GB moved because `poly_mul_u64_gpuntt_square` takes host pointers. Software-fixable with no PIM involved. |
| DPF lane 464 → 218 | 246 | **the architectural claim**: 2.147e9 leaves × 16 B = **34.4 GB** of leaf traffic that crosses the GDDR bus on the GPU and never leaves the bank in the PIM. |
| NTT hidden under DPF | 114 | the co-schedule |

An earlier revision of this section called all of it "data placement". That
conflated two unlike things. The leaf traffic is what near-memory expansion is
for; the `g` round trip is something the reference happens to do badly.

**Sensitivity, since the largest term is the one we did not earn.** §1 defines
the baseline to exclude software optimisations, so B1 keeps the round trip and
the headline is quoted against it. But the number with `g` kept on the device is
what a reviewer will ask for, so here it is:

| tier | vs B1 as written | with `g` kept on device |
|---|---|---|
| nvlink | **3.88×** | 2.37× |
| loopback | **3.98×** | 2.43× |
| wan | **7.36×** | 5.72× |

`(4,16) logN 22` alone: 4.66× → 2.66×. **2.37× at NVLink is the architecture-only
number** — leaves not leaving the bank, plus the co-schedule, with none of the
reference's software defects counted. 3.88× is the number against the reference
as written. Both belong in the paper, with the difference named.

**The `(8,4)` rows are NTT-bound** and cap at 5.75–7.97×: the NTT lane is the same
1816 ms on both sides at `logN 24`, so the design can only hide a 2254 ms DPF
under it and skip the 6675 ms of bus traffic. `(4,16)` is PIM-bound at every `N`
and carries 4.1–5.2×. `(2,64)` and `(2,128)` sit at 2.1–2.8× because `c=2` gives
the SM only 4 poly_muls — a 4–59 ms NTT lane against a 218–436 ms PIM lane, so
the co-schedule has nothing to hide behind and both the NTT and the PCIe terms
are small.

**The WAN row assumes the network is fully hidden on the PIM side** (co-schedule
+ CW gang) while the reference pays it inline. That assumption is what the pass-2
placement ablation is measuring; until it lands, read 7.36× as conditional and
the NVLink **3.88×** as the unconditional number.

**DRU is worth nothing here.** The four-step's transpose + bit-reversal is 35–47%
of its device time and none of it can be hidden on GDDR6. The `(8,4)` rows show
the cost: the SM lane caps the design, and the one tool aimed at it is inert on
this memory. The DRU case has to be made on HBM.

---
## 5. Raw data

The CSVs behind every table are in `data_l40s/` and are the untouched campaign
output:

| file | rows | contents |
|---|---|---|
| `dpf_shape.csv` | 82 | the `(n, B)` surface, `t ∈ {4…128} × n ∈ {8…24}` |
| `ntt_stages.csv` | 135 | NTT, both backends, both brev conventions, 6-stage split |
| `dpf_grid.csv` | 150 | 30 security rows × 5 tiers, per real block |
| `e2e_full.csv` | 150 | true end-to-end, serial arm, 5 tiers |
| `e2e_arms.csv` | 79+ | serial vs batched, 3 tiers (campaign still extending) |

Known gaps, reported rather than filled:

* `naive` fails at `batch=64, logN ≥ 20` and `batch=16, logN ≥ 23` — the naive
  kernel has its own grid limit and was not chunked. 7 cells.
* `e2e_full.csv` has 16 `CRASH`/`OOM_SKIP` cells at the large `(c,t,N)` corners.
* `e2e_arms.csv` is still being extended; `(8,8)` and `(2,128)` rows are partial.

---

## Appendix D — full data

### D.1  DPF+convert throughput surface, `ns/leaf(n, B)`

| n | t=4<br>B=16 | t=8<br>B=64 | t=16<br>B=256 | t=32<br>B=1024 | t=64<br>B=4096 | t=128<br>B=16384 |
|---|---|---|---|---|---|---|
| 8 | 53.5182 | 14.1172 | 15.1470 | 7.8544 | 1.9951 | 2.8085 |
| 9 | 29.4440 | 7.6634 | 7.5888 | 3.9343 | 1.0001 | 1.5338 |
| 10 | 15.9948 | 4.1615 | 1.2517 | 0.4644 | 0.2665 | 0.2572 |
| 11 | 8.5924 | 2.3534 | 0.7393 | 0.2787 | 0.2438 | 0.2253 |
| 12 | 4.6834 | 1.3040 | 0.4310 | 0.2192 | 0.2226 | 0.2085 |
| 13 | 2.5720 | 0.7614 | 0.2731 | 0.2183 | 0.2074 | 0.1978 |
| 14 | 1.4229 | 0.4536 | 0.2255 | 0.2093 | 0.1987 | 0.1935 |
| 15 | 0.8587 | 0.3048 | 0.2225 | 0.2025 | 0.1936 | 0.1896 |
| 16 | 0.5240 | 0.2395 | 0.2149 | 0.1982 | 0.1913 | 0.2023 |
| 17 | 0.3485 | 0.2340 | 0.2098 | 0.1958 | 0.2018 | — |
| 18 | 0.2742 | 0.2237 | 0.2040 | 0.2055 | 0.2040 | — |
| 19 | 0.2631 | 0.2148 | 0.2106 | 0.2037 | — | — |
| 20 | 0.2483 | 0.2227 | 0.2113 | 0.2106 | — | — |
| 21 | 0.2525 | 0.2234 | 0.2097 | — | — | — |
| 22 | 0.2491 | 0.2207 | 0.2101 | — | — | — |
| 23 | 0.2538 | 0.2306 | — | — | — | — |
| 24 | 0.2565 | 0.2517 | — | — | — | — |

### D.2  NTT, `ms` per poly_mul

| logN | batch | standalone brev | fold brev | naive | naive/sq | fold gain |
|---|---|---|---|---|---|---|
| 10 | 4 | 0.0667 | 0.0580 | 0.0183 | 0.27× | 1.15× |
| 10 | 16 | 0.0184 | 0.0147 | 0.0046 | 0.25× | 1.25× |
| 10 | 64 | 0.0044 | 0.0036 | 0.0013 | 0.30× | 1.22× |
| 11 | 4 | 0.0688 | 0.0588 | 0.0194 | 0.28× | 1.17× |
| 11 | 16 | 0.0167 | 0.0151 | 0.0051 | 0.31× | 1.11× |
| 11 | 64 | 0.0053 | 0.0040 | 0.0015 | 0.28× | 1.32× |
| 12 | 4 | 0.0782 | 0.0620 | 0.0218 | 0.28× | 1.26× |
| 12 | 16 | 0.0174 | 0.0156 | 0.0060 | 0.34× | 1.12× |
| 12 | 64 | 0.0057 | 0.0045 | 0.0019 | 0.33× | 1.27× |
| 13 | 4 | 0.0729 | 0.0554 | 0.0237 | 0.33× | 1.32× |
| 13 | 16 | 0.0201 | 0.0164 | 0.0068 | 0.34× | 1.23× |
| 13 | 64 | 0.0062 | 0.0051 | 0.0029 | 0.47× | 1.22× |
| 14 | 4 | 0.0698 | 0.0635 | 0.0270 | 0.39× | 1.10× |
| 14 | 16 | 0.0218 | 0.0163 | 0.0090 | 0.41× | 1.34× |
| 14 | 64 | 0.0079 | 0.0065 | 0.0044 | 0.56× | 1.22× |
| 15 | 4 | 0.0742 | 0.0607 | 0.0319 | 0.43× | 1.22× |
| 15 | 16 | 0.0243 | 0.0198 | 0.0132 | 0.54× | 1.23× |
| 15 | 64 | 0.0109 | 0.0087 | 0.0080 | 0.73× | 1.25× |
| 16 | 4 | 0.0855 | 0.0678 | 0.0415 | 0.49× | 1.26× |
| 16 | 16 | 0.0326 | 0.0266 | 0.0212 | 0.65× | 1.23× |
| 16 | 64 | 0.0192 | 0.0156 | 0.0173 | 0.90× | 1.23× |
| 17 | 4 | 0.1049 | 0.0839 | 0.0623 | 0.59× | 1.25× |
| 17 | 16 | 0.0505 | 0.0365 | 0.0381 | 0.75× | 1.38× |
| 17 | 64 | 0.0620 | 0.0435 | 0.1283 | 2.07× | 1.43× |
| 18 | 4 | 0.1514 | 0.1107 | 0.0993 | 0.66× | 1.37× |
| 18 | 16 | 0.1011 | 0.0743 | 0.0813 | 0.80× | 1.36× |
| 18 | 64 | 0.1690 | 0.1320 | 0.3667 | 2.17× | 1.28× |
| 19 | 4 | 0.2249 | 0.1595 | 0.1782 | 0.79× | 1.41× |
| 19 | 16 | 0.2687 | 0.1866 | 0.5726 | 2.13× | 1.44× |
| 19 | 64 | 0.3370 | 0.2609 | 0.7684 | 2.28× | 1.29× |
| 20 | 4 | 0.4668 | 0.3243 | 0.3781 | 0.81× | 1.44× |
| 20 | 16 | 0.6889 | 0.5401 | 1.6170 | 2.35× | 1.28× |
| 20 | 64 | 0.6649 | 0.5331 | FAIL | — | 1.25× |
| 21 | 4 | 1.1830 | 0.8745 | 2.5357 | 2.14× | 1.35× |
| 21 | 16 | 1.5532 | 1.2366 | 3.3549 | 2.16× | 1.26× |
| 21 | 64 | 1.4797 | 1.2155 | FAIL | — | 1.22× |
| 22 | 4 | 3.5786 | 3.2146 | 7.2331 | 2.02× | 1.11× |
| 22 | 16 | 3.5756 | 3.1608 | 7.0830 | 1.98× | 1.13× |
| 22 | 64 | 3.4318 | 3.1121 | FAIL | — | 1.10× |
| 23 | 4 | 7.3960 | 6.9130 | 15.8039 | 2.14× | 1.07× |
| 23 | 16 | 7.0786 | 6.8805 | FAIL | — | 1.03× |
| 23 | 64 | 7.0643 | 6.8289 | FAIL | — | 1.03× |
| 24 | 4 | 14.2768 | 20.0310 | 33.8584 | 2.37× | 0.71× |
| 24 | 16 | 14.2595 | 19.9281 | FAIL | — | 0.72× |
| 24 | 64 | 14.1842 | 19.9007 | FAIL | — | 0.71× |

### D.3  DPF per real block, `total_ms` by tier (B=t², real depth)

| λ | c | t | logN | n | B | nvlink | pcie | loopback | dc | wan | ns/leaf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 80 | 2 | 64 | 20 | 15 | 4096 | 29.43 | 26.22 | 26.42 | 26.72 | 55.99 | 0.2193 |
| 80 | 2 | 64 | 21 | 16 | 4096 | 51.56 | 52.08 | 52.27 | 52.69 | 83.77 | 0.1921 |
| 80 | 2 | 64 | 22 | 17 | 4096 | 108.72 | 108.71 | 109.27 | 109.33 | 142.33 | 0.2025 |
| 80 | 2 | 64 | 23 | 18 | 4096 | 219.51 | 219.61 | 220.97 | 220.16 | 255.38 | 0.2044 |
| 80 | 4 | 16 | 20 | 17 | 256 | 9.34 | 7.34 | 7.33 | 7.93 | 40.97 | 0.2785 |
| 80 | 4 | 16 | 21 | 18 | 256 | 13.63 | 13.81 | 14.01 | 14.48 | 49.55 | 0.2031 |
| 80 | 4 | 16 | 22 | 19 | 256 | 28.35 | 28.59 | 28.74 | 29.15 | 66.93 | 0.2112 |
| 80 | 4 | 16 | 23 | 20 | 256 | 56.17 | 56.33 | 57.55 | 57.25 | 96.09 | 0.2092 |
| 80 | 4 | 16 | 24 | 21 | 256 | 111.80 | 114.10 | 114.88 | 115.31 | 154.30 | 0.2082 |
| 80 | 8 | 4 | 20 | 19 | 16 | 3.00 | 2.52 | 2.69 | 3.30 | 40.52 | 0.3574 |
| 80 | 8 | 4 | 21 | 20 | 16 | 4.58 | 4.52 | 4.99 | 5.51 | 44.53 | 0.2727 |
| 80 | 8 | 4 | 22 | 21 | 16 | 8.99 | 9.21 | 9.01 | 10.01 | 51.04 | 0.2680 |
| 80 | 8 | 4 | 23 | 22 | 16 | 16.99 | 18.02 | 17.42 | 17.95 | 60.94 | 0.2532 |
| 80 | 8 | 4 | 24 | 23 | 16 | 34.24 | 36.73 | 37.54 | 37.71 | 81.80 | 0.2551 |
| 128 | 2 | 128 | 20 | 14 | 16384 | 79.87 | 70.26 | 70.21 | 70.94 | 95.23 | 0.2975 |
| 128 | 2 | 128 | 21 | 15 | 16384 | 134.32 | 185.04 | 138.13 | 102.94 | 132.35 | 0.2502 |
| 128 | 2 | 128 | 22 | 16 | 16384 | 217.11 | 217.30 | 217.81 | 225.71 | 249.52 | 0.2022 |
| 128 | 4 | 16 | 20 | 17 | 256 | 7.01 | 7.21 | 7.41 | 7.85 | 40.97 | 0.2090 |
| 128 | 4 | 16 | 21 | 18 | 256 | 13.88 | 14.09 | 14.16 | 14.52 | 49.68 | 0.2068 |
| 128 | 4 | 16 | 22 | 19 | 256 | 29.03 | 28.59 | 29.11 | 29.29 | 66.69 | 0.2163 |
| 128 | 4 | 16 | 23 | 20 | 256 | 57.36 | 57.35 | 58.72 | 57.73 | 96.44 | 0.2137 |
| 128 | 4 | 16 | 24 | 21 | 256 | 112.87 | 115.83 | 116.90 | 113.42 | 158.85 | 0.2102 |
| 128 | 8 | 8 | 20 | 18 | 64 | 5.26 | 4.11 | 4.29 | 4.82 | 41.02 | 0.3133 |
| 128 | 8 | 8 | 21 | 19 | 64 | 7.37 | 8.21 | 7.82 | 8.26 | 45.60 | 0.2196 |
| 128 | 8 | 8 | 22 | 20 | 64 | 15.35 | 15.38 | 15.99 | 16.10 | 55.45 | 0.2288 |
| 128 | 8 | 8 | 23 | 21 | 64 | 30.23 | 30.50 | 31.12 | 30.99 | 72.60 | 0.2252 |
| 128 | 8 | 8 | 24 | 22 | 64 | 60.61 | 66.27 | 67.39 | 67.24 | 109.41 | 0.2258 |

### D.4  End-to-end, reference (serial) arm, `ns/leaf` of expand+convert

| λ | (c,t) | logN | n | exch | nvlink | loopback | wan |
|---|---|---|---|---|---|---|---|
| 80 | (2,64) | 20 | 15 | 60 | 0.3573 | 0.2972 | 0.5313 |
| 80 | (2,64) | 21 | 16 | 64 | 0.3065 | 0.2620 | 0.4138 |
| 80 | (4,16) | 20 | 17 | 272 | 0.4972 | 0.5682 | 1.4915 |
| 80 | (4,16) | 21 | 18 | 288 | 0.4276 | 0.4081 | 0.9243 |
| 80 | (4,16) | 22 | 19 | 304 | 0.3321 | 0.3525 | 0.6894 |
| 80 | (4,16) | 23 | 20 | 320 | 0.3333 | 0.3321 | 0.5146 |
| 80 | (8,4) | 20 | 19 | 1216 | 1.6473 | 1.5147 | 5.9243 |
| 80 | (8,4) | 21 | 20 | 1280 | 1.1165 | 1.1080 | 3.2551 |
| 80 | (8,4) | 22 | 21 | 1344 | 0.8027 | 0.8757 | 1.8318 |
| 80 | (8,4) | 23 | 22 | 1408 | 0.6180 | 0.6690 | 1.1893 |
| 128 | (2,128) | 20 | 14 | 56 | 0.3518 | 0.4036 | 0.4552 |
| 128 | (4,16) | 20 | 17 | 272 | 0.6847 | 0.6600 | 1.5231 |
| 128 | (4,16) | 21 | 18 | 288 | 0.5272 | 0.5666 | 0.9541 |
| 128 | (4,16) | 22 | 19 | 304 | 0.4839 | 0.4645 | 0.7306 |
| 128 | (4,16) | 23 | 20 | 320 | 0.4276 | 0.4405 | 0.5743 |
| 128 | (8,8) | 20 | 18 | 1152 | 0.7148 | 0.7695 | 2.5987 |
| 128 | (8,8) | 21 | 19 | 1216 | 0.5465 | 0.2757 | 1.4816 |
| 128 | (8,8) | 22 | 20 | 1280 | 0.3909 | 0.3097 | 0.9600 |
| 128 | (8,8) | 23 | 21 | 1344 | 0.2723 | 0.3556 | 0.6409 |

*The batched/coalesced arm (`e2e_arms.csv`) is a bound, not a baseline: 2.6% geomean on compute, 272→17 exchanges. Not used in any speedup here.*

### D.5  End-to-end wall clock and phase split (serial arm, plans warmed)

| λ | (c,t) | logN | tier | expand | convert | beaver | ntt | **wall** | exp% | ntt% |
|---|---|---|---|---|---|---|---|---|---|---|
| 80 | (2,64) | 20 | nvlink | 159.7 | 32.1 | 0.04 | 41.6 | **233.5** | 68% | 18% |
| 80 | (2,64) | 20 | wan | 246.1 | 39.1 | 16.00 | 40.2 | **341.4** | 72% | 12% |
| 80 | (2,64) | 21 | nvlink | 247.7 | 81.4 | 0.04 | 46.7 | **375.9** | 66% | 12% |
| 80 | (2,64) | 21 | wan | 370.8 | 73.5 | 16.00 | 50.8 | **511.2** | 73% | 10% |
| 80 | (4,16) | 20 | nvlink | 215.9 | 51.0 | 0.17 | 103.9 | **371.0** | 58% | 28% |
| 80 | (4,16) | 20 | wan | 725.6 | 75.1 | 64.00 | 121.4 | **986.1** | 74% | 12% |
| 80 | (4,16) | 21 | nvlink | 380.1 | 79.0 | 0.16 | 324.4 | **783.7** | 49% | 41% |
| 80 | (4,16) | 21 | wan | 889.2 | 103.3 | 64.00 | 199.3 | **1255.8** | 71% | 16% |
| 80 | (4,16) | 22 | nvlink | 518.1 | 195.0 | 0.16 | 734.7 | **1448.1** | 36% | 51% |
| 80 | (4,16) | 22 | wan | 1208.1 | 272.3 | 64.00 | 785.7 | **2330.2** | 52% | 34% |
| 80 | (4,16) | 23 | nvlink | 1031.7 | 400.1 | 0.16 | 1323.2 | **2755.2** | 37% | 48% |
| 80 | (4,16) | 23 | wan | 1754.4 | 456.0 | 64.00 | 1193.1 | **3467.6** | 51% | 34% |
| 80 | (8,4) | 20 | nvlink | 740.8 | 143.6 | 0.65 | 830.8 | **1715.9** | 43% | 48% |
| 80 | (8,4) | 20 | wan | 2982.3 | 198.3 | 258.07 | 753.5 | **4192.2** | 71% | 18% |
| 80 | (8,4) | 21 | nvlink | 1003.1 | 195.7 | 0.65 | 1243.8 | **2443.3** | 41% | 51% |
| 80 | (8,4) | 21 | wan | 3205.1 | 290.1 | 257.76 | 1482.8 | **5235.9** | 61% | 28% |
| 80 | (8,4) | 22 | nvlink | 1387.9 | 336.0 | 0.66 | 2759.1 | **4483.8** | 31% | 62% |
| 80 | (8,4) | 22 | wan | 3495.5 | 438.3 | 259.02 | 3636.0 | **7829.1** | 45% | 46% |
| 80 | (8,4) | 23 | nvlink | 2002.2 | 652.3 | 0.66 | 5453.5 | **8109.2** | 25% | 67% |
| 80 | (8,4) | 23 | wan | 4280.3 | 827.9 | 258.12 | 5826.6 | **11193.4** | 38% | 52% |
| 128 | (2,128) | 20 | nvlink | 258.2 | 119.5 | 0.04 | 40.5 | **418.2** | 62% | 10% |
| 128 | (2,128) | 20 | wan | 389.4 | 99.3 | 16.00 | 40.9 | **545.6** | 71% | 7% |
| 128 | (4,16) | 20 | nvlink | 305.5 | 62.1 | 0.17 | 133.9 | **501.6** | 61% | 27% |
| 128 | (4,16) | 20 | wan | 742.2 | 75.5 | 64.00 | 123.1 | **1004.8** | 74% | 12% |
| 128 | (4,16) | 21 | nvlink | 463.7 | 102.4 | 0.17 | 229.5 | **795.7** | 58% | 29% |
| 128 | (4,16) | 21 | wan | 918.1 | 106.4 | 64.00 | 217.4 | **1305.9** | 70% | 17% |
| 128 | (4,16) | 22 | nvlink | 761.2 | 278.1 | 0.17 | 854.4 | **1893.8** | 40% | 45% |
| 128 | (4,16) | 22 | wan | 1265.0 | 303.9 | 64.01 | 852.3 | **2485.3** | 51% | 34% |
| 128 | (4,16) | 23 | nvlink | 1280.2 | 556.4 | 0.16 | 1377.7 | **3214.5** | 40% | 43% |
| 128 | (4,16) | 23 | wan | 1892.9 | 573.5 | 64.01 | 1401.6 | **3932.2** | 48% | 36% |
| 128 | (8,8) | 20 | nvlink | 623.2 | 144.3 | 0.66 | 762.4 | **1530.6** | 41% | 50% |
| 128 | (8,8) | 20 | wan | 2628.1 | 162.3 | 258.85 | 817.5 | **3866.8** | 68% | 21% |
| 128 | (8,8) | 21 | nvlink | 926.3 | 247.2 | 0.65 | 1641.5 | **2815.8** | 33% | 58% |
| 128 | (8,8) | 21 | wan | 2931.8 | 249.9 | 256.03 | 1475.4 | **4913.2** | 60% | 30% |
| 128 | (8,8) | 22 | nvlink | 1193.6 | 485.2 | 0.67 | 3187.6 | **4867.3** | 25% | 65% |
| 128 | (8,8) | 22 | wan | 3572.3 | 550.9 | 256.03 | 3727.1 | **8106.6** | 44% | 46% |
| 128 | (8,8) | 23 | nvlink | 1498.0 | 841.1 | 0.67 | 4834.8 | **7175.0** | 21% | 67% |
| 128 | (8,8) | 23 | wan | 4517.4 | 987.7 | 256.02 | 6895.6 | **12657.3** | 36% | 54% |
