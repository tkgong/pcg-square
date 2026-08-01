# End-to-end speedup of the co-scheduled PIM+GPU design

Both sides gang their communication (`n+2` messages, identical bytes). The PIM
side additionally overlaps the network with the in-bank expansion — that overlap
is the only thing the GPU cannot replicate, because its DPF and NTT share one SM
while the PIM's SPU and the SM are independent.

```
GPU baseline  =  DPF + NTT + NIC              serial: one compute resource
PIM+GPU       =  max(SPU, SM, NIC)            three independent lanes
NIC           =  (n+2) · α                    same message count on both sides
```

`ᴺ` marks a cell where the NIC lane has overtaken the compute lane.
`a*` = `max(SPU, SM) / (n+2)` is the α at which that flip happens.

---

## 1. Provenance

| input | source | grade |
|---|---|---|
| **L40S GPU, all phases** | 180-cell α sweep on an exclusive L40S, `alpha_sweep.csv` | measured |
| **B200 GPU, all phases** | `2dd9682` — Slurm job 239193, node `b0010`, 1 × B200 sm_100, driver 595.71.05, CUDA 13.0, 1973 s, 2026-07-30. Raw: `GPU_baseline/pcg_baseline_out/` (677 files) | measured |
| **NTT lane, both** | `ntt_stages.csv`, device-only four-step, standalone bit-reversal | measured |
| **DRU offload share (B200)** | transpose+brev fraction of the four-step, 24–29% | measured |
| **PIM per-level ladder** | ramulator2, `n=1…13` on both orgs; `t_E(i)=CK(n=i+1)−CK(n=i)` | measured |
| **PIM output-layer share** | 45.8% at both n=11 and n=12, org-independent | measured |
| **PIM occupancy penalty** | `I=1024` and `I=2048` both cost 45,373 CK → half-full machine wastes half | measured |
| **B200 full card ×2** | 1024 SPU/I=4096 and 2048 SPU/I=8192 both 181,648 CK — exact | measured |
| **DRU contention** | ±`--sm-stream`: L40S +0.00%, B200 +0.01% | measured |
| **co-schedule mechanism** | `ISR_NET_DELAY`/`ISR_NET_WAIT`; `serial = base + ladder` to ±0.05% | measured |
| GPU batching cost | +2.6% on compute, applied to the ganged GPU | measured |

**Caveat on the B200 GPU run.** The node was not idle: a co-tenant held 7 of 8
GPUs and 100% of the CPU cores for the whole run. The GPU die was cgroup-isolated
(0 MiB, 0% at both ends), so device-resident kernel work is clean; host-mediated
paths — CPU, PCIe root complex, NUMA fabric — were not protected. Cross-checked
against its own `(n,B)` surface: **30/30 cells at 0.95–1.05×**.

---

## 2. Results

### L40S (192 SPU)

| λ | (c,t) | logN | occ | SPU ms | SM ms | a\* | 100 µs | 500 µs | 10 ms | 20 ms | 30 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 80 | (2,64) | 22 | 1× | 218 | 29 | **11.5 ms** | 2.35× | 2.38× | 3.21× | 2.34×ᴺ | 1.90×ᴺ |
| 80 | (2,64) | 23 | 1× | 437 | 59 | **21.8 ms** | 2.26× | 2.27× | 2.71× | 3.17× | 2.64×ᴺ |
| 80 | (4,16) | 22 | 1× | 218 | 114 | **10.4 ms** | 2.69× | 2.73× | 3.65× | 2.40×ᴺ | 1.93×ᴺ |
| 80 | (4,16) | 23 | 1× | 437 | 227 | **19.9 ms** | 2.66× | 2.68× | 3.16× | 3.64×ᴺ | 2.76×ᴺ |
| 80 | (4,16) | 24 | 1× | 874 | 456 | **38.0 ms** | 2.65× | 2.66× | 2.91× | 3.18× | 3.44× |
| 80 | (8,4) | 22 | 1× | 218 | 439 | **19.1 ms** | 2.36× | 2.38× | 2.88× | 3.25×ᴺ | 2.50×ᴺ |
| 80 | (8,4) | 23 | 1× | 437 | 904 | **37.7 ms** | 2.25× | 2.26× | 2.51× | 2.77× | 3.04× |
| 80 | (8,4) | 24 | 1× | 874 | 1816 | **72.6 ms** | 2.26× | 2.26× | 2.39× | 2.53× | 2.67× |
| 128 | (2,128) | 22 | 1× | 437 | 29 | **24.3 ms** | 2.13× | 2.15× | 2.54× | 2.95× | 2.72×ᴺ |
| 128 | (4,16) | 22 | 1× | 218 | 114 | **10.4 ms** | 2.69× | 2.73× | 3.65× | 2.40×ᴺ | 1.93×ᴺ |
| 128 | (4,16) | 23 | 1× | 437 | 227 | **19.9 ms** | 2.66× | 2.68× | 3.16× | 3.64×ᴺ | 2.76×ᴺ |
| 128 | (4,16) | 24 | 1× | 874 | 456 | **38.0 ms** | 2.65× | 2.66× | 2.91× | 3.18× | 3.44× |
| 128 | (8,8) | 22 | 1× | 437 | 439 | **20.0 ms** | 3.29× | 3.31× | 3.78× | 4.28×ᴺ | 3.18×ᴺ |
| 128 | (8,8) | 23 | 1× | 874 | 904 | **39.3 ms** | 3.20× | 3.21× | 3.45× | 3.71× | 3.96× |
| 128 | (8,8) | 24 | 1× | 1748 | 1816 | **75.6 ms** | 3.18× | 3.19× | 3.31× | 3.45× | 3.58× |
| **geomean** | | | | | | | **2.60×** | **2.61×** | **3.05×** | **3.08×** | **2.76×** |
| **NET-bound** | | | | | | | 0/15 | 0/15 | 0/15 | 7/15 | 9/15 |

### B200 (2048 SPU)

| λ | (c,t) | logN | occ | SPU ms | SM ms | a\* | 100 µs | 500 µs | 10 ms | 20 ms | 30 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 80 | (2,64) | 22 | 1× | 24 | 6 | **1.2 ms** | 7.86× | 8.18× | 1.97×ᴺ | 1.48×ᴺ | 1.32×ᴺ |
| 80 | (2,64) | 23 | 1× | 47 | 11 | **2.4 ms** | 7.68× | 7.85× | 2.81×ᴺ | 1.90×ᴺ | 1.60×ᴺ |
| 80 | (2,64) | 24 | 1× | 95 | 23 | **4.5 ms** | 7.30× | 7.39× | 4.28×ᴺ | 2.64×ᴺ | 2.09×ᴺ |
| 80 | (4,16) | 22 | 1× | 24 | 21 | **1.1 ms** | 9.36× | 9.72× | 2.05×ᴺ | 1.52×ᴺ | 1.35×ᴺ |
| 80 | (4,16) | 23 | 1× | 47 | 43 | **2.2 ms** | 8.98× | 9.16× | 2.92×ᴺ | 1.96×ᴺ | 1.64×ᴺ |
| 80 | (4,16) | 24 | 1× | 95 | 89 | **4.1 ms** | 8.57× | 8.67× | 4.52×ᴺ | 2.76×ᴺ | 2.17×ᴺ |
| 80 | (8,4) | 22 | 2× | 47 | 82 | **3.6 ms** | 4.76× | 4.87× | 2.70×ᴺ | 1.85×ᴺ | 1.57×ᴺ |
| 80 | (8,4) | 23 | 2× | 95 | 171 | **7.1 ms** | 4.43× | 4.49× | 4.15×ᴺ | 2.58×ᴺ | 2.05×ᴺ |
| 80 | (8,4) | 24 | 2× | 189 | 352 | **14.1 ms** | 4.22× | 4.25× | 4.92× | 3.97×ᴺ | 2.98×ᴺ |
| 128 | (2,128) | 22 | 1× | 47 | 6 | **2.6 ms** | 7.53× | 7.68× | 2.97×ᴺ | 1.99×ᴺ | 1.66×ᴺ |
| 128 | (2,128) | 23 | 1× | 95 | 11 | **5.0 ms** | 7.39× | 7.47× | 4.67×ᴺ | 2.84×ᴺ | 2.22×ᴺ |
| 128 | (2,128) | 24 | 1× | 189 | 23 | **9.5 ms** | 7.31× | 7.35× | 7.91×ᴺ | 4.46×ᴺ | 3.30×ᴺ |
| 128 | (4,16) | 22 | 1× | 24 | 21 | **1.1 ms** | 9.36× | 9.72× | 2.05×ᴺ | 1.52×ᴺ | 1.35×ᴺ |
| 128 | (4,16) | 23 | 1× | 47 | 43 | **2.2 ms** | 8.98× | 9.16× | 2.92×ᴺ | 1.96×ᴺ | 1.64×ᴺ |
| 128 | (4,16) | 24 | 1× | 95 | 89 | **4.1 ms** | 8.57× | 8.67× | 4.52×ᴺ | 2.76×ᴺ | 2.17×ᴺ |
| 128 | (8,8) | 22 | 1× | 47 | 82 | **3.7 ms** | 6.67× | 6.77× | 3.49×ᴺ | 2.24×ᴺ | 1.83×ᴺ |
| 128 | (8,8) | 23 | 1× | 95 | 171 | **7.4 ms** | 5.90× | 5.96× | 5.39×ᴺ | 3.19×ᴺ | 2.46×ᴺ |
| 128 | (8,8) | 24 | 1× | 189 | 352 | **14.7 ms** | 5.95× | 5.98× | 6.62× | 5.36×ᴺ | 3.91×ᴺ |
| **geomean** | | | | | | | **7.07×** | **7.20×** | **3.65×** | **2.44×** | **1.98×** |
| **NET-bound** | | | | | | | 0/18 | 0/18 | 16/18 | 18/18 | 18/18 |

---

## 3. The flip point

`a* = max(SPU, SM) / (n+2)` — where the NIC lane overtakes the compute lane.

| | `a*` range | still compute-bound at 10 ms? |
|---|---|---|
| **L40S** (192 SPU) | **10.4 – 75.6 ms** | yes — 0/15 cells flipped |
| **B200** (2048 SPU) | **1.13 – 14.7 ms** | no — 16/18 flipped |

B200's flip point is an order of magnitude earlier because its compute lane is
9× narrower (24–352 ms against L40S's 218–1816 ms) while the message count is
identical. **A faster PIM buys less network tolerance, not more.** That is the
central trade-off of this design and it is not visible in any single-α table.

Earliest flip: B200 `(4,16) logN 22` at **1.13 ms** — a 24 ms compute lane is
saturated by 25 messages at that RTT. Latest: L40S `(8,8) logN 24` at **75.6 ms**.

## 4. What the co-schedule is worth on its own

Ablating only the overlap (both sides still ganged):

| α | L40S off → on | gain | B200 off → on | gain |
|---|---|---|---|---|
| 50 µs | 1.73 → 2.59 | **1.50×** | 4.59 → 7.05 | **1.54×** |
| 500 µs | 1.71 → 2.60 | 1.52× | 4.27 → 7.20 | 1.69× |
| 2 ms | 1.69 → 2.69 | 1.59× | 3.55 → 7.01 | **1.97×** |
| 5 ms | 1.64 → 2.84 | **1.74×** | 2.83 → 5.33 | 1.89× |

The co-schedule is the largest single term: of L40S's 2.59× at 50 µs, 1.50× is
the overlap and only 1.73× is "the SPU computes the DPF faster than the SM".
Peak at α ≈ 2–5 ms, where the NIC and compute lanes are comparable and the
overlap has the most to hide. The simulator measures the same peak directly:
`sched` vs `serial` is **1.61×** where ladder = compute.

## 5. Deployment reading

**L40S** — compute-bound from 100 µs to 10 ms, 2.6–3.1×. Covers same-rack
through same-metro without the number moving.

**B200** — 7.1–7.2× only up to ~500 µs. Past 1.13 ms cells begin flipping; by
10 ms, 16 of 18 are network-bound and the geomean is 3.65×. Deploying this design
on B200 requires the two parties inside one datacenter.

## 6. Not measured

* The GPU's ganged arm OOMs at logN ≥ 22 (needs `c²` × the leaf pool); its
  compute there is taken from the serial arm plus the measured +2.6% batching
  cost rather than run directly.
* Bandwidth is excluded. At 3 Gbps the α-equivalent is `2·16t²/BW`: +1.4 µs at
  t=4, +21.8 µs at t=16, +1398 µs at t=128. For `(2,128)` that dominates and
  shifts its working point right by more than a millisecond.
* `a*` assumes the gang is available at every N on both sides.
