# PIM→GPU Optimized Dataflow for Ring-LPN PCG (OLE)

Target: accelerate `PCG_OLE<P>::gen_and_expand` (silent expansion) by splitting it
across a GPU and an HBM-PIM (near-bank compute in the GPU's own HBM).

- **DPF tree expansion + leaf reduction → PIM** (bank-local, memory-bound).
- **NTT butterflies → GPU** (needs the on-chip shared-memory exchange network).

This document specifies a dataflow that goes **beyond the naive two-phase model**
("PIM does all DPF, then GPU does all NTT"). The wins are algorithmic, not just
"run each kernel on the better engine."

> Prerequisite (orthogonal): replace `%`/`__udivmodti4` in `FFp<P>::operator*`
> with Barrett/Montgomery reduction. See `profiling_report.md §4.1` — modular
> reduction is 71–84% of CPU at t=4. Both GPU-NTT and PIM CW-scaling need it.

---

## 1. The real algorithm (from `pcg_ole_2pc/pcg_ole_impl.h`)

```
for (ii,jj) in c×c pairs:                                   # c²=4 pairs
    [2PC: position add / payload mult / DeltaShift]          # small at large N
    all_leaves = dpf.batch_gen_full_eval(B instances)        # impl:169   B·D leaves
    S = Σ leaves ; [network exchange S → CW]                 # impl:177-216  one round-trip
    g_ij[base+d] += leaf · CW   (scale + scatter into len 2N) # impl:218-229
z = Σ_{i,j} poly_mul( poly_mul(a_i,a_j), reduce_{2N→N}(g_ij) ) # impl:245-256  ← heavy NTT
x = Σ_i poly_mul(a_i, my_errors_i)                           # impl:242-243  errors sparse → cheap
```

Magnitudes (c=2, t=4, N=2²⁴; from `profiling_report.md`):

| quantity | size | note |
|---|---|---|
| raw leaves `c²·B·D = 2c²·t·N` | **≈ 8 GB** | produced then immediately reduced |
| reduced output `g` `c²·N` | **≈ 0.5 GB** | **t× smaller than leaves** |
| `a_i` (public, fixed) | — | ⇒ `NTT(a_i a_j)` is a **precomputable constant** |

---

## 2. Why the naive "A (all DPF) → B (all NTT)" is wasteful

1. It materializes the **8 GB leaf array** and ships it around. But the leaves are a
   *produce-then-immediately-reduce* intermediate (scaled by CW and scattered into the
   t×-smaller `g`). They should never leave HBM.
2. A hard barrier between DPF and NTT leaves PIM and GPU idle in turn.

---

## 3. The optimized dataflow — four algorithm-level moves

### Move 1 — Fuse `expand → sum → scale → scatter → reduce` entirely in PIM
All of these are **bank-local**: DPF expansion (ChaCha/AES, no modmul), local sum,
CW-scale (one scalar modmul per leaf), scatter into `g`, negacyclic 2N→N fold
(add/sub mod P). Only the length-N dense `g_ij` crosses to the GPU.

→ External-bus traffic drops from **8 GB (leaves) to c²·N (≈ 0.5 GB)** — **~16× less**.
This is the single largest, purely algorithmic data-movement win: recognizing the
leaves are an immediately-reduced intermediate.

### Move 2 — Frequency-domain accumulation + single INTT + precomputed `NTT(a_i a_j)`
```
z = INTT( Σ_{i,j} NTT(a_i a_j) ∘ NTT(g_ij) )
```
- `NTT(a_i a_j)` precomputed once (a is public), resident on the GPU.
- Per pair the GPU does only **one forward `NTT(g_ij)` + one pointwise multiply-accumulate**
  into a frequency-domain accumulator `z_hat`.
- After all pairs, **one INTT**.

Collapses "2 full `poly_mul`s (each fwd+inv) per pair" → "c² forwards + 1 inverse".

### Move 3 — Cross-pair producer/consumer pipeline (the real "beyond A+B")
Do **not** finish all DPF before starting NTT. As soon as `g_ij` is ready, the GPU
NTTs it while PIM expands the next pair:

```
PIM lane (INTERNAL bandwidth)              GPU lane (EXTERNAL bus + compute)
──────────────────────────                ───────────────────────────────────
                                           precompute NTT(a_i a_j) resident; z_hat = 0
pair k:  expand B trees
         local sum S_k
         [tiny net exchange → CW_k]    ‖   NTT(g_{k-1}); z_hat += NTT(a_ia_j)∘NTT(g_{k-1})
         scale + scatter → g_k
         reduce 2N→N
   ── hand off g_k (N words) ──→
end:                                       z = INTT(z_hat)        (single)
```

**Why the overlap is near-free (key insight):** PIM's DPF expansion is bound by
**internal bank bandwidth**; the GPU's NTT is bound by the **external HBM bus**. They
bottleneck on *different* resources, so running them concurrently does not contend.
(Contrast: "transpose on PIM" contends on the external bus → marginal. DPF on PIM does not.)

### Move 4 — Minimal handoff + pipeline depth from batching
- Only `g_k` (length N) crosses PIM→GPU; `z_hat` stays resident on GPU.
- c²=4 pairs give only a 4-deep pipeline. Real depth comes from **batching many
  independent OLE instances** (PCG produces many correlations; mirrors the NTT
  library's `batch_size`) — interleave them through the same PIM‖GPU pipeline.

---

## 4. Division of labor (by communication pattern, not by "who can compute")

| step | pattern | engine | why |
|---|---|---|---|
| DPF tree expand | bank-local, no cross-element | **PIM** | memory-bound; internal BW; never leaves HBM |
| local sum / CW-scale / scatter / 2N→N fold | element-wise, bank-local | **PIM** | fuses with expand; only g leaves |
| forward NTT(g), pointwise-MAC, INTT | **cross-element butterfly (stride 2ᵏ)** | **GPU** | needs on-chip shared-mem exchange; PIM has no cheap cross-bank path |
| `x = Σ a_i·err_i` | sparse | GPU (or fold in) | t nonzeros, cheap |

---

## 5. Constraints / risks to surface

1. **CW needs one network round-trip mid-reduction** (`impl:189-216`, B uint64). It
   splits the PIM reduction into `expand+sum | [net] | scale+scatter`. Hide pair k's
   network latency behind pair k-1's PIM/GPU work via the cross-pair/instance pipeline.
2. **The CW-scale is a per-leaf scalar modmul** (`impl:224`). Bank-local, but the DPU
   must support (scalar) modmul. If the DPU only has ChaCha-class add/xor, this step is
   the one part that needs a modmul unit added — it is "multiply by a per-instance
   scalar," lighter than general pointwise, but still modmul.
3. **Barrett/Montgomery is a prerequisite** for GPU-NTT and PIM CW-scale. DPF expansion
   itself has no modmul.
4. **PIM↔GPU coherence / mode switch**: GPU must see PIM-written `g` (flush/invalidate);
   channel mode normal↔PIM around handoffs. Amortize across the batched pipeline.

---

## 6. Expected benefit summary

- External-bus traffic: **~16× less** (only `g` crosses, not the 8 GB leaves).
- DPF expand ‖ GPU NTT overlap: **near-free** (internal vs external bandwidth).
- GPU NTT work minimized: c² forwards + 1 inverse, with `NTT(a)` precomputed.
- Does **not** depend on the contested "transpose-on-PIM" trick.

The robust value here is **DPF + its bank-local reduction staying in HBM**, overlapped
with GPU NTT that bottlenecks on a different bandwidth domain. See
`pim_gpu_workplan.md` for how to build and validate this.
