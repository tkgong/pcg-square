# DRU: RTL-level specification for the NTT movement path

Scope: everything needed to write the RTL for the address path. The datapath
(steering MUX, 32B write combiner, request port) is unchanged from the DPF
transpose design and is specified only where the NTT path touches it.

Status of every claim is marked: **[D]** derived here, **[M]** measured,
**[O]** open / undetermined.

---

## 1. Problem statement

One pass moves `B` polynomials of `N = 2^n` 64-bit coefficients under a fixed
index permutation. `n1 = 2^a`, `n2 = 2^b`, `a = ceil(n/2)`, `b = floor(n/2)`,
`a + b = n`. Legal `n` is 10..24, so `a in [5,12]`, `b in [5,12]`.

Within one polynomial, index `i in [0,N)` is read as a row-major `n1 x n2`
matrix, `i = i1*n2 + i2`:

```
i1 = i[n-1:b]   (a bits)      i1[k] = i[b+k],  k in [0,a)
i2 = i[b-1:0]   (b bits)      i2[k] = i[k],    k in [0,b)
```

A pass computes `dst[P(i)] = src[i]`. The six permutations the four-step chain
needs are listed in §2.

---

## 2. The permutation is a bit permutation of the index  **[D]**

Every required `P` maps each destination index bit to exactly one source index
bit. Writing `D[p] = i[sigma(p)]`:

| mode | name | meaning | `sigma(p)` |
|---|---|---|---|
| 0 | `XPOSE` | `n1 x n2 -> n2 x n1` | `(p+b) mod n` |
| 1 | `XPOSE_I` | `n2 x n1 -> n1 x n2` | `(p+a) mod n` |
| 2 | `BREV_LO` | reverse the `n2` axis | `p` if `p>=b`, else `b-1-p` |
| 3 | `BREV_HI` | reverse the `n1` axis | `p` if `p<b`, else `n+b-1-p` |
| 4 | `BREV_LO_X` | `BREV_LO` then `XPOSE` | `n-1-p` if `p>=a`, else `p+b` |
| 5 | `BREV_HI_X` | `BREV_HI` then `XPOSE` | `p-a` if `p>=a`, else `n-1-p` |

Derivations (all six checked bit by bit):

- mode 0: `D` high `b` bits `= i2`, low `a` bits `= i1`. For `p>=a`,
  `D[p]=i2[p-a]=i[p-a]`; for `p<a`, `D[p]=i1[p]=i[p+b]`. Since `a+b=n`, both
  branches are `(p+b) mod n`.
- mode 1 is the inverse of mode 0, hence rotation by `a`.
- mode 3: `i1[k]=i[b+k]`, so `reverse_a(i1)[k]=i[n-1-k]`; with `p=b+k`,
  `sigma(p)=n-1-(p-b)=n+b-1-p`.
- mode 4: `sigma_4 = sigma_2 o sigma_0`. For `p>=a`, `(p+b) mod n = p-a < b`, so
  `sigma_2` gives `b-1-(p-a) = n-1-p`. For `p<a`, `p+b >= b`, so `sigma_2` is
  the identity.
- mode 5: `sigma_5 = sigma_3 o sigma_0`, same case split.

### 2.1 The six modes collapse to {reverse, rotate}  **[D]**

Reducing mod `n`, the two branch pairs unify:

```
b-1-p  (p<b)  and  n+b-1-p  (p>=b)   ==   (b-1-p) mod n
n-1-p                                 ==   (-1-p)  mod n
```

so every `sigma` is of the form `sigma(p) = (+/-p + s) mod n`. Setting
`rev = 1` for the negated forms:

| mode | `rev` | `s` |
|---|---|---|
| 0 `XPOSE` | 0 | `b` |
| 1 `XPOSE_I` | 0 | `a` |
| 2 `BREV_LO` | see §2.2 | |
| 3 `BREV_HI` | see §2.2 | |
| 4 `BREV_LO_X` | 1 | `b-1`, and 0/`b` on the other branch |
| 5 `BREV_HI_X` | 1 | `n-1`, and 0/`-a` on the other branch |

Modes 0, 1 are pure rotations. Modes 4, 5 are a full-index reversal followed by
a rotation — a single `{rev, s}` pair each:

```
mode 4:  rev=1, s=b-1     ( (b-1-p) mod n  for all p, checked both branches )
mode 5:  rev=1, s=n-1     ( (-1-p)  mod n  for all p, checked both branches )
```

Verification of the mode-4 unification: for `p<a`, `(b-1-p) mod n`; the table
says `p+b`. These agree only if `b-1-p ≡ p+b (mod n)`, which is false. **So
modes 4 and 5 are NOT single `{rev,s}` pairs** — they are field-wise, with a
different `{rev,s}` on each of the two fields. See §2.2.

### 2.2 Correct structure: per-field {reverse, rotate}  **[D]**

Modes 2, 3, 4, 5 act differently on the two index fields, so the network is
**two independent `{rev, s}` units, one per field, plus a field swap**:

```
stage 1  FIELD SPLIT   i -> (i1 = i[n-1:b], i2 = i[b-1:0])
stage 2  per-field     u1 = rev1 ? reverse_a(i1) : i1
                       u2 = rev2 ? reverse_b(i2) : i2
stage 3  FIELD SWAP    (v1,v2) = swap ? (u2,u1) : (u1,u2)
stage 4  CONCAT        D = v1 * (2^width(v2)) + v2
```

Mode decode:

| mode | `rev1` | `rev2` | `swap` |
|---|---|---|---|
| 0 `XPOSE` | 0 | 0 | 1 |
| 1 `XPOSE_I` | 0 | 0 | 1 |
| 2 `BREV_LO` | 0 | 1 | 0 |
| 3 `BREV_HI` | 1 | 0 | 0 |
| 4 `BREV_LO_X` | 0 | 1 | 1 |
| 5 `BREV_HI_X` | 1 | 0 | 1 |

Modes 0 and 1 have identical decode because the two transposes differ only in
which factor is the leading dimension — that difference is carried by `a`/`b`
in the concat width, not by the permutation. **This is the whole point: fusing
a bit reversal with a transpose sets `rev2=1` and `swap=1` simultaneously, so it
costs one pass, not two.**

### 2.3 RTL  **[D]**

`a`, `b`, `n`, `mode` are constant for the whole pass, so all selects are
**descriptor-time constants**, not per-cycle control. Reversal within a
variable-width field is a full-width reversal plus an alignment shift, and the
alignment folds into the concat shift, giving one funnel shifter per field:

```verilog
// descriptor-time constants, computed once when the descriptor is latched
localparam W = 24;                      // max n
reg [4:0] a_r, b_r;                     // log n1, log n2
reg [2:0] mode_r;
wire rev1  = mode_r[0] & ~mode_r[2] | (mode_r == 3'd5);   // see table
wire rev2  = ...;                                          // ditto
wire swap  = (mode_r == 0) | (mode_r == 1) | (mode_r >= 4);

// per-cycle datapath: pure wiring + two funnel shifts with static amounts
function [W-1:0] rev_w(input [W-1:0] x);                  // free: wiring
  integer k; for (k=0;k<W;k=k+1) rev_w[k] = x[W-1-k];
endfunction

wire [W-1:0] i1 = idx >> b_r;
wire [W-1:0] i2 = idx & ((1<<b_r)-1);
wire [W-1:0] u1 = rev1 ? (rev_w(i1) >> (W - a_r)) : i1;
wire [W-1:0] u2 = rev2 ? (rev_w(i2) >> (W - b_r)) : i2;
wire [W-1:0] v1 = swap ? u2 : u1;
wire [W-1:0] v2 = swap ? u1 : u2;
wire [4:0]  wlo = swap ? a_r : b_r;                        // width of v2
wire [W-1:0] D  = (v1 << wlo) | v2;
```

Cost, counted as 2:1 mux equivalents on a 24-bit field:

| element | muxes |
|---|---|
| `>> b_r` and `& mask` (field split) | 2 x 24 x 5 = 240 |
| two conditional reversals (wiring + one shift each) | 2 x (24 + 24x5) = 288 |
| field swap | 2 x 24 = 48 |
| concat shift `<< wlo` | 24 x 5 = 120 |
| **total** | **~700 muxes ≈ 2.1k gates** |

**This supersedes the ~600-gate figure I gave earlier, which counted one mux
layer and ignored the shifters.** [D] The variable-width field split and concat
are what cost — they exist because `N` sweeps `2^10..2^24`. Fixing `N` per
build would remove ~500 of the 700 muxes.

Area consequence: logic grows from ~3k to **~5k gates**, not ~4k. Still no
multiplier, no SRAM array.

---

## 3. Physical address assembly  **[D for the slicing, O for the channel hash]**

Linear element index within a pass: `E = {batch, D}`, `batch` is `beta` bits
(`B = c^2 in {4,16,64}` so `beta in {2,4,6}`), `D` is `n` bits.

Byte address `= base + 8*E`. With a 32B column (4 x 8B elements), 64 columns
per row, 16 banks per channel:

```
off  = E[1:0]      element within the 32B column   -> steering MUX select
col  = E[7:2]      6 bits
bank = E[11:8]     4 bits
row  = E[28:12] + row_base
```

`src` uses `base = src_base`, index `i`; `dst` uses `base = dst_base`, index
`D`. Both sides use the same slicing — the permutation is applied to the index
before slicing, which is why no new datapath is needed.

Channel residency holds by construction for every mode: `E` above is the
**channel-local** element index, and the channel field is not part of the
permuted index — it passes through untouched. Each channel's DRU permutes only
its own slice, so no mode can generate inter-channel traffic. The DRU has no
path to another channel and needs none.

What must be agreed is the resulting **layout contract** with the consuming SM
kernel: the destination layout is whatever the DRU writes, so the next kernel's
indexing must match it. That is a two-sided convention, not a DRU capability
question.

---

## 4. Loop nest and counters  **[D]**

Three nested loops, outermost first:

| level | counter | width | extent | role |
|---|---|---|---|---|
| 0 | `Bc` | 7 | `B` | batch element |
| 1 | `Tc` | 12 | `N / 2^(2t)` | tile within the polynomial |
| 2 | `Uc` | 6 | `2^(2t) / 4` | 32B column within the tile |

`off = Uc[1:0]` drives the steering MUX; the gather reads 4 elements whose
addresses differ only in `bank`, so one `GRD` group is 4 requests to 4
consecutive banks and produces exactly one destination column.

Increment order: `Uc` every accepted gather group, `Tc` on `Uc` wrap, `Bc` on
`Tc` wrap, `DONE` on `Bc` wrap. Adds one 7-bit incrementer and one comparator
over the DPF design.

---

## 5. Handshake  **[D except where noted]**

```
IDLE      -> wait for descriptor valid (active CFR set loaded)
GATHER    -> issue 4 GRD (one per bank) when tap grants; stall while denied
COMBINE   -> as each beat returns, steering MUX selects element off=Uc[1:0],
             write combiner latches it at its position
EMIT      -> when all 4 positions are valid, issue one CWR (32B)
             increment counters; back to GATHER unless DONE
DONE      -> issue DBELL; if the shadow descriptor is valid, promote it to the
             active CFR set on the next cycle and re-enter GATHER
```

Tap arbitration: `GRD`/`CWR` are presented only when `databus_free` is
asserted. **[O]** The ESC counter increments per denied cycle and forces a
grant at `K`; `K` is unmeasured. Until `K` is fixed the deadlock argument for
the co-scheduled pipeline is incomplete, because the SM's next kernel waits on
`DBELL`.

Outstanding requests needed to sustain the measured demand: 372 GB/s aggregate
over 128 pseudo-channels is 2.9 GB/s per channel [M, derived from the measured
stage split]; at ~200 ns effective latency that is ~580 B, i.e. **~20
outstanding 32B requests**. The existing 128-entry tap FIFO is ample.

---

## 6. Descriptor format  **[D]**

106 bits, one active copy in the CFRs and one pending copy in the shadow
register:

| field | bits | notes |
|---|---|---|
| `mode` | 3 | §2.2 |
| `log_n1` (`a`) | 5 | 5..12 |
| `log_n2` (`b`) | 5 | 5..12 |
| `log_batch` (`beta`) | 3 | 2, 4 or 6 |
| `src_base` | 27 | element address |
| `dst_base` | 27 | element address |
| `tile_log` (`t`) | 3 | tile size exponent |
| `n_tiles` | 12 | loop-1 extent |
| `n_cols` | 6 | loop-2 extent |
| reserved | 15 | |

Written by `CFRW`; a pass is armed by `XPOSE`; `SIGNAL` configures the
doorbell target. No new opcode is required — `mode` is a descriptor field, not
an instruction.

---

## 7. Open items, in order of how much they can change the design

1. **[O] Channel residency for the pure-bit-reversal modes** (§3). Could force
   a tile-size constraint or a re-cost of 2 of the 8 passes.
2. **[O] `K`** for the escalation counter (§5). Needs the simulator.
3. **[O] Gate count** is a mux-equivalent hand count (§2.3), not synthesis.
4. **[O] The 12 -> 8 pass fusion has not been measured**, only derived. It is
   the source of the 1.24x latency claim and half of the 1.88x throughput
   claim.
