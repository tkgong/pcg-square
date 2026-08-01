# PIM ISA and kernel assembly (two-tier design, all-bank broadcast)

Companion to the architecture figures. Three kernels, instruction-level:
ChaCha8 DPF expand on the bank PU, leaf-convert scaling on the MAU, NTT
stage-1 butterflies on the MAU. Cycle numbers assume the command clock
(= PU clock): 2.25 GHz GDDR6/L40S, 1.6 GHz HBM2e/HBM3, 2.0 GHz HBM3e;
MAU multiplier pipeline 14 stages on GDDR die periphery, 6 on an HBM base die.

## 0. Register model

Bank PU: `V0..V15` vector registers (256b = 8 lanes x 32b, word-major:
`Vi` holds word i of the 8 seeds in flight) + `SRF` (scalar/constant side:
ChaCha sigma, key schedule, per-level CW pair, feed-forward constants).
17 registers total, zero slack (established in compute_datapath.md).
Dual-ALU variant: XADD unit (8x32b add) + VXORL unit (8x32b fused
xor-rotate); RF banked by column parity giving an effective 4r2w.

MAU: no GPR. Host-written CFR block (CW[0..t-1], bank/row bases, BMASK,
omega seed, clock tier) + internal S-accumulators (t x 64b), 2x1KB
in-FIFOs (one per served channel), 1KB out-FIFO, coefficient generator
(second small modmul running the omega^6 six-way interleaved recurrence).

## 1. Bank PU: ChaCha8 DPF EXTEND batch

One batch = 8 parent seeds -> 16 children. 128b-seed caliber (dpf.cpp
reference): 4 read cols + re-read 4 + 8 write cols; instruction budget
256 core + 16 ff + 4 PACKLR32 + 2 VCXOR + 1 VXOR = 279 (single-ALU cl),
~155 dual-ALU. Broadcast mode: the VLD/VST columns arrive as one ABRD/ABWR
all-bank command serving all 8 PUs of the channel.

```asm
; ---------------- prologue ----------------
VLD     V12, [Br, Rin, C+0]     ; seed word0        (ABRD, broadcast)
VLD     V13, [Br, Rin, C+1]
VLD     V14, [Br, Rin, C+2]
VLD     V15, [Br, Rin, C+3]     ; 128b seed -> state[12..15]
VBCAST  V0,  SRF[0]             ; sigma0..3 -> state[0..3]
VBCAST  V1,  SRF[1]
VBCAST  V2,  SRF[2]
VBCAST  V3,  SRF[3]
VBCAST  V4,  SRF[4]             ; key schedule -> state[4..11] (8 insts)
...
VBCAST  V11, SRF[11]

; ---------------- core: REP 4 double-rounds ----------------
; dual-ALU schedule: QR pairs run anti-phase (A even ticks, B odd ticks).
; One double round = 64 insts / 36 ticks; x4 = 256 insts / ~140 ticks.
; tick  XADD unit                    VXORL unit
;  0    VADD32 V0,V0,V4              -
;  1    VADD32 V1,V1,V5              VXROL V12,V0,#16
;  2    VADD32 V8,V8,V12             VXROL V13,V1,#16
;  3    VADD32 V9,V9,V13             VXROL V4,V8,#12
;  4    VADD32 V0,V0,V4              VXROL V5,V9,#12
;  5    VADD32 V1,V1,V5              VXROL V12,V0,#8
;  6    VADD32 V8,V8,V12             VXROL V13,V1,#8
;  7    VADD32 V9,V9,V13             VXROL V4,V8,#7
;  8    -                            VXROL V5,V9,#7
;  9-17 column pair-2:  QR(V2,V6,V10,V14) || QR(V3,V7,V11,V15)  (same shape)
; 18-26 diagonal pair-3: QR(V0,V5,V10,V15) || QR(V1,V6,V11,V12)
; 27-35 diagonal pair-4: QR(V2,V7,V8,V13)  || QR(V3,V4,V9,V14)

; ---------------- epilogue ----------------
VADD32  V0,V0,SRF[0]            ; feed-forward, constant half (12 insts)
...
VADD32  V11,V11,SRF[11]
VLD     Vt0-Vt3, [Br, Rin, C+0..3]  ; re-read seed (REREAD caliber, 4 cols)
VADD32  V12,V12,Vt0             ; feed-forward, seed half (4 insts)
...
VADD32  V15,V15,Vt3
PACKLR32 V0,V1                  ; cross-lane child interleave (4 insts)
PACKLR32 V2,V3
PACKLR32 V4,V5
PACKLR32 V6,V7
VCXOR   V0, SRF.CW_L, V12       ; predicated xor: lanes with lsb(seed)=1
VCXOR   V4, SRF.CW_R, V12
VXOR    V2, V0, V4              ; free m-th child
VST     V0-V7, [Bw, Rout, C'+0..7]  ; 16 children = 8 cols (ABWR broadcast)
```

Single-ALU total 279 CK/batch; dual-ALU ~155 CK (the 16-inst pure-add ff
tail has no VXORL partner; a 1-deep cross-batch scoreboard fills it with the
next batch's rotates). Column I/O is fully hidden under compute at both
tiers. Broadcast adds nothing to this listing -- it changes only how the
VLD/VST columns are commanded (one ABRD/ABWR per column for all PUs).

## 1b. Bank PU: fused leaf CONVERT (NEW Beaver algorithm — no MAU, no multiplier)

The Beaver-corrected output layer (pcg_ole_impl.h) removed the per-leaf modmul.
Leaf → field is now: `C = Convert(H'(leaf))`, `τ = lsb(leaf)`, and after the
Beaver open `y = C + (τ ? CW : 0)`. `H'` is an independently-keyed ChaCha8
(nonce = OUT_DOMAIN); `Convert` is a sparse-prime reduce; the "scale" is a
control-bit-gated add. **All ARX + a multiplier-free mod-ALU — no MAU.** The
LAST tree level fuses expand + H' + convert + partial-sum into one PU op:

```asm
; ---- last level: expand parent, then hash+convert each of the 2 children ----
LOOP op = 0 .. D/8-1:                 ; one 8-parent batch -> 16 leaves
  ; (a) expand: same ChaCha8 core as levels 0..n-2  (~155 CK dual-ALU)
  VLD  x0..x3, B_par, row_in, col=op        ; 4 seed cols (+ reread for child R)
  <ChaCha8 batch>  ; 32 QR x (VADD32 + VXROL), 16 ff, PACKLR32  -> sL|sR
  ; (b) per child c in {L,R}: H'(child) = ChaCha8(key=child, nonce=OUT_DOMAIN)
  VMOV  key0..key3, childc                  ; child seed as ChaCha key
  VBCAST n14,n15, OUT_DOMAIN                 ; domain-separation nonce
  <ChaCha8 batch>                            ; ~155 CK dual-ALU per child (x2)
  ; (c) Convert + control bit  (mod-ALU, NO multiplier)
  REDP  Cc, hash_lo                          ; C = lo64(H') mod P (sparse fold)
  ADDP  sumC[k], sumC[k], Cc                 ; per-instance running Sigma C
  ADD32 sumT[k], sumT[k], lsb(childc)        ; Sigma tau (tau = child LSB)
  ; (d) store 8B (low 64b only -- half the write columns)
  VST  B_g8, row_out, col=op, Cc.lo64        ; 8B/leaf, wr = WR_COLS/2
MAU_none                                     ; there is no MAU; nothing offloaded
; per-batch compute ~= 1x expand + 2x child-hash + REDP/ADDP ~= 3x the 155 CK
; EXTEND -> cl ~= 465 (dual-ALU) / 846 (single).  Only the last level pays this;
; levels 0..n-2 stay at 155/282.  No re-read reduce tail (saved vs old fused/mau).
; [host, once per instance: Beaver open A=Sigma tau, B=Sigma C + beta -> CW]
; [scatter y = C + tau?CW recomputes H' in a second pass OR is folded here when
;  CW is available before the write; modeled as the same last-level op stream.]
```

## 2. MAU: leaf-convert (two passes)  [OLD algorithm — historical, modmul-based]

Macro-ISR level; the MAU retires one 8-leaf-group micro-op per cycle.

```asm
; ---- pass 1: bin sums S_k ----
CFR.WR   BASE_LEAF, row0             ; leaf region base (round-parity buffer)
LOOP i = 0 .. D/4-1:                 ; one 8-leaf group (2 cols) per iter
  MAU_LOAD  ch, B_leaf, row0+i/32, col=2i%64, n=2
  MAU_SUM   k                        ; per leaf, 1/CK:
                                     ;   S[k] = MADD(S[k], +-leaf_j)  (MMULT bypass)
MAU_FLUSH                            ; S[0..t-1] -> CFR -> host
; host: exchange S with peer, compute CW_k = beta/v mod P, write CFR.CW[*]

; ---- pass 2: CW scale + scatter into g ----
LOOP i = 0 .. D/4-1:
  MAU_LOAD  ch, B_leaf, row0+i/32, col=2i%64, n=2
  MAU_SCALE k, t, D                  ; per leaf, issue 1/CK, latency 16 CK:
                                     ;  S1-14: P0 = leaf_j x CW[k]      (MMULT 62x62)
                                     ;  S15:   r  = lo + (hi<<25) + (hi<<26) - hi
                                     ;         (sparse-prime reduce, hardwired)
                                     ;  S16:   pos = (b/t + b%t)*(D/2) + d   (AGU)
                                     ;         gacc[pos mod N] = MADD(gacc, +-r)
  MAU_STORE ch, B_g, GROW+i/32, col=2i%64, n=2
MAU_FLUSH
; throughput 1 leaf/CK; one instance = D/2 CK + 16 CK drain
```

## 3. MAU: NTT stage-1 butterfly (column NTT, N1 = 2^14, forward DIF)

```asm
CFR.WR    OMEGA_SEED, w_s
MAU_TWIST wseed=w_s, ilv=6           ; coeff-gen: c[0..5]=w^j warmup, then
                                     ; c[j+6] = MMULT2(c[j], w^6) -- second
                                     ; small modmul, 6-way interleaved, 1 w/CK
LOOP s = 0 .. 13:                    ; 14 stages
  LOOP blk = 0 .. N1/2/64 - 1:       ; 64 pairs per block (one row)
    MAU_LOAD  ch, B_ntt,  rowA(s,blk), n=64
    MAU_LOAD  ch, B_ntt,  rowB(s,blk), n=64   ; stride 2^s pairs
    MAU_BFLY  stage=s, dir=fwd, n=64          ; per pair, 1/CK:
                                              ;  S1-14: u = b_i x c_i   (MMULT)
                                              ;  S15:   u = sparse-P reduce
                                              ;  S16:   a' = MADD(a_i,u) ||
                                              ;         b' = MSUB(a_i,u)  (parallel pair)
    MAU_STORE ch, B_ntt', rowA'(s,blk), n=64  ; double-buffered writeback
    MAU_STORE ch, B_ntt', rowB'(s,blk), n=64
MAU_FLUSH
; per stage: 8192 pairs = 8192 CK (load/store hidden under the pipeline)
; one column NTT = 14 x 8192 + fills ~= 131K CK ~= 58 us @2.25GHz / 82 us @1.6GHz
; butterfly latency 21 CK (GDDR 14-stage mult) / 9 CK (HBM base die); 1 pair/CK
```

## 4. The unifying point

`MAU_SUM`, `MAU_SCALE` and `MAU_BFLY` are the three settings of the dest
mux: stages S1-S15 (MMULT -> sparse-prime reduction) are one physical
pipeline; only S16's destination differs (S-accumulator / g-AGU MADD /
parallel MADD-MSUB pair). One fused pipeline covers all three kernels with
no second datapath -- which is also the microarchitectural root of the
case-A capacity constraint: leaf scaling and stage-1 butterflies time-share
the same 1 op/CK budget (measured by the campaign's `_ntt1` contention
points and the `[mau-die]` occupancy check).

## 5. Addendum: issue units, MC duties, and the epilogue schedule

### 5.1 PU instruction reference (unit / MC role / cycles)

| inst | issue -> execute | memory controller's role | CK |
|---|---|---|---|
| SRF/CFR write | MC -> bank SRF | reg-RW mode (TMOD), per-level, amortized | ~2/word |
| VLD col | useq -> LSU load | row open (ACT16) + ABRD at tCCD pace | 2-3, hidden |
| VST col | useq -> LSU store q | ABWR + nRTW/nWTR turnaround | 2-3, hidden |
| VADD32(I) | XADD array | none | 1 |
| VXROL/VXOR/VCXOR | VXORL array | none | 1 |
| PACKLR32 | PACK crossbar (only cross-lane path; 2r2w VRF) | none | 1 |
| VBCAST | SRF bypass (scalar broadcast, no VRF read port) | none | 0-1 |
| **REDP** Vd,Vs | **mod-ALU: sparse-prime reduce** `r = lo + (hi<<25)+(hi<<26) - hi` + 1-2 cond-sub (P = 2^62-3·2^25+1). ≥62-bit carry, **no multiplier** | none | ~3-4 |
| **ADDP** Vd,Va,Vb | **mod-ALU: modular add** `s = a+b; s>=P?s-P` (≥62-bit carry across the 32b lanes) | none | ~2 |
| **CADDP** Vd,Vs,Vctrl | **mod-ALU: control-bit-gated modular add** (= ADDP iff lsb(Vctrl); the `τ?+CW` scatter) | none | ~2 |
| **NEGP** Vd,Vs | **mod-ALU: modular negate** `P - v` (party-1 sign) | none | ~2 |
| REP / DONE | useq | DONE: sampled via wired-AND as the gate for the next cl-bearing broadcast | 0 |

The four **mod-ALU** ops (REDP/ADDP/CADDP/NEGP) are the *only* new functional
unit the Beaver algorithm needs: a ≥62-bit adder with carry across the ARX
ALU's 32-bit lanes + the sparse-prime shift-fold + a conditional subtract.
**Multiplier-free** — the sole per-instance field multiply (Beaver `CW = A·B`)
runs on the host 2PC layer, not the PU. The ChaCha8 out-hash `H'` reuses the
existing VADD32/VXROL datapath: zero new FU for the hash.

MC duties overall: ISR->command expansion (1 trace line = ACT16(amortized) +
ABRD x opsize + ABWR x wrsize), DRAM timing legality, CFR/BMASK writes via
TMOD, done-gating, routing MAU cl=0 column streams, refresh.

### 5.2 The 16 column commands per EXTEND op

reads 8 = 4 seed cols (8 seeds x 16B / 32B) + 4 re-read cols (REREAD: ff
needs the seed half of the initial state; SRF holds only the constant half
and the 16 VRF regs are scrambled by the rounds -- the zero-slack register
budget forces the re-read). writes 8 = 16 children x 16B / 32B. Total 512B/op
= the opsize=8 / wrsize=8 fields of the trace line. ACT16 amortizes 1/8 ops
(64 cols/row / 8 cols). Dropping REREAD (4 shadow regs, +128B RF) would cut
reads to 4 -- worth revisiting only at per-bank PU density.

### 5.3 Dual-ALU epilogue: three units in parallel, ~11 net CK

After the 144-CK core: ff (16 VADD32) runs on XADD ticks 1-16; PACK x4 hides
under ff ticks 9-12 (own crossbar); VCXOR x2 + VXOR hide under ff ticks 13-15
(VXORL idle); VST drains via LSU (0 compute CK). Critical path = ff 16 CK,
minus ~5 CK cross-batch software-pipelining credit => cl = 144 + ~11 = 155.
The only pairing-blind segment is ff (pure adds, no VXORL partner); fixing it
would need a second adder path (+8 lanes for ~7% -- rejected, below the
dual-ALU step's return of +40% area for 44% cycles).

### 5.4 LSU staging vs compute (measured)

LSU alone (cl=0 broadcast floor): ~67 CK/op/channel incl. ACT16, turnaround,
barriers (bc_cl0_none: 138,564 CK / 2,056 ops). Compute: 155 (dual) / 272-279
(single). Steady state = max => LSU fully hidden; slack 2.3x / 4x. The
parallelism holds at full load ONLY under broadcast: independent addressing
multiplies LSU demand by 8 PUs (measured 0.0921 vs 0.0549 ns/leaf at cl155).
Load staging is a skid buffer (no tags/reuse): it absorbs the mismatch
between the controller's ABRD schedule, the RF's 4r2w port budget (both
write ports busy during core), and register liveness (batch k+1's columns
arrive while V12-V15 still hold batch k) -- FIFO form of double buffering at
half the transistors/bit of extra RF ports.
