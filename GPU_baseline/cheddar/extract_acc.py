#!/usr/bin/env python3
"""Two kernel variants for the PCG output z = INTT( sum_ij a^_i . a^_j . NTT(g_ij) ), derived mechanically from Cheddar's
NTTPhase1 / NTTPhase2 in gen/cheddar_ntt_kernels.cuh (extract.sh):
  NTTPhase2Acc     Cheddar's row kernel (last P2 stages) whose write-back is replaced by a multiply-accumulate epilogue:
                   a block loops over `items` polynomials g_ij of its group (blockIdx.y), transforms the same row of each,
                   multiplies the result by a^_i and a^_j at the same positions (item = i * c + j, a^ = ahat + i * N) and
                   accumulates in registers; the group's partial sum is written once. Butterflies, twiddles and the
                   shared-memory exchange are untouched; a __syncthreads() separates the items (shared memory is reused).
  NTTPhase1ContigIn Cheddar's column kernel (first P1 stages) reading the polynomial in the transposed (column-contiguous)
                   layout that the SPUs write g_ij in, and storing in Cheddar's natural layout for the row kernel: only the
                   load addresses change (as extract_contig.sh), the stores are Cheddar's own strided stores.
Usage: extract_acc.py IN OUT"""
import re, sys
src = open(sys.argv[1]).read()


def func(name, head):
    s = src.index(head); s = src.rindex("template <", 0, s); e = src.index("\n}\n", s) + 3
    return src[s:e]


# ---- NTTPhase2Acc ----
p2 = func("NTTPhase2", "__global__ void NTTPhase2(")
n = [0]
def sub(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old)); n[0] += 1; return text.replace(old, new)
p2 = sub(p2, "__global__ void NTTPhase2(", "__global__ void NTTPhase2Acc(")
p2 = sub(p2, "const InputPtrList<make_signed_t<word>, 1> src) {",
         "const InputPtrList<make_signed_t<word>, 1> src,\n"
         "                          const word *ahat, int c, int items, word *acc, ::Modulus<word> mod, int pairs, const word *bhat) {")
p2 = sub(p2, "  int y_idx = blockIdx.y;\n",
         "  signed_word accl[kPerThreadElems];\n"
         "  for (int i = 0; i < kPerThreadElems; i++) accl[i] = 0;\n"
         "  for (int item = blockIdx.y * items; item < (blockIdx.y + 1) * items; ++item) {\n"
         "  int y_idx = item;\n")
p2 = sub(p2, "  signed_word *dst_ptr = dst_limb + (x_idx << kStageMerging);\n"
             "  basic::VectorizedMove<signed_word, kPerThreadElems>(dst_ptr, local);\n}\n",
         "  {   // multiply-accumulate epilogue: acc += local . a^_i . a^_j at the positions this thread would have stored\n"
         "    int pi = item / c, pj = item % c;\n"
         "    if (pairs) {   // item k enumerates the unordered pairs i <= j (the SPUs accumulated g_ij + g_ji into one buffer)\n"
         "      int base = 0; pi = 0; while (item >= base + (c - pi)) { base += c - pi; pi++; } pj = pi + (item - base);\n"
         "    }\n"
         "    const word *ai = ahat + ((size_t) pi << log_degree) + ((size_t) x_idx << kStageMerging);\n"
         "    const word *aj = ahat + ((size_t) pj << log_degree) + ((size_t) x_idx << kStageMerging);\n"
         "    // bhat: the products a^_i . a^_j precomputed once per public a (one per item): one read and one multiply\n"
         "    const word *bij = bhat ? bhat + ((size_t) item << log_degree) + ((size_t) x_idx << kStageMerging) : nullptr;\n"
         "    for (int i = 0; i < kPerThreadElems; i++) {\n"
         "      word v;\n"
         "      if (bij) v = ::OPERATOR_GPU<word>::mult((word) local[i], bij[i], mod);\n"
         "      else { v = ::OPERATOR_GPU<word>::mult((word) local[i], ai[i], mod); v = ::OPERATOR_GPU<word>::mult(v, aj[i], mod); }\n"
         "      accl[i] = (signed_word) ::OPERATOR_GPU<word>::add((word) accl[i], v, mod);\n"
         "    }\n"
         "  }\n"
         "  __syncthreads();   // the next item reuses the shared-memory exchange buffer\n"
         "  }\n"
         "  const size_t out = ((size_t) blockIdx.y << log_degree) + ((size_t) (blockIdx.x * blockDim.x + threadIdx.x) << kStageMerging);\n"
         "  basic::VectorizedMove<signed_word, kPerThreadElems>(reinterpret_cast<signed_word *>(acc) + out, accl);\n}\n")

# ---- NTTPhase1ContigIn ----
p1 = func("NTTPhase1", "__global__ void NTTPhase1(")
p1 = sub(p1, "__global__ void NTTPhase1(", "__global__ void NTTPhase1ContigIn(")
head, tail = p1.split("  int dst_idx =", 1)
head = sub(head, "(stage_group_idx << (log_degree - kNumStages))", "stage_group_idx")
head = sub(head, "(blockIdx.x << kLogWarpBatching)", "(blockIdx.x << (kNumStages + kLogWarpBatching))")
head = sub(head, "batch_idx +", "(batch_idx << kNumStages) +")
head = sub(head, "(i << (log_degree - kStageMerging))", "(i << (kNumStages - kStageMerging))")
p1 = head + "  int dst_idx =" + tail
assert "dst_limb[dst_idx + (i << (log_degree - kNumStages))]" in p1      # the store is Cheddar's strided store

out = ("#pragma once\n#include \"gen/cheddar_ntt_kernels.cuh\"\n#include \"gpuntt/common/modular_arith.cuh\"\n"
       "namespace cheddar { namespace kernel {\n" + p2 + "\n" + p1 + "} }\n")
open(sys.argv[2], "w").write(out)
print(f"{sys.argv[2]}: NTTPhase2Acc and NTTPhase1ContigIn written, {n[0]} edits applied")
