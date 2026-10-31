// Minimal stand-in for Cheddar's core/NTT.h (which pulls in rmm-backed DeviceVector/Parameter):
// only the enums, the element-function alias, the extended on-the-fly-twiddle flag and InputPtrList the NTT kernels use.
#pragma once
#include "core/Type.h"
namespace cheddar {
constexpr bool kExtendedOT = true;   // core/Parameter.h:20
namespace kernel {
template <typename word>
using elem_func_t = void (*)(make_signed_t<word> &, const make_signed_t<word>, const word, const word, const make_signed_t<word>);
}
enum class NTTType { NTT, INTT };
enum class Phase { Phase1, Phase2 };
template <typename word, int num_poly>
struct InputPtrList {   // common/PtrList.h:38, without the DeviceVector constructor
  const word *ptrs_[num_poly];
  int extra_ = 0;
  __host__ __device__ InputPtrList() { for (int i = 0; i < num_poly; i++) ptrs_[i] = nullptr; }
};
}  // namespace cheddar
