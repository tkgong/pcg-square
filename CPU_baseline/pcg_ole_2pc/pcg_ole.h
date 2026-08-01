// PCG for OLE over R_p = Z_p[X]/(X^N+1), based on Ring-LPN with regular noise.
// Templated on the prime P so that field arithmetic uses FFp<P> with operator
// overloading:  FFp<P> a, b, c;  c = a * b + a.inv();

#ifndef PCG_OLE_2PC_PCG_OLE_H__
#define PCG_OLE_2PC_PCG_OLE_H__

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <emp-tool/io/net_io_channel.h>

#include "bool_circuit/adder.h"
#include "bool_circuit/circuit.h"
#include "common/ffp.h"
#include "common/ntt.h"
#include "common/timing.h"
#include "half_tree_dpf/dpf.h"

#ifdef PCG_ENABLE_CUDA
#include "common/poly_mul_cuda.h"
#endif

namespace pcg_ole {

// CUDA_V1           = hand-rolled radix-2 Montgomery NTT (common/poly_mul_cuda.cu)
// CUDA_GPUNTT_MERGE = GPU-NTT "merge" backend (transpose-free; default GPU path)
// CUDA_GPUNTT_4STEP = GPU-NTT "4-step" backend (transpose-heavy; only N in [4096, 2^24])
// AUTO              = merge if available, else 4-step, else v1, else CPU
enum class PolyMulBackend { CPU, CUDA_V1, CUDA_GPUNTT_MERGE, CUDA_GPUNTT_4STEP, AUTO };

inline std::string to_string(PolyMulBackend backend) {
    switch (backend) {
        case PolyMulBackend::CPU:               return "cpu";
        case PolyMulBackend::CUDA_V1:           return "cuda-v1";
        case PolyMulBackend::CUDA_GPUNTT_MERGE: return "cuda-gpuntt-merge";
        case PolyMulBackend::CUDA_GPUNTT_4STEP: return "cuda-gpuntt-4step";
        case PolyMulBackend::AUTO:              return "auto";
    }
    return "unknown";
}

inline PolyMulBackend parse_poly_mul_backend(const std::string& s) {
    if (s == "cpu") return PolyMulBackend::CPU;
    if (s == "cuda") return PolyMulBackend::CUDA_GPUNTT_MERGE;         // back-compat: default GPU
    if (s == "cuda-v1") return PolyMulBackend::CUDA_V1;
    if (s == "cuda-gpuntt") return PolyMulBackend::CUDA_GPUNTT_MERGE;  // back-compat alias for merge
    if (s == "cuda-gpuntt-merge") return PolyMulBackend::CUDA_GPUNTT_MERGE;
    if (s == "cuda-gpuntt-4step") return PolyMulBackend::CUDA_GPUNTT_4STEP;
    if (s == "auto") return PolyMulBackend::AUTO;
    throw std::runtime_error("unknown poly-mul backend: " + s);
}

// ---- DPF full-domain evaluation backend (half_tree_dpf::EvalBackend) ------

inline std::string to_string(half_tree_dpf::EvalBackend b) {
    return b == half_tree_dpf::EvalBackend::GPU ? "gpu" : "cpu";
}

inline half_tree_dpf::EvalBackend parse_dpf_backend(const std::string& s) {
    if (s == "cpu") return half_tree_dpf::EvalBackend::CPU;
    if (s == "gpu") return half_tree_dpf::EvalBackend::GPU;
    throw std::runtime_error("unknown dpf backend: " + s);
}

// ---- polynomial over R_p = FFp<P>[X] / (X^N + 1) -------------------------

template<uint64_t P>
using Poly = std::vector<FFp<P>>;

template<uint64_t P>
Poly<P> poly_zero(int N) { return Poly<P>(N, FFp<P>::zero()); }

template<uint64_t P>
Poly<P> poly_add(const Poly<P>& a, const Poly<P>& b) {
    Poly<P> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

template<uint64_t P>
Poly<P> poly_sub(const Poly<P>& a, const Poly<P>& b) {
    Poly<P> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

// Schoolbook O(N²) — fallback when P is not NTT-friendly.
template<uint64_t P>
Poly<P> poly_mul_schoolbook(const Poly<P>& a, const Poly<P>& b, int N) {
    Poly<P> r = poly_zero<P>(N);
    for (int i = 0; i < N; ++i) {
        if (a[i].is_zero()) continue;
        for (int j = 0; j < N; ++j) {
            if (b[j].is_zero()) continue;
            int k = i + j;
            FFp<P> v = a[i] * b[j];
            if (k < N) r[k] += v;
            else       r[k - N] -= v;  // X^N = -1
        }
    }
    return r;
}

// NTT-accelerated O(N log N) — requires (P-1) % (2N) == 0.
template<uint64_t P>
Poly<P> poly_mul_ntt(const Poly<P>& a, const Poly<P>& b, int N,
                     const NTT<P>& ntt) {
    Poly<P> r(N);
    ntt.poly_mul(r.data(), a.data(), b.data());
    return r;
}

// Default: use NTT if P is NTT-friendly for this N, else schoolbook.
template<uint64_t P>
Poly<P> poly_mul(const Poly<P>& a, const Poly<P>& b, int N) {
    if ((P - 1) % (2 * static_cast<uint64_t>(N)) == 0) {
        // Thread-local NTT to amortize twiddle precomputation.
        thread_local int cached_N = 0;
        thread_local NTT<P>* cached_ntt = nullptr;
        if (cached_N != N) {
            delete cached_ntt;
            cached_ntt = new NTT<P>(N);
            cached_N = N;
        }
        return poly_mul_ntt<P>(a, b, N, *cached_ntt);
    }
    return poly_mul_schoolbook<P>(a, b, N);
}

template<uint64_t P>
bool cuda_v1_poly_mul_supported(int N, std::string* reason = nullptr) {
#ifdef PCG_ENABLE_CUDA
    return pcg_cuda::poly_mul_supported(P, N, reason);
#else
    if (reason) *reason = "CUDA support was not enabled at build time";
    return false;
#endif
}

template<uint64_t P>
bool cuda_gpuntt_merge_poly_mul_supported(int N, std::string* reason = nullptr) {
#ifdef PCG_ENABLE_CUDA
    return pcg_cuda::poly_mul_gpuntt_supported(P, N, reason);
#else
    if (reason) *reason = "CUDA support was not enabled at build time";
    return false;
#endif
}

template<uint64_t P>
bool cuda_gpuntt_4step_poly_mul_supported(int N, std::string* reason = nullptr) {
#ifdef PCG_ENABLE_CUDA
    return pcg_cuda::poly_mul_gpuntt_4step_supported(P, N, reason);
#else
    if (reason) *reason = "CUDA support was not enabled at build time";
    return false;
#endif
}

// "Is any CUDA poly-mul path available for this N?" — used by tests to decide
// whether to skip when a non-CPU backend was requested.
template<uint64_t P>
bool cuda_poly_mul_supported(int N, std::string* reason = nullptr) {
    return cuda_gpuntt_merge_poly_mul_supported<P>(N, reason) ||
           cuda_gpuntt_4step_poly_mul_supported<P>(N, reason) ||
           cuda_v1_poly_mul_supported<P>(N, reason);
}

template<uint64_t P>
std::vector<Poly<P>> poly_mul_batch(const std::vector<Poly<P>>& lhs,
                                    const std::vector<Poly<P>>& rhs,
                                    int N,
                                    PolyMulBackend backend) {
    if (lhs.size() != rhs.size())
        throw std::runtime_error("poly_mul_batch: lhs/rhs batch size mismatch");

    // Resolve which concrete CUDA path to take: 0 = none (CPU), 1 = v1,
    // 2 = GPU-NTT merge, 3 = GPU-NTT 4-step. AUTO prefers merge, then 4-step,
    // then v1, then CPU (merge covers the widest N range, so it wins by default).
    std::string reason;
    int cuda_path = 0;
    if (backend == PolyMulBackend::CUDA_GPUNTT_MERGE) {
        if (cuda_gpuntt_merge_poly_mul_supported<P>(N, &reason)) cuda_path = 2;
        else throw std::runtime_error("GPU-NTT merge poly-mul requested but unavailable: " + reason);
    } else if (backend == PolyMulBackend::CUDA_GPUNTT_4STEP) {
        if (cuda_gpuntt_4step_poly_mul_supported<P>(N, &reason)) cuda_path = 3;
        else throw std::runtime_error("GPU-NTT 4-step poly-mul requested but unavailable: " + reason);
    } else if (backend == PolyMulBackend::CUDA_V1) {
        if (cuda_v1_poly_mul_supported<P>(N, &reason)) cuda_path = 1;
        else throw std::runtime_error("CUDA v1 poly-mul requested but unavailable: " + reason);
    } else if (backend == PolyMulBackend::AUTO) {
        if (cuda_gpuntt_merge_poly_mul_supported<P>(N, &reason)) cuda_path = 2;
        else if (cuda_gpuntt_4step_poly_mul_supported<P>(N, &reason)) cuda_path = 3;
        else if (cuda_v1_poly_mul_supported<P>(N, &reason)) cuda_path = 1;
    }

    const int batch = static_cast<int>(lhs.size());
    std::vector<Poly<P>> out(batch, Poly<P>(N));

#ifdef PCG_ENABLE_CUDA
    if (cuda_path != 0 && batch > 0) {
        PCG_PROFILE_SCOPE("poly_mul.cuda_batch_total");
        std::vector<uint64_t> a_raw(static_cast<size_t>(batch) * N);
        std::vector<uint64_t> b_raw(static_cast<size_t>(batch) * N);
        std::vector<uint64_t> c_raw(static_cast<size_t>(batch) * N);

        {
            PCG_PROFILE_SCOPE("poly_mul.cuda_pack");
            for (int k = 0; k < batch; ++k) {
                if (static_cast<int>(lhs[k].size()) != N ||
                    static_cast<int>(rhs[k].size()) != N)
                    throw std::runtime_error("poly_mul_batch: polynomial length mismatch");
                for (int i = 0; i < N; ++i) {
                    a_raw[static_cast<size_t>(k) * N + i] = lhs[k][i].val();
                    b_raw[static_cast<size_t>(k) * N + i] = rhs[k][i].val();
                }
            }
        }
        {
            PCG_PROFILE_SCOPE("poly_mul.cuda_device_api");
            if (cuda_path == 2)
                pcg_cuda::poly_mul_u64_gpuntt(c_raw.data(), a_raw.data(), b_raw.data(),
                                              batch, N, P, nullptr);
            else if (cuda_path == 3)
                pcg_cuda::poly_mul_u64_gpuntt_4step(c_raw.data(), a_raw.data(), b_raw.data(),
                                                    batch, N, P, nullptr);
            else
                pcg_cuda::poly_mul_u64(c_raw.data(), a_raw.data(), b_raw.data(),
                                       batch, N, P, nullptr);
        }
        {
            PCG_PROFILE_SCOPE("poly_mul.cuda_unpack");
            for (int k = 0; k < batch; ++k)
                for (int i = 0; i < N; ++i)
                    out[k][i] = FFp<P>(c_raw[static_cast<size_t>(k) * N + i]);
        }
        return out;
    }
#endif

    {
        PCG_PROFILE_SCOPE("poly_mul.cpu_batch_total");
        for (int k = 0; k < batch; ++k)
            out[k] = poly_mul<P>(lhs[k], rhs[k], N);
    }
    return out;
}

template<uint64_t P>
Poly<P> poly_mul_backend(const Poly<P>& a, const Poly<P>& b, int N,
                         PolyMulBackend backend) {
    std::vector<Poly<P>> lhs{a};
    std::vector<Poly<P>> rhs{b};
    auto out = poly_mul_batch<P>(lhs, rhs, N, backend);
    return std::move(out[0]);
}

template<uint64_t P>
Poly<P> poly_scale(const Poly<P>& a, FFp<P> s) {
    Poly<P> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] * s;
    return r;
}

// ---- PCG parameters -------------------------------------------------------

struct PCGParams {
    int N;      // ring degree (power of 2)
    int c;      // compression factor (typically 2)
    int t;      // noise weight per sparse polynomial
    int w;      // DPF branching width (m = 2^w)
    half_tree_dpf::PRGType prg_type = half_tree_dpf::PRGType::CHACHA8;
    PolyMulBackend poly_mul_backend = PolyMulBackend::CPU;
    // GPU requires PCG_ENABLE_CUDA (AES and ChaCha8 both have device PRGs);
    // otherwise falls back to CPU.
    half_tree_dpf::EvalBackend dpf_eval_backend = half_tree_dpf::EvalBackend::CPU;
};

// ---- OLE output -----------------------------------------------------------

template<uint64_t P>
struct OLEOutput {
    Poly<P> x;   // party's x share ∈ R_p
    Poly<P> z;   // party's z share, s.t. z_0 + z_1 = x_0 · x_1
};

// ---- PCG for OLE (templated on prime P) -----------------------------------

template<uint64_t P>
class PCG_OLE {
public:
    PCG_OLE(int party, emp::NetIO* io, const PCGParams& params);

    OLEOutput<P> gen_and_expand(bool_circuit::TripleGen& tg,
                                const std::vector<Poly<P>>& public_a);

private:
    int          party_;
    emp::NetIO*  io_;
    PCGParams    params_;
};

}  // namespace pcg_ole

// Template implementation included from header.
#include "pcg_ole_2pc/pcg_ole_impl.h"

#endif  // PCG_OLE_2PC_PCG_OLE_H__
