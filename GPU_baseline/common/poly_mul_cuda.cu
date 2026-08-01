#include "common/poly_mul_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace pcg_cuda {
namespace {

constexpr uint64_t kPrime = 4611686018326724609ULL;
constexpr int kMaxSupportedN = 32768;

void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::ostringstream os;
    os << what << ": " << cudaGetErrorString(err);
    throw std::runtime_error(os.str());
}

bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int log2_int(int n) {
    int out = 0;
    while (n > 1) {
        n >>= 1;
        ++out;
    }
    return out;
}

uint64_t mod_mul_host(uint64_t a, uint64_t b, uint64_t p) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) % p);
}

uint64_t mod_pow_host(uint64_t base, uint64_t exp, uint64_t p) {
    uint64_t out = 1;
    while (exp) {
        if (exp & 1) out = mod_mul_host(out, base, p);
        base = mod_mul_host(base, base, p);
        exp >>= 1;
    }
    return out;
}

uint64_t mod_inv_host(uint64_t x, uint64_t p) {
    return mod_pow_host(x, p - 2, p);
}

uint64_t find_primitive_root(uint64_t p) {
    const uint64_t phi = p - 1;
    uint64_t x = phi;
    std::vector<uint64_t> factors;
    for (uint64_t d = 2; d * d <= x; ++d) {
        if (x % d != 0) continue;
        factors.push_back(d);
        while (x % d == 0) x /= d;
    }
    if (x > 1) factors.push_back(x);

    for (uint64_t g = 2; g < p; ++g) {
        bool ok = true;
        for (uint64_t q : factors) {
            if (mod_pow_host(g, phi / q, p) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) return g;
    }
    throw std::runtime_error("failed to find primitive root");
}

int64_t inv_mod_base_host(uint64_t q) {
    uint64_t inv = 1;
    for (int bits = 1; bits < 64; bits <<= 1) {
        inv *= 2 - q * inv;
    }
    return static_cast<int64_t>(inv);
}

uint64_t to_montgomery_host(uint64_t x, uint64_t p) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(x) << 64) % p);
}

template <typename T>
void copy_to_device(T*& dst, const std::vector<T>& src) {
    check(cudaMalloc(reinterpret_cast<void**>(&dst), sizeof(T) * src.size()), "cudaMalloc table");
    check(cudaMemcpy(dst, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice),
          "cudaMemcpy table");
}

__device__ __forceinline__ int64_t signed_mul_hi64(int64_t a, int64_t b) {
    uint64_t au = static_cast<uint64_t>(a);
    uint64_t bu = static_cast<uint64_t>(b);
    int64_t hi = static_cast<int64_t>(__umul64hi(au, bu));
    if (a < 0) hi -= static_cast<int64_t>(bu);
    if (b < 0) hi -= static_cast<int64_t>(au);
    return hi;
}

__device__ __forceinline__ uint64_t mont_mul(uint64_t a, uint64_t b,
                                             uint64_t q, int64_t q_inv) {
    int64_t as = static_cast<int64_t>(a);
    int64_t bs = static_cast<int64_t>(b);
    uint64_t lo = static_cast<uint64_t>(as) * static_cast<uint64_t>(bs);
    int64_t hi = signed_mul_hi64(as, bs);
    int64_t temp = static_cast<int64_t>(lo * static_cast<uint64_t>(q_inv));
    int64_t temp_hi = signed_mul_hi64(temp, static_cast<int64_t>(q));
    int64_t res = hi - temp_hi;
    if (res < 0) res += static_cast<int64_t>(q);
    if (static_cast<uint64_t>(res) >= q) res -= static_cast<int64_t>(q);
    return static_cast<uint64_t>(res);
}

__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t s = a + b;
    return s >= q ? s - q : s;
}

__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t q) {
    return a >= b ? a - b : a + q - b;
}

__device__ __forceinline__ uint32_t bit_reverse_device(uint32_t x, int bits) {
    return __brev(x) >> (32 - bits);
}

__global__ void pack_twist_bitrev_kernel(const uint64_t* in, uint64_t* out,
                                         const uint64_t* twist_mont,
                                         uint64_t mont_converter,
                                         int total, int N, int logN,
                                         uint64_t q, int64_t q_inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i = idx % N;
    int batch = idx / N;
    int br = static_cast<int>(bit_reverse_device(static_cast<uint32_t>(i), logN));
    uint64_t x = in[idx] % q;
    uint64_t xm = mont_mul(x, mont_converter, q, q_inv);
    out[batch * N + br] = mont_mul(xm, twist_mont[i], q, q_inv);
}

__global__ void bitrev_copy_kernel(const uint64_t* in, uint64_t* out,
                                   int total, int N, int logN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i = idx % N;
    int batch = idx / N;
    int br = static_cast<int>(bit_reverse_device(static_cast<uint32_t>(i), logN));
    out[batch * N + br] = in[idx];
}

__global__ void ntt_stage_kernel(uint64_t* a, const uint64_t* roots_mont,
                                 int batch_count, int N, int len,
                                 uint64_t q, int64_t q_inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * (N >> 1);
    if (idx >= total) return;

    int half = len >> 1;
    int per_batch = N >> 1;
    int batch = idx / per_batch;
    int rem = idx % per_batch;
    int group = rem / half;
    int j = rem - group * half;
    int base = batch * N + group * len + j;
    int root_idx = j * (N / len);

    uint64_t u = a[base];
    uint64_t v = mont_mul(a[base + half], roots_mont[root_idx], q, q_inv);
    a[base] = add_mod(u, v, q);
    a[base + half] = sub_mod(u, v, q);
}

__global__ void pointwise_mul_kernel(const uint64_t* a, const uint64_t* b,
                                     uint64_t* out, int total,
                                     uint64_t q, int64_t q_inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = mont_mul(a[idx], b[idx], q, q_inv);
}

__global__ void post_scale_kernel(const uint64_t* in, uint64_t* out,
                                  const uint64_t* post_mont, int total, int N,
                                  uint64_t q, int64_t q_inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i = idx % N;
    uint64_t scaled = mont_mul(in[idx], post_mont[i], q, q_inv);
    out[idx] = mont_mul(scaled, 1, q, q_inv);
}

struct CudaPolyMulPlan {
    explicit CudaPolyMulPlan(int degree) : N(degree), logN(log2_int(degree)) {
        const uint64_t g = find_primitive_root(kPrime);
        const uint64_t psi = mod_pow_host(g, (kPrime - 1) / (2 * static_cast<uint64_t>(N)), kPrime);
        const uint64_t psi_inv = mod_inv_host(psi, kPrime);
        const uint64_t omega = mod_mul_host(psi, psi, kPrime);
        const uint64_t omega_inv = mod_mul_host(psi_inv, psi_inv, kPrime);
        const uint64_t inv_N = mod_inv_host(static_cast<uint64_t>(N), kPrime);

        q_inv = inv_mod_base_host(kPrime);
        const uint64_t one_mont = to_montgomery_host(1, kPrime);
        mont_converter = to_montgomery_host(one_mont, kPrime);

        std::vector<uint64_t> psi_pow(N), post(N), root_fwd(N / 2), root_inv(N / 2);
        uint64_t cur = 1;
        for (int i = 0; i < N; ++i) {
            psi_pow[i] = to_montgomery_host(cur, kPrime);
            cur = mod_mul_host(cur, psi, kPrime);
        }
        cur = inv_N;
        for (int i = 0; i < N; ++i) {
            post[i] = to_montgomery_host(cur, kPrime);
            cur = mod_mul_host(cur, psi_inv, kPrime);
        }
        cur = 1;
        for (int i = 0; i < N / 2; ++i) {
            root_fwd[i] = to_montgomery_host(cur, kPrime);
            cur = mod_mul_host(cur, omega, kPrime);
        }
        cur = 1;
        for (int i = 0; i < N / 2; ++i) {
            root_inv[i] = to_montgomery_host(cur, kPrime);
            cur = mod_mul_host(cur, omega_inv, kPrime);
        }

        copy_to_device(d_psi, psi_pow);
        copy_to_device(d_post, post);
        copy_to_device(d_root_fwd, root_fwd);
        copy_to_device(d_root_inv, root_inv);
    }

    ~CudaPolyMulPlan() {
        cudaFree(d_psi);
        cudaFree(d_post);
        cudaFree(d_root_fwd);
        cudaFree(d_root_inv);
        cudaFree(d_in_a);
        cudaFree(d_in_b);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_tmp);
        cudaFree(d_out);
    }

    void ensure_capacity(int batch) {
        if (batch <= capacity_batch) return;
        cudaFree(d_in_a);
        cudaFree(d_in_b);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_tmp);
        cudaFree(d_out);
        size_t elems = static_cast<size_t>(batch) * static_cast<size_t>(N);
        size_t bytes = elems * sizeof(uint64_t);
        check(cudaMalloc(reinterpret_cast<void**>(&d_in_a), bytes), "cudaMalloc d_in_a");
        check(cudaMalloc(reinterpret_cast<void**>(&d_in_b), bytes), "cudaMalloc d_in_b");
        check(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes), "cudaMalloc d_a");
        check(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes), "cudaMalloc d_b");
        check(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes), "cudaMalloc d_c");
        check(cudaMalloc(reinterpret_cast<void**>(&d_tmp), bytes), "cudaMalloc d_tmp");
        check(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes), "cudaMalloc d_out");
        capacity_batch = batch;
    }

    void forward(uint64_t* d_input, uint64_t* d_work, int batch) {
        const int total = batch * N;
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        pack_twist_bitrev_kernel<<<grid, block>>>(d_input, d_work, d_psi, mont_converter,
                                                  total, N, logN, kPrime, q_inv);
        check(cudaGetLastError(), "launch pack_twist_bitrev_kernel");
        for (int len = 2; len <= N; len <<= 1) {
            int butterflies = batch * (N >> 1);
            dim3 sgrid((butterflies + block.x - 1) / block.x);
            ntt_stage_kernel<<<sgrid, block>>>(d_work, d_root_fwd, batch, N, len,
                                               kPrime, q_inv);
            check(cudaGetLastError(), "launch ntt_stage_kernel forward");
        }
    }

    void inverse(uint64_t* d_input, uint64_t* d_output, int batch) {
        const int total = batch * N;
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        bitrev_copy_kernel<<<grid, block>>>(d_input, d_tmp, total, N, logN);
        check(cudaGetLastError(), "launch bitrev_copy_kernel");
        for (int len = 2; len <= N; len <<= 1) {
            int butterflies = batch * (N >> 1);
            dim3 sgrid((butterflies + block.x - 1) / block.x);
            ntt_stage_kernel<<<sgrid, block>>>(d_tmp, d_root_inv, batch, N, len,
                                               kPrime, q_inv);
            check(cudaGetLastError(), "launch ntt_stage_kernel inverse");
        }
        post_scale_kernel<<<grid, block>>>(d_tmp, d_output, d_post, total, N,
                                           kPrime, q_inv);
        check(cudaGetLastError(), "launch post_scale_kernel");
    }

    void multiply(uint64_t* out, const uint64_t* a, const uint64_t* b,
                  int batch, PolyMulStats* stats) {
        ensure_capacity(batch);
        const size_t elems = static_cast<size_t>(batch) * static_cast<size_t>(N);
        const size_t bytes = elems * sizeof(uint64_t);
        check(cudaMemcpy(d_in_a, a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy a");
        check(cudaMemcpy(d_in_b, b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy b");

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (stats) {
            check(cudaEventCreate(&start), "cudaEventCreate start");
            check(cudaEventCreate(&stop), "cudaEventCreate stop");
            check(cudaEventRecord(start), "cudaEventRecord start");
        }

        forward(d_in_a, d_a, batch);
        forward(d_in_b, d_b, batch);
        dim3 block(256);
        dim3 grid((static_cast<unsigned>(elems) + block.x - 1) / block.x);
        pointwise_mul_kernel<<<grid, block>>>(d_a, d_b, d_c, static_cast<int>(elems),
                                              kPrime, q_inv);
        check(cudaGetLastError(), "launch pointwise_mul_kernel");
        inverse(d_c, d_out, batch);

        if (stats) {
            check(cudaEventRecord(stop), "cudaEventRecord stop");
            check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
            stats->device_ms = ms;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        } else {
            check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }

        check(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy out");
    }

    int N;
    int logN;
    int64_t q_inv = 0;
    uint64_t mont_converter = 0;
    int capacity_batch = 0;
    uint64_t *d_psi = nullptr, *d_post = nullptr, *d_root_fwd = nullptr, *d_root_inv = nullptr;
    uint64_t *d_in_a = nullptr, *d_in_b = nullptr, *d_a = nullptr, *d_b = nullptr;
    uint64_t *d_c = nullptr, *d_tmp = nullptr, *d_out = nullptr;
};

std::mutex& cache_mutex() {
    static std::mutex m;
    return m;
}

std::map<int, std::unique_ptr<CudaPolyMulPlan>>& plan_cache() {
    static std::map<int, std::unique_ptr<CudaPolyMulPlan>> cache;
    return cache;
}

CudaPolyMulPlan& get_plan(int N) {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto& cache = plan_cache();
    auto it = cache.find(N);
    if (it == cache.end()) {
        it = cache.emplace(N, std::unique_ptr<CudaPolyMulPlan>(new CudaPolyMulPlan(N))).first;
    }
    return *it->second;
}

}  // namespace

bool is_cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool poly_mul_supported(uint64_t prime, int N, std::string* reason) {
    if (prime != kPrime) {
        if (reason) *reason = "CUDA backend only supports the project 62-bit prime";
        return false;
    }
    if (!is_power_of_two(N) || N < 2 || N > kMaxSupportedN) {
        if (reason) *reason = "CUDA backend requires power-of-two N in [2,32768]";
        return false;
    }
    if ((prime - 1) % (2 * static_cast<uint64_t>(N)) != 0) {
        if (reason) *reason = "prime is not NTT-friendly for this N";
        return false;
    }
    if (!is_cuda_available()) {
        if (reason) *reason = "no CUDA device is available";
        return false;
    }
    if (reason) reason->clear();
    return true;
}

std::string device_name() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return "unavailable";
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return "unavailable";
    return prop.name;
}

std::string runtime_version() {
    int version = 0;
    if (cudaRuntimeGetVersion(&version) != cudaSuccess) return "unavailable";
    std::ostringstream os;
    os << (version / 1000) << "." << ((version % 1000) / 10);
    return os.str();
}

void poly_mul_u64(uint64_t* out, const uint64_t* a, const uint64_t* b,
                  int batch, int N, uint64_t prime, PolyMulStats* stats) {
    std::string reason;
    if (!poly_mul_supported(prime, N, &reason)) {
        throw std::runtime_error(reason);
    }
    if (batch <= 0) {
        throw std::runtime_error("batch must be positive");
    }
    get_plan(N).multiply(out, a, b, batch, stats);
}

}  // namespace pcg_cuda
