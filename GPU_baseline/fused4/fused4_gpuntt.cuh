// fused4 on GPU-NTT's own kernels: the same four-step as fused4_ntt.cuh, but every
// sub-transform runs the GPU-NTT merge butterfly kernels (identical arithmetic
// and tiling to the merge backend it is compared against), with the four-step
// factors fused in:
//   * forward row transform: the twiddle is applied in the first kernel's load;
//   * inverse row transform: the library's final per-element n^{-1} multiply is
//     replaced by a per-element table (twiddle^{-1} * N^{-1}) -- no extra multiply;
//   * column transforms: unchanged library kernels (psi absorbed via X_N_plus).
// Transposes stay separate: they are the DRU lane.
//
// The kernels below are GPU-NTT's ForwardCore / InverseCore / InverseCoreLowRing
// (Alisah Ozcan, https://github.com/Alisah-Ozcan/GPU-NTT, Apache-2.0; paper
// https://eprint.iacr.org/2023/1410), reduced to the unsigned 64-bit path, with
// the fused table multiplies added. The butterfly schedule, indexing and launch
// configurations (CreateForwardNTTKernel / CreateInverseNTTKernel) are unchanged.

#ifndef GPU_BASELINE_FUSED4_GPUNTT_CUH__
#define GPU_BASELINE_FUSED4_GPUNTT_CUH__

#include "fused4/fused4_ntt.cuh"

#include "gpuntt/common/nttparameters.cuh"

namespace fused4g {

using namespace gpuntt;
using TU = Data64;

// ForwardCore with an optional fused pre-multiply on the global load.
// pre row index = (row_offset + blockIdx.z) % rows_per_poly.
template <bool PRE>
__global__ void fwd_core(TU* polynomial_in, TU* polynomial_out,
                         const Root<TU>* __restrict__ root_of_unity_table,
                         Modulus<TU> modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power, bool not_last_kernel,
                         const TU* __restrict__ pre, int row_offset, int rows_per_poly) {
    const int idx_x = threadIdx.x, idx_y = threadIdx.y;
    const int block_x = blockIdx.x, block_y = blockIdx.y, block_z = blockIdx.z;
    extern __shared__ char shared_memory_typed[];
    TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
    const Modulus<TU> modulus_reg = modulus;

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t) 1 << logm;
    location_t global_addresss =
        idx_x + (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (2 * block_y * offset) +
        (location_t) (block_z << N_power);
    location_t omega_addresss =
        idx_x + (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    TU v0 = polynomial_in[global_addresss];
    TU v1 = polynomial_in[global_addresss + offset];
    if (PRE) {
        const TU* prow = pre + (static_cast<location_t>((row_offset + block_z) % rows_per_poly) << N_power);
        const location_t in_row = global_addresss & ((location_t(1) << N_power) - 1);
        v0 = OPERATOR_GPU<TU>::mult(v0, prow[in_row], modulus_reg);
        v1 = OPERATOR_GPU<TU>::mult(v1, prow[in_row + offset], modulus_reg);
    }
    shared_memory[shared_addresss] = v0;
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] = v1;

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if (not_last_kernel) {
#pragma unroll
        for (int lp = 0; lp < outer_iteration_count; lp++) {
            __syncthreads();
            current_root_index = m + (omega_addresss >> t_2);
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus_reg);
            t = t >> 1; t_2 -= 1; t_ -= 1; m <<= 1;
            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    } else {
#pragma unroll
        for (int lp = 0; lp < (shared_index - 5); lp++) {
            __syncthreads();
            current_root_index = m + (omega_addresss >> t_2);
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus_reg);
            t = t >> 1; t_2 -= 1; t_ -= 1; m <<= 1;
            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
#pragma unroll
        for (int lp = 0; lp < 6; lp++) {
            current_root_index = m + (omega_addresss >> t_2);
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus_reg);
            t = t >> 1; t_2 -= 1; t_ -= 1; m <<= 1;
            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }
    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] = shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

// InverseCore; in the last kernel the constant n^{-1} multiply becomes a table
// multiply when POST (same multiply count as the library).
template <bool POST>
__global__ void inv_core(TU* polynomial_in, TU* polynomial_out,
                         const Root<TU>* __restrict__ inverse_root_of_unity_table,
                         Modulus<TU> modulus, int shared_index, int logm, int k,
                         int outer_iteration_count, int N_power, Ninverse<TU> n_inverse,
                         bool last_kernel, const TU* __restrict__ post, int row_offset,
                         int rows_per_poly) {
    const int idx_x = threadIdx.x, idx_y = threadIdx.y;
    const int block_x = blockIdx.x, block_y = blockIdx.y, block_z = blockIdx.z;
    extern __shared__ char shared_memory_typed[];
    TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
    const Modulus<TU> modulus_reg = modulus;
    const Ninverse<TU> n_inverse_reg = n_inverse;

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t) 1 << logm;
    location_t global_addresss =
        idx_x + (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (2 * block_y * offset) +
        (location_t) (block_z << N_power);
    location_t omega_addresss =
        idx_x + (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] = polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for (int lp = 0; lp < loops; lp++) {
        __syncthreads();
        current_root_index = m + (omega_addresss >> t_2);
        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus_reg);
        t = t << 1; t_2 += 1; t_ += 1; m >>= 1;
        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (last_kernel) {
        TU f0 = n_inverse_reg, f1 = n_inverse_reg;
        if (POST) {
            const TU* prow = post + (static_cast<location_t>((row_offset + block_z) % rows_per_poly) << N_power);
            const location_t in_row = global_addresss & ((location_t(1) << N_power) - 1);
            f0 = prow[in_row];
            f1 = prow[in_row + offset];
        }
        polynomial_out[global_addresss] = OPERATOR_GPU<TU>::mult(shared_memory[shared_addresss], f0, modulus_reg);
        polynomial_out[global_addresss + offset] =
            OPERATOR_GPU<TU>::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)], f1, modulus_reg);
    } else {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] = shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

// InverseCoreLowRing (n <= 2^10); final n^{-1} multiply replaced by the table when POST.
template <bool POST>
__global__ void inv_lowring(TU* polynomial_in, TU* polynomial_out,
                            const Root<TU>* __restrict__ inverse_root_of_unity_table,
                            Modulus<TU> modulus, int shared_index, int N_power,
                            Ninverse<TU> n_inverse, int total_batch, const TU* __restrict__ post,
                            int rows_per_poly) {
    const int idx_x = threadIdx.x, idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int batch_index = (block_x * blockDim.y) + idx_y;
    const bool active_batch = (batch_index < total_batch);
    const int batch_index_safe = active_batch ? batch_index : 0;
    extern __shared__ char shared_memory_typed[];
    TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
    const Modulus<TU> modulus_reg = modulus;

    int t_2 = 0, t_ = 0;
    int offset = idx_y << N_power;
    int loops = N_power;
    int m = (int) 1 << (N_power - 1);
    const int half_n = 1 << (N_power - 1);
    const int row_base = idx_y << N_power;
    const int shared_address0 = row_base + idx_x;
    const int shared_address1 = shared_address0 + half_n;
    location_t global_addresss = idx_x + (location_t) (batch_index_safe << N_power);
    location_t omega_addresss = idx_x;

    shared_memory[shared_address0] = polynomial_in[global_addresss];
    shared_memory[shared_address1] = polynomial_in[global_addresss + half_n];
    int shared_addresss = idx_x;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    __syncthreads();
#pragma unroll
    for (int lp = 0; lp < loops; lp++) {
        int group_in_shared_address = in_shared_address + offset;
        current_root_index = m + (omega_addresss >> t_2);
        GentlemanSandeUnit(shared_memory[group_in_shared_address], shared_memory[group_in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus_reg);
        t = t << 1; t_2 += 1; t_ += 1; m >>= 1;
        if (lp + 1 < loops) in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        __syncthreads();
    }
    __syncthreads();
    TU f0 = n_inverse, f1 = n_inverse;
    if (POST) {
        const TU* prow = post + (static_cast<location_t>(batch_index_safe % rows_per_poly) << N_power);
        f0 = prow[idx_x];
        f1 = prow[idx_x + half_n];
    }
    TU o0 = OPERATOR_GPU<TU>::mult(shared_memory[shared_address0], f0, modulus_reg);
    TU o1 = OPERATOR_GPU<TU>::mult(shared_memory[shared_address1], f1, modulus_reg);
    if (active_batch) {
        polynomial_out[global_addresss] = o0;
        polynomial_out[global_addresss + half_n] = o1;
    }
}

// One negacyclic sub-transform length 2^logn (X_N_plus with a given 2n-th root),
// launched exactly as GPU-NTT's GPU_NTT / GPU_INTT would for n >= 2^10.
struct SubNTT {
    int logn;
    Modulus<TU> mod;
    Root<TU>* fwd = nullptr;
    Root<TU>* inv = nullptr;
    std::vector<KernelConfig> fcfg, icfg;
    static constexpr int kMaxZ = 65535;

    SubNTT(int lg, uint64_t p, uint64_t psi) : logn(lg), mod(p) {
        if (lg < 10 || lg > 12) throw std::runtime_error("fused4g: sub-transform length must be 2^10..2^12");
        const uint64_t omega = pcg_cuda::mod_mul_u64(psi, psi, p);
        NTTFactors<TU> f(Modulus<TU>(p), omega, psi);
        NTTParameters<TU> params(lg, f, ReductionPolynomial::X_N_plus);
        auto hf = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
        auto hi = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);
        const size_t bytes = static_cast<size_t>(params.root_of_unity_size) * sizeof(Root<TU>);
        fused4::check(cudaMalloc(&fwd, bytes), "cudaMalloc sub fwd");
        fused4::check(cudaMalloc(&inv, bytes), "cudaMalloc sub inv");
        fused4::check(cudaMemcpy(fwd, hf.data(), bytes, cudaMemcpyHostToDevice), "memcpy sub fwd");
        fused4::check(cudaMemcpy(inv, hi.data(), bytes, cudaMemcpyHostToDevice), "memcpy sub inv");
        fcfg = CreateForwardNTTKernel<TU>()[lg];
        icfg = CreateInverseNTTKernel<TU>()[lg];
    }
    ~SubNTT() { cudaFree(fwd); cudaFree(inv); }
    SubNTT(const SubNTT&) = delete;
    SubNTT& operator=(const SubNTT&) = delete;

    // rows contiguous rows of length 2^logn; pre (optional) fused into the first kernel.
    void forward(TU* data, int rows, const TU* pre, int rows_per_poly) const {
        for (int off = 0; off < rows; off += kMaxZ) {
            const int r = std::min(kMaxZ, rows - off);
            TU* base = data + (static_cast<size_t>(off) << logn);
            for (size_t i = 0; i < fcfg.size(); i++) {
                const auto& c = fcfg[i];
                dim3 grid(c.griddim_x, c.griddim_y, r), block(c.blockdim_x, c.blockdim_y);
                if (i == 0 && pre)
                    fwd_core<true><<<grid, block, c.shared_memory>>>(base, base, fwd, mod, c.shared_index, c.logm,
                        c.outer_iteration_count, logn, c.not_last_kernel, pre, off, rows_per_poly);
                else
                    fwd_core<false><<<grid, block, c.shared_memory>>>(base, base, fwd, mod, c.shared_index, c.logm,
                        c.outer_iteration_count, logn, c.not_last_kernel, nullptr, off, 1);
                fused4::check(cudaGetLastError(), "fwd_core");
            }
        }
    }
    // post (optional) replaces the final n^{-1}; without it the scale is 1 (no scaling).
    void inverse(TU* data, int rows, const TU* post, int rows_per_poly) const {
        const Ninverse<TU> one = 1;
        if (logn == 10) {
            const auto& c = icfg[0];
            dim3 grid((rows + c.blockdim_y - 1) / c.blockdim_y, 1, 1), block(c.blockdim_x, c.blockdim_y);
            if (post)
                inv_lowring<true><<<grid, block, c.shared_memory>>>(data, data, inv, mod, c.shared_index, logn,
                                                                    one, rows, post, rows_per_poly);
            else
                inv_lowring<false><<<grid, block, c.shared_memory>>>(data, data, inv, mod, c.shared_index, logn,
                                                                     one, rows, nullptr, 1);
            fused4::check(cudaGetLastError(), "inv_lowring");
            return;
        }
        for (int off = 0; off < rows; off += kMaxZ) {
            const int r = std::min(kMaxZ, rows - off);
            TU* base = data + (static_cast<size_t>(off) << logn);
            for (size_t i = 0; i < icfg.size(); i++) {
                const auto& c = icfg[i];
                dim3 grid(c.griddim_x, c.griddim_y, r), block(c.blockdim_x, c.blockdim_y);
                if (c.not_last_kernel && post)
                    inv_core<true><<<grid, block, c.shared_memory>>>(base, base, inv, mod, c.shared_index, c.logm, c.k,
                        c.outer_iteration_count, logn, one, c.not_last_kernel, post, off, rows_per_poly);
                else
                    inv_core<false><<<grid, block, c.shared_memory>>>(base, base, inv, mod, c.shared_index, c.logm, c.k,
                        c.outer_iteration_count, logn, one, c.not_last_kernel, nullptr, off, 1);
                fused4::check(cudaGetLastError(), "inv_core");
            }
        }
    }
};

// Same pipeline and tables as fused4::Plan; sub-transforms on GPU-NTT's kernels.
struct PlanG {
    fused4::Plan base;          // tables (tw, twi), transposes, pointwise
    SubNTT col, row;            // col: length n1 (psi1), row: length n2 (psi2)

    PlanG(int logN, uint64_t p)
        : base(logN, p),
          col((logN + 1) / 2, p, pcg_cuda::mod_pow_u64(pcg_cuda::negacyclic_psi(p, 1 << logN), 1ull << (logN / 2), p)),
          row(logN / 2, p, pcg_cuda::mod_pow_u64(pcg_cuda::negacyclic_psi(p, 1 << logN), 1ull << ((logN + 1) / 2), p)) {}

    void multiply(TU* a, TU* b, TU* ta, TU* tb, int batch, int lanes = 3) const {
        const bool sm = lanes & 1, dru = lanes & 2;
        const int n1 = base.n1, n2 = base.n2;
        for (TU* x : {a, b}) {
            TU* tx = (x == a) ? ta : tb;
            if (dru) base.transpose(x, tx, n1, n2, batch);                 // T1
            if (sm)  col.forward(tx, batch * n2, nullptr, 1);              // K1
            if (dru) base.transpose(tx, x, n2, n1, batch);                 // T2
            if (sm)  row.forward(x, batch * n1, base.tw, n1);              // K2 (+twiddle)
        }
        if (sm)  base.pointwise(a, b, batch);                              // PW
        if (sm)  row.inverse(a, batch * n1, base.twi, n1);                 // K3 (+twiddle^-1 N^-1)
        if (dru) base.transpose(a, ta, n1, n2, batch);                     // T4
        if (sm)  col.inverse(ta, batch * n2, nullptr, 1);                  // K4
        if (dru) base.transpose(ta, a, n2, n1, batch);                     // T5
    }
};

}  // namespace fused4g

#endif  // GPU_BASELINE_FUSED4_GPUNTT_CUH__
