// =====================================================================================
// STAGE 3 · kernel_utils.hpp  --  shape checking shared by every EP
// =====================================================================================
// Every EP must agree on WHAT an operator means (its shapes and broadcasting rules); they
// differ only in HOW fast they compute it. So the "what" lives here, once:
//   * GemmArgs     : validated pointers/sizes for  Y = alpha * A x B + beta * C
//   * BinaryPlan   : how Add/Mul/Sub broadcast their two inputs
// =====================================================================================
#pragma once
#include "stage3_ep/execution_provider.hpp"

#include <cmath>
#include <stdexcept>

namespace mo {

// How the optional C (bias) of Gemm is broadcast onto the M x N output.
enum class BiasMode { None, Scalar, Row, Column, Full };

struct GemmArgs {
    const float *A = nullptr, *B = nullptr, *C = nullptr;
    float *Y = nullptr;
    int64_t M = 0, K = 0, N = 0;
    float alpha = 1.f, beta = 1.f;
    BiasMode mode = BiasMode::None;

    // beta*C value for output element (i, j). Used by scalar code and edge cases.
    float bias_at(int64_t i, int64_t j) const {
        switch (mode) {
        case BiasMode::None: return 0.f;
        case BiasMode::Scalar: return beta * C[0];
        case BiasMode::Row: return beta * C[j];
        case BiasMode::Column: return beta * C[i];
        case BiasMode::Full: return beta * C[i * N + j];
        }
        return 0.f;
    }
};

inline BiasMode bias_mode_for(const Tensor &c, int64_t M, int64_t N) {
    if (c.numel() == 1) return BiasMode::Scalar;
    if (c.rank() <= 2 && c.cols() == N && c.rows() == 1) return BiasMode::Row;          // {N} or {1,N}
    if (c.rank() == 2 && c.dim(0) == M && c.dim(1) == 1) return BiasMode::Column;       // {M,1}
    if (c.rank() == 2 && c.dim(0) == M && c.dim(1) == N) return BiasMode::Full;         // {M,N}
    throw std::runtime_error("Gemm: C of shape " + shape_str(c.shape()) + " cannot broadcast to [" +
                             std::to_string(M) + "," + std::to_string(N) + "]");
}

// MatMul(A, B): A may have any rank >= 2 (leading dims are flattened into rows), B must be 2-D.
inline Tensor prepare_matmul(const std::vector<const Tensor *> &in, GemmArgs &g) {
    if (in.size() < 2 || !in[0] || !in[1]) throw std::runtime_error("MatMul: needs 2 inputs");
    const Tensor &a = *in[0], &b = *in[1];
    if (a.rank() < 2 || b.rank() != 2 || a.cols() != b.dim(0))
        throw std::runtime_error("MatMul: incompatible shapes " + shape_str(a.shape()) + " x " + shape_str(b.shape()));
    Shape out_shape = a.shape();
    out_shape.back() = b.dim(1);
    Tensor y = Tensor::empty(out_shape);
    g.A = a.data(); g.B = b.data(); g.Y = y.data();
    g.M = a.rows(); g.K = a.cols(); g.N = b.dim(1);
    g.alpha = 1.f; g.beta = 0.f; g.mode = BiasMode::None;
    return y;
}

// Gemm(A, B, C?) with transA = transB = 0 (the fast EPs require this; the optimizer folds
// constant transposes away, and the CPU EP handles the general case itself).
inline Tensor prepare_gemm_notrans(const std::vector<const Tensor *> &in, const Attributes &at, GemmArgs &g) {
    if (in.size() < 2 || !in[0] || !in[1]) throw std::runtime_error("Gemm: needs at least 2 inputs");
    if (at.get_int("transA", 0) || at.get_int("transB", 0))
        throw std::runtime_error("Gemm: this kernel needs transA=transB=0");
    const Tensor &a = *in[0], &b = *in[1];
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(0))
        throw std::runtime_error("Gemm: incompatible shapes " + shape_str(a.shape()) + " x " + shape_str(b.shape()));
    Tensor y = Tensor::empty({a.dim(0), b.dim(1)});
    g.A = a.data(); g.B = b.data(); g.Y = y.data();
    g.M = a.dim(0); g.K = a.dim(1); g.N = b.dim(1);
    g.alpha = at.get_float("alpha", 1.f);
    g.beta = at.get_float("beta", 1.f);
    const Tensor *c = in.size() > 2 ? in[2] : nullptr;
    g.mode = c ? bias_mode_for(*c, g.M, g.N) : BiasMode::None;
    g.C = c ? c->data() : nullptr;
    return y;
}

// Rows of output handed to one parallel_for chunk: about 32K multiply-adds per chunk, a
// multiple of 4 (the SIMD kernels work on 4 rows at a time), and one single chunk when the
// whole product is too small to be worth waking threads.
constexpr int64_t kGemmParallelMinMacs = 1 << 21;      // ~2M multiply-adds; tune with bench_minionnx
inline int64_t gemm_row_grain(const GemmArgs &g) {
    const int64_t per_row = std::max<int64_t>(1, g.K * g.N);
    int64_t grain = std::max<int64_t>(4, 32768 / per_row);
    grain = (grain + 3) / 4 * 4;
    if (g.M * g.K * g.N < kGemmParallelMinMacs) grain = std::max<int64_t>(grain, g.M);
    return grain;
}

// ----------------------------------------------------------------- elementwise binary ops
enum class BinaryMode { Same, Row, Scalar };
struct BinaryPlan {
    const Tensor *big, *small;   // output has big's shape
    BinaryMode mode;
    bool swapped;                // true if inputs were swapped (matters for Sub)
};

inline bool is_row_of(const Tensor &small, const Tensor &big) {
    if (small.numel() != big.cols() || small.rank() > big.rank()) return false;
    for (int i = 0; i + 1 < small.rank(); i++) if (small.dim(i) != 1) return false;   // [1,...,1,N]
    return true;
}

inline BinaryPlan plan_binary(const char *op, const Tensor &a, const Tensor &b) {
    if (a.shape() == b.shape()) return {&a, &b, BinaryMode::Same, false};
    if (b.numel() == 1) return {&a, &b, BinaryMode::Scalar, false};
    if (a.numel() == 1) return {&b, &a, BinaryMode::Scalar, true};
    if (is_row_of(b, a)) return {&a, &b, BinaryMode::Row, false};
    if (is_row_of(a, b)) return {&b, &a, BinaryMode::Row, true};
    throw std::runtime_error(std::string(op) + ": unsupported broadcast " + shape_str(a.shape()) + " vs " +
                             shape_str(b.shape()));
}

constexpr int64_t kElemGrain = 16384;      // elements per parallel_for chunk for cheap ops

inline const Tensor &require(const std::vector<const Tensor *> &in, size_t i, const char *op) {
    if (i >= in.size() || !in[i]) throw std::runtime_error(std::string(op) + ": missing input " + std::to_string(i));
    return *in[i];
}

} // namespace mo
