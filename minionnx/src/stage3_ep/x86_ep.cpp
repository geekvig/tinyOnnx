// =====================================================================================
// STAGE 3 · x86_ep.cpp  --  AVX2 + FMA Execution Provider ("x86")
// =====================================================================================
// SIMD = Single Instruction, Multiple Data. One AVX2 register (__m256) holds 8 floats, and
// one FMA instruction computes  c = a * b + c  for all 8 lanes at once.
//
//   scalar:  c[j] += a * b[j]            1 multiply-add per instruction
//   AVX2  :  c[j..j+7] += a * b[j..j+7]  8 multiply-adds per instruction (and 2 per cycle!)
//
// THE MICRO-KERNEL (the heart of every fast matrix multiply):
//   compute a 4 x 16 block of the output, keeping it in 8 registers for the whole K loop:
//
//        B row p:      [ b0 ......... b7 | b8 ........ b15 ]   (2 loads)
//        A column p:   a0, a1, a2, a3                          (4 broadcasts)
//        acc[r][h] += a_r * b_h                                (8 FMAs)
//
//   12 instructions do 64 multiply-adds, and the accumulators never touch memory until the
//   end. That "register blocking" is why this beats the reference kernel by several times.
//
// HOW WE COMPILE AVX2 CODE WITHOUT REQUIRING AVX2 EVERYWHERE:
//   __attribute__((target("avx2,fma"))) tells GCC to use AVX2 only inside these functions.
//   The rest of the program stays generic, and at run time available() checks the CPU
//   (__builtin_cpu_supports) before any AVX2 instruction is executed. An old CPU simply
//   never selects this EP -- no "Illegal instruction" crash.
//
// Ops: MatMul, Gemm(transA=transB=0), Add, Mul, Relu. Everything else falls back to "cpu".
// =====================================================================================
#include "stage3_ep/kernel_utils.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#define MO_X86_EP 1
#include <immintrin.h>
#endif

namespace mo {

#ifdef MO_X86_EP
namespace {

#define MO_AVX2 __attribute__((target("avx2,fma")))

// Load 8 values of beta*C for output row i, columns j..j+7 (according to the bias mode).
MO_AVX2 inline __m256 bias8(const GemmArgs &g, int64_t i, int64_t j) {
    const __m256 beta = _mm256_set1_ps(g.beta);
    switch (g.mode) {
    case BiasMode::None: return _mm256_setzero_ps();
    case BiasMode::Scalar: return _mm256_mul_ps(beta, _mm256_set1_ps(g.C[0]));
    case BiasMode::Row: return _mm256_mul_ps(beta, _mm256_loadu_ps(g.C + j));
    case BiasMode::Column: return _mm256_mul_ps(beta, _mm256_set1_ps(g.C[i]));
    case BiasMode::Full: return _mm256_mul_ps(beta, _mm256_loadu_ps(g.C + i * g.N + j));
    }
    return _mm256_setzero_ps();
}

// y = alpha * acc + beta*C, stored to row i, columns j..j+7
MO_AVX2 inline void store8(const GemmArgs &g, int64_t i, int64_t j, __m256 acc) {
    __m256 y = _mm256_fmadd_ps(_mm256_set1_ps(g.alpha), acc, bias8(g, i, j));
    _mm256_storeu_ps(g.Y + i * g.N + j, y);
}

// Scalar column tail (N % 8 columns) for `rows` rows starting at i.
inline void tail_cols(const GemmArgs &g, int64_t i, int rows, int64_t j0) {
    for (int r = 0; r < rows; r++)
        for (int64_t j = j0; j < g.N; j++) {
            float acc = 0.f;
            for (int64_t p = 0; p < g.K; p++) acc += g.A[(i + r) * g.K + p] * g.B[p * g.N + j];
            g.Y[(i + r) * g.N + j] = g.alpha * acc + g.bias_at(i + r, j);
        }
}

MO_AVX2 void gemm_rows_avx2(const GemmArgs &g, int64_t rb, int64_t re) {
    const int64_t K = g.K, N = g.N;
    const float *A = g.A, *B = g.B;
    int64_t i = rb;
    for (; i + 4 <= re; i += 4) {                                   // ---- 4-row strips
        const float *a0 = A + i * K, *a1 = a0 + K, *a2 = a1 + K, *a3 = a2 + K;
        int64_t j = 0;
        for (; j + 16 <= N; j += 16) {                              // 4 x 16 micro-kernel
            __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
            for (int64_t p = 0; p < K; p++) {
                const __m256 b0 = _mm256_loadu_ps(B + p * N + j);
                const __m256 b1 = _mm256_loadu_ps(B + p * N + j + 8);
                __m256 a = _mm256_broadcast_ss(a0 + p);
                c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
                a = _mm256_broadcast_ss(a1 + p);
                c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
                a = _mm256_broadcast_ss(a2 + p);
                c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
                a = _mm256_broadcast_ss(a3 + p);
                c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
            }
            store8(g, i, j, c00);     store8(g, i, j + 8, c01);
            store8(g, i + 1, j, c10); store8(g, i + 1, j + 8, c11);
            store8(g, i + 2, j, c20); store8(g, i + 2, j + 8, c21);
            store8(g, i + 3, j, c30); store8(g, i + 3, j + 8, c31);
        }
        for (; j + 8 <= N; j += 8) {                                // 4 x 8
            __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps(), c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
            for (int64_t p = 0; p < K; p++) {
                const __m256 b = _mm256_loadu_ps(B + p * N + j);
                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + p), b, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1 + p), b, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2 + p), b, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3 + p), b, c3);
            }
            store8(g, i, j, c0); store8(g, i + 1, j, c1); store8(g, i + 2, j, c2); store8(g, i + 3, j, c3);
        }
        if (j < N) tail_cols(g, i, 4, j);
    }
    for (; i < re; i++) {                                           // ---- leftover rows (M % 4)
        const float *a = A + i * K;
        int64_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 c = _mm256_setzero_ps();
            for (int64_t p = 0; p < K; p++) c = _mm256_fmadd_ps(_mm256_broadcast_ss(a + p), _mm256_loadu_ps(B + p * N + j), c);
            store8(g, i, j, c);
        }
        if (j < N) tail_cols(g, i, 1, j);
    }
}

void gemm_avx2(ThreadPool &pool, const GemmArgs &g) {
    pool.parallel_for(0, g.M, gemm_row_grain(g), [&](int64_t rb, int64_t re) { gemm_rows_avx2(g, rb, re); });
}

MO_AVX2 void relu_avx2(const float *x, float *o, int64_t lo, int64_t hi) {
    const __m256 z = _mm256_setzero_ps();
    int64_t i = lo;
    for (; i + 8 <= hi; i += 8) _mm256_storeu_ps(o + i, _mm256_max_ps(z, _mm256_loadu_ps(x + i)));
    for (; i < hi; i++) o[i] = x[i] > 0.f ? x[i] : 0.f;
}

// NOTE: a lambda inside a target("avx2") function does NOT inherit the attribute, so we
// use small attributed helper functions instead of lambdas here (GCC would refuse to inline
// the intrinsics: "target specific option mismatch").
MO_AVX2 inline __m256 vop(bool is_mul, __m256 a, __m256 b) { return is_mul ? _mm256_mul_ps(a, b) : _mm256_add_ps(a, b); }
inline float sop(bool is_mul, float a, float b) { return is_mul ? a * b : a + b; }

// Add / Mul: `is_mul` selects the operation. Both are commutative, so a swapped plan is fine.
MO_AVX2 void binary_avx2(bool is_mul, const BinaryPlan &p, float *o, int64_t lo, int64_t hi) {
    const float *x = p.big->data(), *s = p.small->data();
    const int64_t N = p.big->cols();
#define op(a, b) vop(is_mul, a, b)
#define ops(a, b) sop(is_mul, a, b)
    if (p.mode == BinaryMode::Row) {                                // walk whole rows so the row vector lines up
        for (int64_t r = lo; r < hi; r++) {
            const float *xr = x + r * N;
            float *orow = o + r * N;
            int64_t j = 0;
            for (; j + 8 <= N; j += 8) _mm256_storeu_ps(orow + j, op(_mm256_loadu_ps(xr + j), _mm256_loadu_ps(s + j)));
            for (; j < N; j++) orow[j] = ops(xr[j], s[j]);
        }
        return;
    }
    int64_t i = lo;
    if (p.mode == BinaryMode::Same) {
        for (; i + 8 <= hi; i += 8) _mm256_storeu_ps(o + i, op(_mm256_loadu_ps(x + i), _mm256_loadu_ps(s + i)));
        for (; i < hi; i++) o[i] = ops(x[i], s[i]);
    } else {
        const __m256 sv = _mm256_set1_ps(s[0]);
        for (; i + 8 <= hi; i += 8) _mm256_storeu_ps(o + i, op(_mm256_loadu_ps(x + i), sv));
        for (; i < hi; i++) o[i] = ops(x[i], s[0]);
    }
#undef op
#undef ops
}

Tensor k_binary(bool is_mul, const KernelContext &ctx, const std::vector<const Tensor *> &in) {
    const char *nm = is_mul ? "Mul" : "Add";
    const BinaryPlan p = plan_binary(nm, require(in, 0, nm), require(in, 1, nm));
    Tensor y = Tensor::empty(p.big->shape());
    float *o = y.data();
    if (p.mode == BinaryMode::Row) {                                // parallel over rows
        const int64_t N = std::max<int64_t>(1, p.big->cols());
        ctx.intra->parallel_for(0, p.big->rows(), std::max<int64_t>(1, kElemGrain / N),
                                [&](int64_t lo, int64_t hi) { binary_avx2(is_mul, p, o, lo, hi); });
    } else {
        ctx.intra->parallel_for(0, p.big->numel(), kElemGrain,
                                [&](int64_t lo, int64_t hi) { binary_avx2(is_mul, p, o, lo, hi); });
    }
    return y;
}

class X86ExecutionProvider final : public ExecutionProvider {
public:
    X86ExecutionProvider() {
        kernels_["MatMul"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &) {
            GemmArgs g;
            Tensor y = prepare_matmul(in, g);
            gemm_avx2(*c.intra, g);
            return y;
        };
        kernels_["Gemm"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &at) {
            GemmArgs g;
            Tensor y = prepare_gemm_notrans(in, at, g);
            gemm_avx2(*c.intra, g);
            return y;
        };
        kernels_["Relu"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &) {
            const Tensor &a = require(in, 0, "Relu");
            Tensor y = Tensor::empty(a.shape());
            c.intra->parallel_for(0, a.numel(), kElemGrain, [&](int64_t lo, int64_t hi) { relu_avx2(a.data(), y.data(), lo, hi); });
            return y;
        };
        kernels_["Add"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &) { return k_binary(false, c, in); };
        kernels_["Mul"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &) { return k_binary(true, c, in); };
    }
    std::string name() const override { return "x86"; }
    std::string description() const override { return "x86: AVX2+FMA SIMD kernels (MatMul, Gemm, Add, Mul, Relu)"; }
    bool available() const override {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }
    // GetCapability: the fast Gemm kernel only handles the non-transposed form.
    bool supports(const Node &n, const Graph &g) const override {
        if (n.op_type == "Gemm" && (n.attrs.get_int("transA", 0) || n.attrs.get_int("transB", 0))) return false;
        return ExecutionProvider::supports(n, g);
    }
};

} // namespace

std::unique_ptr<ExecutionProvider> make_x86_ep() { return std::make_unique<X86ExecutionProvider>(); }

#else   // not an x86 GCC/Clang build: the EP does not exist in this binary

std::unique_ptr<ExecutionProvider> make_x86_ep() { return nullptr; }

#endif

} // namespace mo
