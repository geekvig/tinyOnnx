// =====================================================================================
// STAGE 3 · arm_ep.cpp  --  NEON Execution Provider ("arm") for 64-bit ARM (AArch64)
// =====================================================================================
// NEON is ARM's SIMD unit. A float32x4_t register holds 4 floats (128 bits) -- half of an
// AVX2 register -- but AArch64 has 32 of them (x86 AVX2 has 16), so we can keep many
// accumulators in registers. NEON is mandatory on every AArch64 CPU, so no run-time check
// is needed: if this file was compiled for AArch64, the EP is available.
//
// Same micro-kernel idea as the x86 EP, sized for NEON: a 4 x 8 output block lives in
// 8 registers (4 rows x 2 registers of 4 floats).
//
//        B row p:   [ b0 b1 b2 b3 | b4 b5 b6 b7 ]          2 loads (vld1q_f32)
//        acc[r][h] = vfmaq_n_f32(acc[r][h], b_h, a_r)      8 fused multiply-adds by a scalar
//
// How to test it on a PC without ARM hardware: cross-compile with aarch64-linux-gnu-g++
// and run under QEMU (see README "ARM"). That is how this file was verified.
//
// Ops: MatMul, Gemm(transA=transB=0), Add, Mul, Relu.
// =====================================================================================
#include "stage3_ep/kernel_utils.hpp"

#if defined(__aarch64__) && defined(__ARM_NEON)
#define MO_ARM_EP 1
#include <arm_neon.h>
#endif

namespace mo {

#ifdef MO_ARM_EP
namespace {

inline float32x4_t bias4(const GemmArgs &g, int64_t i, int64_t j) {
    switch (g.mode) {
    case BiasMode::None: return vdupq_n_f32(0.f);
    case BiasMode::Scalar: return vdupq_n_f32(g.beta * g.C[0]);
    case BiasMode::Row: return vmulq_n_f32(vld1q_f32(g.C + j), g.beta);
    case BiasMode::Column: return vdupq_n_f32(g.beta * g.C[i]);
    case BiasMode::Full: return vmulq_n_f32(vld1q_f32(g.C + i * g.N + j), g.beta);
    }
    return vdupq_n_f32(0.f);
}

inline void store4(const GemmArgs &g, int64_t i, int64_t j, float32x4_t acc) {
    vst1q_f32(g.Y + i * g.N + j, vfmaq_n_f32(bias4(g, i, j), acc, g.alpha));   // bias + acc*alpha
}

inline void tail_cols(const GemmArgs &g, int64_t i, int rows, int64_t j0) {
    for (int r = 0; r < rows; r++)
        for (int64_t j = j0; j < g.N; j++) {
            float acc = 0.f;
            for (int64_t p = 0; p < g.K; p++) acc += g.A[(i + r) * g.K + p] * g.B[p * g.N + j];
            g.Y[(i + r) * g.N + j] = g.alpha * acc + g.bias_at(i + r, j);
        }
}

void gemm_rows_neon(const GemmArgs &g, int64_t rb, int64_t re) {
    const int64_t K = g.K, N = g.N;
    const float *A = g.A, *B = g.B;
    int64_t i = rb;
    for (; i + 4 <= re; i += 4) {
        const float *a0 = A + i * K, *a1 = a0 + K, *a2 = a1 + K, *a3 = a2 + K;
        int64_t j = 0;
        for (; j + 8 <= N; j += 8) {                                  // 4 x 8 micro-kernel
            float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0), c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
            for (int64_t p = 0; p < K; p++) {
                const float32x4_t b0 = vld1q_f32(B + p * N + j), b1 = vld1q_f32(B + p * N + j + 4);
                c00 = vfmaq_n_f32(c00, b0, a0[p]); c01 = vfmaq_n_f32(c01, b1, a0[p]);
                c10 = vfmaq_n_f32(c10, b0, a1[p]); c11 = vfmaq_n_f32(c11, b1, a1[p]);
                c20 = vfmaq_n_f32(c20, b0, a2[p]); c21 = vfmaq_n_f32(c21, b1, a2[p]);
                c30 = vfmaq_n_f32(c30, b0, a3[p]); c31 = vfmaq_n_f32(c31, b1, a3[p]);
            }
            store4(g, i, j, c00);     store4(g, i, j + 4, c01);
            store4(g, i + 1, j, c10); store4(g, i + 1, j + 4, c11);
            store4(g, i + 2, j, c20); store4(g, i + 2, j + 4, c21);
            store4(g, i + 3, j, c30); store4(g, i + 3, j + 4, c31);
        }
        for (; j + 4 <= N; j += 4) {                                  // 4 x 4
            float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0), c2 = vdupq_n_f32(0), c3 = vdupq_n_f32(0);
            for (int64_t p = 0; p < K; p++) {
                const float32x4_t b = vld1q_f32(B + p * N + j);
                c0 = vfmaq_n_f32(c0, b, a0[p]); c1 = vfmaq_n_f32(c1, b, a1[p]);
                c2 = vfmaq_n_f32(c2, b, a2[p]); c3 = vfmaq_n_f32(c3, b, a3[p]);
            }
            store4(g, i, j, c0); store4(g, i + 1, j, c1); store4(g, i + 2, j, c2); store4(g, i + 3, j, c3);
        }
        if (j < N) tail_cols(g, i, 4, j);
    }
    for (; i < re; i++) {
        const float *a = A + i * K;
        int64_t j = 0;
        for (; j + 4 <= N; j += 4) {
            float32x4_t c = vdupq_n_f32(0);
            for (int64_t p = 0; p < K; p++) c = vfmaq_n_f32(c, vld1q_f32(B + p * N + j), a[p]);
            store4(g, i, j, c);
        }
        if (j < N) tail_cols(g, i, 1, j);
    }
}

void gemm_neon(ThreadPool &pool, const GemmArgs &g) {
    pool.parallel_for(0, g.M, gemm_row_grain(g), [&](int64_t rb, int64_t re) { gemm_rows_neon(g, rb, re); });
}

Tensor k_binary(bool is_mul, const KernelContext &ctx, const std::vector<const Tensor *> &in) {
    const char *nm = is_mul ? "Mul" : "Add";
    const BinaryPlan p = plan_binary(nm, require(in, 0, nm), require(in, 1, nm));
    Tensor y = Tensor::empty(p.big->shape());
    const float *x = p.big->data(), *s = p.small->data();
    float *o = y.data();
    const int64_t N = std::max<int64_t>(1, p.big->cols());
    auto vop = [is_mul](float32x4_t a, float32x4_t b) { return is_mul ? vmulq_f32(a, b) : vaddq_f32(a, b); };
    auto sop = [is_mul](float a, float b) { return is_mul ? a * b : a + b; };
    if (p.mode == BinaryMode::Row) {
        ctx.intra->parallel_for(0, p.big->rows(), std::max<int64_t>(1, kElemGrain / N), [&](int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; r++) {
                int64_t j = 0;
                for (; j + 4 <= N; j += 4) vst1q_f32(o + r * N + j, vop(vld1q_f32(x + r * N + j), vld1q_f32(s + j)));
                for (; j < N; j++) o[r * N + j] = sop(x[r * N + j], s[j]);
            }
        });
    } else {
        const bool same = p.mode == BinaryMode::Same;
        ctx.intra->parallel_for(0, p.big->numel(), kElemGrain, [&](int64_t lo, int64_t hi) {
            int64_t i = lo;
            for (; i + 4 <= hi; i += 4) vst1q_f32(o + i, vop(vld1q_f32(x + i), same ? vld1q_f32(s + i) : vdupq_n_f32(s[0])));
            for (; i < hi; i++) o[i] = sop(x[i], same ? s[i] : s[0]);
        });
    }
    return y;
}

class ArmExecutionProvider final : public ExecutionProvider {
public:
    ArmExecutionProvider() {
        kernels_["MatMul"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &) {
            GemmArgs g;
            Tensor y = prepare_matmul(in, g);
            gemm_neon(*c.intra, g);
            return y;
        };
        kernels_["Gemm"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &at) {
            GemmArgs g;
            Tensor y = prepare_gemm_notrans(in, at, g);
            gemm_neon(*c.intra, g);
            return y;
        };
        kernels_["Relu"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &) {
            const Tensor &a = require(in, 0, "Relu");
            Tensor y = Tensor::empty(a.shape());
            const float *x = a.data();
            float *o = y.data();
            c.intra->parallel_for(0, a.numel(), kElemGrain, [&](int64_t lo, int64_t hi) {
                int64_t i = lo;
                for (; i + 4 <= hi; i += 4) vst1q_f32(o + i, vmaxq_f32(vdupq_n_f32(0.f), vld1q_f32(x + i)));
                for (; i < hi; i++) o[i] = x[i] > 0.f ? x[i] : 0.f;
            });
            return y;
        };
        kernels_["Add"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &) { return k_binary(false, c, in); };
        kernels_["Mul"] = [](const KernelContext &c, const std::vector<const Tensor *> &in, const Attributes &) { return k_binary(true, c, in); };
    }
    std::string name() const override { return "arm"; }
    std::string description() const override { return "arm: AArch64 NEON SIMD kernels (MatMul, Gemm, Add, Mul, Relu)"; }
    bool supports(const Node &n, const Graph &g) const override {
        if (n.op_type == "Gemm" && (n.attrs.get_int("transA", 0) || n.attrs.get_int("transB", 0))) return false;
        return ExecutionProvider::supports(n, g);
    }
};

} // namespace

std::unique_ptr<ExecutionProvider> make_arm_ep() { return std::make_unique<ArmExecutionProvider>(); }

#else

std::unique_ptr<ExecutionProvider> make_arm_ep() { return nullptr; }

#endif

} // namespace mo
