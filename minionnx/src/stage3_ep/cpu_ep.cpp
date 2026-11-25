// =====================================================================================
// STAGE 3 · cpu_ep.cpp  --  the portable reference Execution Provider ("cpu")
// =====================================================================================
// Plain C++ that any compiler on any CPU can build. It supports every operator MiniOnnx
// knows, so it is the fallback that makes partitioning always succeed. Each kernel splits
// its OUTPUT across the intra-op pool: every output element is written by exactly one
// thread, in a fixed order -> no locks, and bit-identical results for any thread count.
// =====================================================================================
#include "stage3_ep/kernel_utils.hpp"

#include <algorithm>
#include <cmath>

namespace mo {

namespace {

// Y = alpha * A x B + beta * C.  Cache-blocked over K (64 at a time keeps a block of B in
// cache) and tiled 4 rows at a time (each loaded value of B is used 4 times).
// When alpha == 1 each row STARTS at the bias and accumulates the products on top -- this
// is how a fused MatMul+Add gets its bias for free.
void gemm_reference(ThreadPool &pool, const GemmArgs &g) {
    constexpr int64_t KB = 64;
    const bool alpha_one = g.alpha == 1.f;
    pool.parallel_for(0, g.M, gemm_row_grain(g), [&](int64_t rb, int64_t re) {
        for (int64_t i = rb; i < re; i++)
            for (int64_t j = 0; j < g.N; j++) g.Y[i * g.N + j] = alpha_one ? g.bias_at(i, j) : 0.f;
        for (int64_t p0 = 0; p0 < g.K; p0 += KB) {
            const int64_t p1 = std::min(g.K, p0 + KB);
            int64_t i = rb;
            for (; i + 4 <= re; i += 4) {
                float *c0 = g.Y + i * g.N, *c1 = c0 + g.N, *c2 = c1 + g.N, *c3 = c2 + g.N;
                for (int64_t p = p0; p < p1; p++) {
                    const float a0 = g.A[i * g.K + p], a1 = g.A[(i + 1) * g.K + p];
                    const float a2 = g.A[(i + 2) * g.K + p], a3 = g.A[(i + 3) * g.K + p];
                    const float *b = g.B + p * g.N;
                    for (int64_t j = 0; j < g.N; j++) {
                        const float bj = b[j];
                        c0[j] += a0 * bj; c1[j] += a1 * bj; c2[j] += a2 * bj; c3[j] += a3 * bj;
                    }
                }
            }
            for (; i < re; i++) {                                  // leftover rows (M % 4)
                float *c = g.Y + i * g.N;
                for (int64_t p = p0; p < p1; p++) {
                    const float a = g.A[i * g.K + p];
                    const float *b = g.B + p * g.N;
                    for (int64_t j = 0; j < g.N; j++) c[j] += a * b[j];
                }
            }
        }
        if (!alpha_one)
            for (int64_t i = rb; i < re; i++)
                for (int64_t j = 0; j < g.N; j++) g.Y[i * g.N + j] = g.alpha * g.Y[i * g.N + j] + g.bias_at(i, j);
    });
}

Tensor transpose2d(const Tensor &t) {
    const int64_t r = t.dim(0), c = t.dim(1);
    Tensor o = Tensor::empty({c, r});
    for (int64_t i = 0; i < r; i++)
        for (int64_t j = 0; j < c; j++) o[j * r + i] = t[i * c + j];
    return o;
}

Tensor k_matmul(const KernelContext &ctx, const std::vector<const Tensor *> &in, const Attributes &) {
    GemmArgs g;
    Tensor y = prepare_matmul(in, g);
    gemm_reference(*ctx.intra, g);
    return y;
}

// General Gemm: transposes are materialized into temporaries (correct but not free --
// that is why the optimizer folds constant transposes at load time).
Tensor k_gemm(const KernelContext &ctx, const std::vector<const Tensor *> &in, const Attributes &at) {
    const Tensor &a0 = require(in, 0, "Gemm"), &b0 = require(in, 1, "Gemm");
    if (a0.rank() != 2 || b0.rank() != 2) throw std::runtime_error("Gemm: inputs must be 2-D");
    Tensor at_tmp, bt_tmp;
    const Tensor *a = &a0, *b = &b0;
    if (at.get_int("transA", 0)) { at_tmp = transpose2d(a0); a = &at_tmp; }
    if (at.get_int("transB", 0)) { bt_tmp = transpose2d(b0); b = &bt_tmp; }
    Attributes plain = at;
    plain.set_int("transA", 0);
    plain.set_int("transB", 0);
    std::vector<const Tensor *> args = {a, b, in.size() > 2 ? in[2] : nullptr};
    GemmArgs g;
    Tensor y = prepare_gemm_notrans(args, plain, g);
    gemm_reference(*ctx.intra, g);
    return y;
}

template <class Op>
Tensor binary(const char *name, const KernelContext &ctx, const std::vector<const Tensor *> &in, Op op) {
    const Tensor &a = require(in, 0, name), &b = require(in, 1, name);
    const BinaryPlan p = plan_binary(name, a, b);
    Tensor y = Tensor::empty(p.big->shape());
    const float *x = p.big->data(), *s = p.small->data();
    float *o = y.data();
    const int64_t N = p.big->cols();
    const bool sw = p.swapped;
    ctx.intra->parallel_for(0, p.big->numel(), kElemGrain, [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; i++) {
            const float other = p.mode == BinaryMode::Same ? s[i] : p.mode == BinaryMode::Row ? s[i % N] : s[0];
            o[i] = sw ? op(other, x[i]) : op(x[i], other);
        }
    });
    return y;
}

template <class Op>
Tensor unary(const KernelContext &ctx, const Tensor &a, Op op) {
    Tensor y = Tensor::empty(a.shape());
    const float *x = a.data();
    float *o = y.data();
    ctx.intra->parallel_for(0, a.numel(), kElemGrain, [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; i++) o[i] = op(x[i]);
    });
    return y;
}

// Softmax over the last axis, numerically stable: subtract the row max before exp()
// so exp() never overflows (exp(1000) = inf, but exp(1000 - 1000) = 1).
Tensor k_softmax(const KernelContext &ctx, const std::vector<const Tensor *> &in, const Attributes &at) {
    const Tensor &a = require(in, 0, "Softmax");
    int64_t axis = at.get_int("axis", -1);
    if (axis < 0) axis += a.rank();
    if (axis != a.rank() - 1) throw std::runtime_error("Softmax: only the last axis is supported");
    Tensor y = Tensor::empty(a.shape());
    const int64_t R = a.rows(), N = a.cols();
    ctx.intra->parallel_for(0, R, std::max<int64_t>(1, kElemGrain / std::max<int64_t>(1, N)), [&](int64_t rb, int64_t re) {
        for (int64_t r = rb; r < re; r++) {
            const float *x = a.data() + r * N;
            float *o = y.data() + r * N;
            float mx = -INFINITY;
            for (int64_t j = 0; j < N; j++) mx = std::max(mx, x[j]);
            float sum = 0.f;
            for (int64_t j = 0; j < N; j++) { o[j] = std::exp(x[j] - mx); sum += o[j]; }
            for (int64_t j = 0; j < N; j++) o[j] /= sum;
        }
    });
    return y;
}

class CpuExecutionProvider final : public ExecutionProvider {
public:
    CpuExecutionProvider() {
        kernels_["MatMul"] = k_matmul;
        kernels_["Gemm"] = k_gemm;
        kernels_["Add"] = [](auto &c, auto &in, auto &) { return binary("Add", c, in, [](float x, float y) { return x + y; }); };
        kernels_["Sub"] = [](auto &c, auto &in, auto &) { return binary("Sub", c, in, [](float x, float y) { return x - y; }); };
        kernels_["Mul"] = [](auto &c, auto &in, auto &) { return binary("Mul", c, in, [](float x, float y) { return x * y; }); };
        kernels_["Relu"] = [](auto &c, auto &in, auto &) { return unary(c, require(in, 0, "Relu"), [](float x) { return x > 0.f ? x : 0.f; }); };
        kernels_["Sigmoid"] = [](auto &c, auto &in, auto &) { return unary(c, require(in, 0, "Sigmoid"), [](float x) { return 1.f / (1.f + std::exp(-x)); }); };
        kernels_["Tanh"] = [](auto &c, auto &in, auto &) { return unary(c, require(in, 0, "Tanh"), [](float x) { return std::tanh(x); }); };
        kernels_["Identity"] = [](auto &c, auto &in, auto &) { return unary(c, require(in, 0, "Identity"), [](float x) { return x; }); };
        kernels_["Softmax"] = k_softmax;
    }
    std::string name() const override { return "cpu"; }
    std::string description() const override { return "cpu: portable C++ reference kernels (all ops)"; }
};

} // namespace

std::unique_ptr<ExecutionProvider> make_cpu_ep() { return std::make_unique<CpuExecutionProvider>(); }

} // namespace mo
