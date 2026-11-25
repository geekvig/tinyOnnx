// =====================================================================================
// STAGE 3 · cuda_ep.cu  --  NVIDIA GPU Execution Provider ("cuda")      [OPTIONAL]
// =====================================================================================
// Built only when CMake is configured with -DMINIONNX_ENABLE_CUDA=ON (needs the CUDA
// toolkit; on Windows nvcc requires MSVC, so use WSL2/Linux for this EP with g++).
//
// A GPU is a separate computer with its own memory. Every CUDA kernel call therefore has
// three parts:
//     1. copy inputs   host (CPU RAM) --> device (GPU RAM)      cudaMemcpyAsync H2D
//     2. launch        thousands of GPU threads each compute a few output elements
//     3. copy result   device --> host                           cudaMemcpyAsync D2H
//
// TEACHING SIMPLIFICATION (be honest about it): MiniOnnx tensors live in host memory, so
// this EP copies activations in and out for EVERY node. Weights are different: they never
// change, so each weight is uploaded ONCE and cached on the GPU (see device_weight()).
// Real ONNX Runtime keeps activations on the GPU between nodes and inserts explicit
// "MemcpyToHost/FromHost" nodes only where a CPU node needs the data. That is exercise 7
// in the PDF.
//
// Concurrency: kernels are launched on cudaStreamPerThread, so two inter-op threads
// running two nodes do not serialize on the legacy default stream.
//
// STATUS: this file is compile-checked (device + host code) but was NOT run on a GPU
// while writing this tutorial -- no GPU was available. Test it before trusting it.
// =====================================================================================
#include "stage3_ep/kernel_utils.hpp"

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace mo {
namespace {

void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}
#define CUDA_CHECK(x) check((x), #x)

// RAII device buffer: freed automatically, even if an exception is thrown.
struct DeviceBuffer {
    float *ptr = nullptr;
    explicit DeviceBuffer(size_t n) { if (n) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float))); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

// ------------------------------------------------------------------ device kernels
constexpr int TILE = 16;

// Classic shared-memory tiled matrix multiply. Each 16x16 thread block computes a 16x16
// tile of Y. It walks along K in steps of 16, each time loading one tile of A and one tile
// of B into fast on-chip shared memory, so each value is read from slow global memory once
// per block instead of 16 times.
__global__ void gemm_kernel(const float *A, const float *B, const float *C, float *Y, int M, int K, int N,
                            float alpha, float beta, int mode) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        As[threadIdx.y][threadIdx.x] = (row < M && t + threadIdx.x < K) ? A[row * K + t + threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t + threadIdx.y < K) ? B[(t + threadIdx.y) * N + col] : 0.f;
        __syncthreads();                                   // wait until the whole tile is loaded
        for (int p = 0; p < TILE; p++) acc += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        __syncthreads();                                   // wait before overwriting the tile
    }
    if (row < M && col < N) {
        float bias = 0.f;
        if (mode == 1) bias = C[0];
        else if (mode == 2) bias = C[col];
        else if (mode == 3) bias = C[row];
        else if (mode == 4) bias = C[row * N + col];
        Y[row * N + col] = alpha * acc + beta * bias;
    }
}

// op: 0 relu, 1 sigmoid, 2 tanh
__global__ void unary_kernel(const float *x, float *y, long long n, int op) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = op == 0 ? fmaxf(v, 0.f) : op == 1 ? 1.f / (1.f + expf(-v)) : tanhf(v);
}

// op: 0 add, 1 mul;  mode: 0 same, 1 row (small has N elements), 2 scalar
__global__ void binary_kernel(const float *a, const float *b, float *y, long long n, long long N, int op, int mode) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = mode == 0 ? b[i] : mode == 1 ? b[i % N] : b[0];
    y[i] = op == 0 ? a[i] + s : a[i] * s;
}

// ------------------------------------------------------------------ the provider
class CudaExecutionProvider final : public ExecutionProvider {
public:
    CudaExecutionProvider() {
        int n = 0;
        available_ = cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
        if (!available_) cudaGetLastError();               // clear the sticky error
        kernels_["MatMul"] = [this](const KernelContext &, const std::vector<const Tensor *> &in, const Attributes &) {
            GemmArgs g;
            Tensor y = prepare_matmul(in, g);
            run_gemm(g, in[0], in[1], nullptr);
            return y;
        };
        kernels_["Gemm"] = [this](const KernelContext &, const std::vector<const Tensor *> &in, const Attributes &at) {
            GemmArgs g;
            Tensor y = prepare_gemm_notrans(in, at, g);
            run_gemm(g, in[0], in[1], in.size() > 2 ? in[2] : nullptr);
            return y;
        };
        kernels_["Relu"] = [this](const KernelContext &, const std::vector<const Tensor *> &in, const Attributes &) { return run_unary(require(in, 0, "Relu"), 0); };
        kernels_["Sigmoid"] = [this](const KernelContext &, const std::vector<const Tensor *> &in, const Attributes &) { return run_unary(require(in, 0, "Sigmoid"), 1); };
        kernels_["Tanh"] = [this](const KernelContext &, const std::vector<const Tensor *> &in, const Attributes &) { return run_unary(require(in, 0, "Tanh"), 2); };
        kernels_["Add"] = [this](const KernelContext &, const std::vector<const Tensor *> &in, const Attributes &) { return run_binary(in, 0); };
        kernels_["Mul"] = [this](const KernelContext &, const std::vector<const Tensor *> &in, const Attributes &) { return run_binary(in, 1); };
    }
    ~CudaExecutionProvider() override {
        for (auto &kv : weight_cache_) cudaFree(kv.second);
    }
    std::string name() const override { return "cuda"; }
    std::string description() const override { return "cuda: NVIDIA GPU kernels (MatMul, Gemm, Add, Mul, Relu, Sigmoid, Tanh)"; }
    bool available() const override { return available_; }
    bool supports(const Node &n, const Graph &g) const override {
        if (n.op_type == "Gemm" && (n.attrs.get_int("transA", 0) || n.attrs.get_int("transB", 0))) return false;
        return ExecutionProvider::supports(n, g);
    }

    // Weights are stable host pointers (they live in the WeightStore arena), so we can use
    // the host address as the cache key. The first node that needs a weight uploads it.
    void register_weight(const Tensor *t) const {
        std::lock_guard<std::mutex> lk(mu_);
        weights_.insert(t->data());
    }

private:
    // Returns a device pointer for tensor t: the cached copy if t is a weight, otherwise
    // a fresh upload into `scratch`.
    const float *device_ptr(const Tensor *t, std::unique_ptr<DeviceBuffer> &scratch, cudaStream_t s) const {
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = weight_cache_.find(t->data());
            if (it != weight_cache_.end()) return it->second;
        }
        scratch = std::make_unique<DeviceBuffer>(size_t(t->numel()));
        CUDA_CHECK(cudaMemcpyAsync(scratch->ptr, t->data(), size_t(t->numel()) * sizeof(float), cudaMemcpyHostToDevice, s));
        return scratch->ptr;
    }

    void run_gemm(const GemmArgs &g, const Tensor *a, const Tensor *b, const Tensor *c) const {
        cudaStream_t s = cudaStreamPerThread;
        std::unique_ptr<DeviceBuffer> da, db, dc;
        const float *A = device_ptr(a, da, s), *B = cache_weight(b, db, s), *C = c ? cache_weight(c, dc, s) : nullptr;
        DeviceBuffer dy(static_cast<size_t>(g.M * g.N));
        dim3 block(TILE, TILE), grid(unsigned((g.N + TILE - 1) / TILE), unsigned((g.M + TILE - 1) / TILE));
        gemm_kernel<<<grid, block, 0, s>>>(A, B, C, dy.ptr, int(g.M), int(g.K), int(g.N), g.alpha, g.beta, int(g.mode));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(g.Y, dy.ptr, size_t(g.M * g.N) * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));             // buffers die at '}' -> must be finished
    }

    // B and C of a Gemm/MatMul are usually weights: upload once, keep forever.
    // Anything that is not a registered weight is uploaded per call into `scratch`.
    const float *cache_weight(const Tensor *t, std::unique_ptr<DeviceBuffer> &scratch, cudaStream_t s) const {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (weights_.count(t->data())) {
                auto it = weight_cache_.find(t->data());
                if (it != weight_cache_.end()) return it->second;
                float *d = nullptr;
                CUDA_CHECK(cudaMalloc(&d, size_t(t->numel()) * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d, t->data(), size_t(t->numel()) * sizeof(float), cudaMemcpyHostToDevice));
                weight_cache_[t->data()] = d;
                return d;
            }
        }                                                  // lock released before the slow path
        return device_ptr(t, scratch, s);
    }

    Tensor run_unary(const Tensor &a, int op) const {
        cudaStream_t s = cudaStreamPerThread;
        Tensor y = Tensor::empty(a.shape());
        const long long n = a.numel();
        DeviceBuffer dx(static_cast<size_t>(n)), dy(static_cast<size_t>(n));
        CUDA_CHECK(cudaMemcpyAsync(dx.ptr, a.data(), size_t(n) * sizeof(float), cudaMemcpyHostToDevice, s));
        unary_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(dx.ptr, dy.ptr, n, op);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(y.data(), dy.ptr, size_t(n) * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        return y;
    }

    Tensor run_binary(const std::vector<const Tensor *> &in, int op) const {
        const char *nm = op ? "Mul" : "Add";
        const BinaryPlan p = plan_binary(nm, require(in, 0, nm), require(in, 1, nm));
        cudaStream_t s = cudaStreamPerThread;
        Tensor y = Tensor::empty(p.big->shape());
        const long long n = p.big->numel();
        std::unique_ptr<DeviceBuffer> da, db;
        const float *A = device_ptr(p.big, da, s), *B = cache_weight(p.small, db, s);
        DeviceBuffer dy(static_cast<size_t>(n));
        const int mode = p.mode == BinaryMode::Same ? 0 : p.mode == BinaryMode::Row ? 1 : 2;
        binary_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(A, B, dy.ptr, n, p.big->cols(), op, mode);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(y.data(), dy.ptr, size_t(n) * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        return y;
    }

    bool available_ = false;
    mutable std::mutex mu_;
    mutable std::unordered_map<const float *, float *> weight_cache_;
    mutable std::unordered_set<const float *> weights_;
};

} // namespace

std::unique_ptr<ExecutionProvider> make_cuda_ep() { return std::make_unique<CudaExecutionProvider>(); }

// Called by the session after load so the EP knows which host pointers are weights.
void cuda_ep_register_weights(ExecutionProvider *ep, const Graph &g) {
    auto *cuda = dynamic_cast<CudaExecutionProvider *>(ep);
    if (!cuda) return;
    for (auto &kv : g.weights.all()) cuda->register_weight(&kv.second);
}

} // namespace mo
