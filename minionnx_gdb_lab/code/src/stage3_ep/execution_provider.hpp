// =====================================================================================
// STAGE 3 · execution_provider.hpp  --  "who actually does the math?"
// =====================================================================================
// An EXECUTION PROVIDER (EP) is a back-end that knows how to run some operators on some
// hardware. MiniOnnx has four, just like ONNX Runtime has CPU, CUDA, TensorRT, ...:
//
//   "cpu"   portable C++ reference kernels; supports EVERY op; always available
//   "x86"   AVX2 + FMA SIMD kernels for Intel/AMD; available if the CPU has AVX2 at run time
//   "arm"   NEON SIMD kernels for 64-bit ARM (Apple M-series, Raspberry Pi 4/5, phones)
//   "cuda"  NVIDIA GPU kernels; compiled only with -DMINIONNX_ENABLE_CUDA=ON
//
// PARTITIONING ("GetCapability" in ORT): the user lists EPs in priority order, e.g.
// {"cuda", "x86", "cpu"}. For each node we ask the EPs in that order "can you run this?"
// and the first one that says yes gets the node. "cpu" is always appended last, so every
// node finds a home. A Sigmoid may end up on "cpu" while the MatMuls run on "x86".
// =====================================================================================
#pragma once
#include "core/tensor.hpp"
#include "core/thread_pool.hpp"
#include "stage2_graph/graph.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mo {

struct KernelContext {
    ThreadPool *intra = nullptr;    // the session's intra-op pool: kernels may parallel_for on it
};

// A kernel is a PURE function: read inputs, return a fresh output, touch nothing else.
// Purity is what lets many kernels (and many Run() calls) execute at the same time safely.
// An omitted optional input is passed as nullptr.
using KernelFn = std::function<Tensor(const KernelContext &, const std::vector<const Tensor *> &, const Attributes &)>;

class ExecutionProvider {
public:
    virtual ~ExecutionProvider() = default;
    virtual std::string name() const = 0;
    virtual std::string description() const { return name(); }
    // Can this EP be used on THIS machine right now? (CPU features, GPU present, ...)
    virtual bool available() const { return true; }
    // GetCapability: can this EP run this particular node (op type AND attributes)?
    virtual bool supports(const Node &n, const Graph &) const { return kernels_.count(n.op_type) != 0; }

    const KernelFn &kernel(const std::string &op) const {
        auto it = kernels_.find(op);
        if (it == kernels_.end()) throw std::runtime_error("EP '" + name() + "' has no kernel for '" + op + "'");
        return it->second;
    }
    std::vector<std::string> ops() const {
        std::vector<std::string> v;
        for (auto &kv : kernels_) v.push_back(kv.first);
        return v;
    }

protected:
    std::map<std::string, KernelFn> kernels_;
};

// Factories (one per EP source file). They return nullptr when the EP was not compiled in.
std::unique_ptr<ExecutionProvider> make_cpu_ep();
std::unique_ptr<ExecutionProvider> make_x86_ep();
std::unique_ptr<ExecutionProvider> make_arm_ep();
std::unique_ptr<ExecutionProvider> make_cuda_ep();
// Tell the CUDA EP which tensors are weights (so it uploads them once). No-op otherwise.
void cuda_ep_register_weights(ExecutionProvider *ep, const Graph &g);

// Create by name ("cpu", "x86", "arm", "cuda"). Returns nullptr if unknown/not compiled.
std::unique_ptr<ExecutionProvider> create_execution_provider(const std::string &name);
// Names of the EPs compiled into this binary, and whether each is usable on this machine.
std::vector<std::pair<std::string, bool>> list_execution_providers();

// Partition the graph: set node.ep for every node. `eps` is in priority order and should end
// with the CPU EP. Throws if some node is supported by nobody (an unknown operator).
void assign_providers(Graph &g, const std::vector<const ExecutionProvider *> &eps);

} // namespace mo
