// =====================================================================================
// STAGE 3 · registry.cpp  --  create EPs by name, and partition the graph between them
// =====================================================================================
#include "stage3_ep/execution_provider.hpp"

namespace mo {

std::unique_ptr<ExecutionProvider> create_execution_provider(const std::string &name) {
    if (name == "cpu") return make_cpu_ep();
    if (name == "x86") return make_x86_ep();
    if (name == "arm") return make_arm_ep();
    if (name == "cuda") return make_cuda_ep();
    return nullptr;
}

std::vector<std::pair<std::string, bool>> list_execution_providers() {
    std::vector<std::pair<std::string, bool>> out;
    for (const char *n : {"cuda", "x86", "arm", "cpu"})
        if (auto ep = create_execution_provider(n)) out.push_back({n, ep->available()});
    return out;
}

// For every node, the first EP (in priority order) that is available AND supports the node
// wins. This is the whole "graph partitioning" step of ONNX Runtime, in 10 lines.
void assign_providers(Graph &g, const std::vector<const ExecutionProvider *> &eps) {
    for (Node &n : g.nodes) {
        n.ep.clear();
        for (const ExecutionProvider *ep : eps)
            if (ep->available() && ep->supports(n, g)) { n.ep = ep->name(); break; }
        if (n.ep.empty())
            throw std::runtime_error("no execution provider can run node '" + n.name + "' (op_type '" + n.op_type +
                                     "'): operator not supported by MiniOnnx");
    }
}

} // namespace mo
