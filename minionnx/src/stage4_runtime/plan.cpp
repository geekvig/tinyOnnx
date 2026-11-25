// =====================================================================================
// STAGE 4 · plan.cpp
// =====================================================================================
#include "stage4_runtime/plan.hpp"

namespace mo {

ExecutionPlan build_plan(const Graph &g, const std::map<std::string, const ExecutionProvider *> &eps) {
    ExecutionPlan p;
    const int V = g.num_values(), N = int(g.nodes.size());
    for (int v = 0; v < V; v++) {
        p.value_names.push_back(g.name(v));
        p.value_id[g.name(v)] = v;
    }
    p.graph_inputs = g.inputs;
    p.graph_outputs = g.outputs;
    p.topo_order = g.topological_order();

    for (const Node &n : g.nodes) {
        auto it = eps.find(n.ep);
        if (it == eps.end()) throw std::runtime_error("node '" + n.name + "' was not assigned an execution provider");
        if (n.outputs.size() != 1)
            throw std::runtime_error("node '" + n.name + "': only single-output operators are supported");
        PlanNode pn;
        pn.op_type = n.op_type;
        pn.name = n.name;
        pn.ep_name = n.ep;
        pn.in = n.inputs;
        pn.out = n.outputs;
        pn.attrs = n.attrs;
        pn.kernel = &it->second->kernel(n.op_type);
        p.nodes.push_back(std::move(pn));
    }

    std::vector<int> producer = g.producers();
    p.dependents.assign(size_t(N), {});
    p.indegree.assign(size_t(N), 0);
    p.consumer_count.assign(size_t(V), 0);
    p.persistent.assign(size_t(V), 0);
    p.weights.assign(size_t(V), nullptr);

    for (int i = 0; i < N; i++)
        for (int v : p.nodes[size_t(i)].in) {
            if (v == kNoValue) continue;
            p.consumer_count[size_t(v)]++;                 // per input SLOT: Mul(x,x) counts x twice
            const int prod = producer[size_t(v)];
            if (prod >= 0) {                               // an edge from another node
                p.dependents[size_t(prod)].push_back(i);
                p.indegree[size_t(i)]++;
            }
        }
    for (int v = 0; v < V; v++)
        if (const Tensor *w = g.initializer(v)) {
            p.weights[size_t(v)] = std::make_shared<const Tensor>(*w);   // shares the arena buffer
            p.persistent[size_t(v)] = 1;
        }
    for (int v : p.graph_outputs) p.persistent[size_t(v)] = 1;
    return p;
}

} // namespace mo
