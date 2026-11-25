// =====================================================================================
// STAGE 2 · graph.cpp
// =====================================================================================
#include "stage2_graph/graph.hpp"

#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

namespace mo {

int Graph::intern(const std::string &n) {
    auto it = ids_.find(n);
    if (it != ids_.end()) return it->second;
    const int id = int(names_.size());
    names_.push_back(n);
    ids_[n] = id;
    return id;
}

int Graph::find(const std::string &n) const {
    auto it = ids_.find(n);
    return it == ids_.end() ? -1 : it->second;
}

Graph Graph::from_model(const ModelProto &m, const std::string &base_dir) {
    Graph g;
    g.graph_name = m.graph.name;
    for (auto &o : m.opset_import)
        if (o.domain.empty() || o.domain == "ai.onnx") g.opset = o.version;

    g.weights.load(m.graph, base_dir);                         // Stage 1: decode the weights
    for (auto &t : m.graph.initializer) g.intern(t.name);      // initializers get the first ids

    for (auto &vi : m.graph.input) {
        const int id = g.intern(vi.name);
        if (g.is_initializer(id)) continue;                    // older exporters list weights as inputs too
        g.inputs.push_back(id);
        g.input_info[id] = vi;
    }
    for (auto &np : m.graph.node) {
        Node n;
        n.op_type = np.op_type;
        n.name = np.name;
        n.domain = np.domain;
        for (auto &a : np.attribute) n.attrs.set(a);
        for (auto &s : np.input) n.inputs.push_back(s.empty() ? kNoValue : g.intern(s));
        for (auto &s : np.output) {
            if (s.empty()) throw std::runtime_error("node '" + np.name + "' has an empty output name");
            n.outputs.push_back(g.intern(s));
        }
        if (n.name.empty()) n.name = n.op_type + "_" + std::to_string(g.nodes.size());
        g.nodes.push_back(std::move(n));
    }
    for (auto &vi : m.graph.output) g.outputs.push_back(g.intern(vi.name));
    g.validate();
    return g;
}

std::vector<int> Graph::producers() const {
    std::vector<int> p(size_t(num_values()), -1);
    for (int i = 0; i < int(nodes.size()); i++)
        for (int o : nodes[size_t(i)].outputs) p[size_t(o)] = i;
    return p;
}

std::vector<int> Graph::consumer_counts() const {
    std::vector<int> c(size_t(num_values()), 0);
    for (auto &n : nodes)
        for (int v : n.inputs) if (v != kNoValue) c[size_t(v)]++;
    for (int v : outputs) c[size_t(v)]++;
    return c;
}

void Graph::validate() const {
    std::set<int> available(inputs.begin(), inputs.end());     // things that exist before any node runs
    for (int v = 0; v < num_values(); v++) if (is_initializer(v)) available.insert(v);

    std::vector<int> producer(size_t(num_values()), -1);
    for (int i = 0; i < int(nodes.size()); i++) {
        const Node &n = nodes[size_t(i)];
        if (!n.domain.empty() && n.domain != "ai.onnx")
            throw std::runtime_error("node '" + n.name + "': custom operator domain '" + n.domain + "' is not supported");
        for (int o : n.outputs) {
            if (producer[size_t(o)] != -1 || available.count(o))
                throw std::runtime_error("value '" + name(o) + "' has more than one producer");
            producer[size_t(o)] = i;
        }
    }
    for (auto &n : nodes)
        for (int v : n.inputs)
            if (v != kNoValue && producer[size_t(v)] == -1 && !available.count(v))
                throw std::runtime_error("dangling reference: node '" + n.name + "' reads '" + name(v) +
                                         "' but nothing produces it");
    for (int v : outputs)
        if (producer[size_t(v)] == -1 && !available.count(v))
            throw std::runtime_error("graph output '" + name(v) + "' is never produced");
    (void)topological_order();                                 // throws on a cycle
}

// Kahn's algorithm. indegree[n] = number of input EDGES of n that come from other nodes.
// Counting edges (not distinct producers) keeps Mul(x, x) consistent: two edges in, and the
// producer lists n twice in its dependents, so both edges are removed again.
std::vector<int> Graph::topological_order() const {
    const int N = int(nodes.size());
    std::vector<int> producer = producers();
    std::vector<int> indegree(size_t(N), 0);
    // NOTE: `dependents(size_t(N))` would be the "most vexing parse": C++ reads it as the
    // declaration of a FUNCTION taking a size_t named N. static_cast cannot be a parameter.
    std::vector<std::vector<int>> dependents(static_cast<size_t>(N));
    for (int i = 0; i < N; i++)
        for (int v : nodes[size_t(i)].inputs) {
            if (v == kNoValue) continue;
            const int p = producer[size_t(v)];
            if (p < 0) continue;                                   // weight or graph input
            dependents[size_t(p)].push_back(i);
            indegree[size_t(i)]++;
        }
    std::queue<int> ready;
    for (int i = 0; i < N; i++) if (indegree[size_t(i)] == 0) ready.push(i);
    std::vector<int> order;
    order.reserve(size_t(N));
    while (!ready.empty()) {
        const int i = ready.front();
        ready.pop();
        order.push_back(i);
        for (int d : dependents[size_t(i)]) if (--indegree[size_t(d)] == 0) ready.push(d);
    }
    if (int(order.size()) != N) throw std::runtime_error("graph has a cycle: not a valid DAG");
    return order;
}

std::string Graph::to_string() const {
    std::ostringstream os;
    os << "graph '" << graph_name << "' (opset " << opset << "): " << nodes.size() << " nodes, "
       << weights.all().size() << " initializers\n";
    os << "  inputs :";
    for (int v : inputs) os << " " << name(v);
    os << "\n  outputs:";
    for (int v : outputs) os << " " << name(v);
    os << "\n";
    for (int i : topological_order()) {
        const Node &n = nodes[size_t(i)];
        os << "  [" << i << "] " << n.op_type << " '" << n.name << "' (";
        for (size_t k = 0; k < n.inputs.size(); k++)
            os << (k ? ", " : "") << (n.inputs[k] == kNoValue ? std::string("<none>") : name(n.inputs[k]));
        os << ") -> ";
        for (size_t k = 0; k < n.outputs.size(); k++) os << (k ? ", " : "") << name(n.outputs[k]);
        if (!n.ep.empty()) os << "   @" << n.ep;
        os << "\n";
    }
    return os.str();
}

} // namespace mo
