// =====================================================================================
// STAGE 2 · optimizer.cpp
// =====================================================================================
#include "stage2_graph/optimizer.hpp"

#include <algorithm>
#include <set>
#include <sstream>

namespace mo {

std::string OptimizerStats::to_string() const {
    std::ostringstream os;
    os << identities_removed << " identity removed, " << transposes_folded << " transpose folded, " << gemms_fused
       << " MatMul+Add fused, " << dead_nodes_removed << " dead node(s) removed, " << weights_dropped
       << " unused weight(s) dropped";
    return os.str();
}

namespace {
bool is_graph_output(const Graph &g, int v) {
    return std::find(g.outputs.begin(), g.outputs.end(), v) != g.outputs.end();
}
} // namespace

// Identity(x) -> y : make every consumer of y read x instead, then the Identity node is dead.
// Not done when y is a graph output (the output must keep its name).
// Two passes ON PURPOSE: first rewire every consumer in place, THEN drop the Identity nodes.
// (A first version copied nodes into the result while still rewiring, so nodes copied
// earlier kept reading the removed value -> "dangling reference". The unit tests caught it.)
int eliminate_identity(Graph &g) {
    std::vector<char> drop(g.nodes.size(), 0);
    int removed = 0;
    for (size_t i = 0; i < g.nodes.size(); i++) {
        const Node &n = g.nodes[i];
        if (n.op_type == "Identity" && n.inputs.size() == 1 && n.outputs.size() == 1 &&
            n.inputs[0] != kNoValue && !is_graph_output(g, n.outputs[0])) {
            const int from = n.outputs[0], to = n.inputs[0];
            for (auto &m : g.nodes)
                for (int &v : m.inputs) if (v == from) v = to;     // pass 1: rewire (also later Identities)
            drop[i] = 1;
            removed++;
        }
    }
    std::vector<Node> kept;                                          // pass 2: filter
    for (size_t i = 0; i < g.nodes.size(); i++) if (!drop[i]) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    return removed;
}

// Gemm(A, B, C, transB=1) where B is a constant: store B already transposed so the fast
// kernels (which want B as K x N, row-major) can be used, and no work is repeated per run.
// This is a tiny example of CONSTANT FOLDING: compute at load time what never changes.
int fold_gemm_transposes(Graph &g) {
    int folded = 0;
    for (auto &n : g.nodes) {
        if (n.op_type != "Gemm" || n.attrs.get_int("transB", 0) == 0 || n.inputs.size() < 2) continue;
        const Tensor *B = g.initializer(n.inputs[1]);
        if (!B || B->rank() != 2) continue;
        const int64_t r = B->dim(0), c = B->dim(1);
        Tensor T = Tensor::empty({c, r});
        for (int64_t i = 0; i < r; i++)
            for (int64_t j = 0; j < c; j++) T[j * r + i] = (*B)[i * c + j];
        const std::string new_name = g.name(n.inputs[1]) + "__transposed";
        if (!g.weights.find(new_name)) g.weights.add(new_name, T);
        n.inputs[1] = g.intern(new_name);
        n.attrs.set_int("transB", 0);
        folded++;
    }
    return folded;
}

// MatMul -> Add fusion, with the guard that fixed a real bug in the original tinyONNX.
//
// It is ONLY a bias-add (and therefore safe to fuse) when ALL of these hold:
//   (1) the MatMul output t is read exactly once (by this Add) and is not a graph output;
//   (2) the Add reads t exactly once (Add(t, t) is not a bias add);
//   (3) the other operand is a CONSTANT (initializer) shaped {N}, {1,N} or {1}
//       -- a branch merge Add(t, other_activation) is NOT a bias, fusing it is wrong;
//   (4) if the MatMul's B is a constant, N must equal B's width (or be 1).
int fuse_matmul_add(Graph &g) {
    int fused = 0;
    std::vector<int> consumers = g.consumer_counts();
    std::vector<char> removed(g.nodes.size(), 0);
    for (size_t i = 0; i < g.nodes.size(); i++) {
        Node &mm = g.nodes[i];
        if (removed[i] || mm.op_type != "MatMul" || mm.inputs.size() != 2 || mm.outputs.size() != 1) continue;
        const int t = mm.outputs[0];
        if (consumers[size_t(t)] != 1 || is_graph_output(g, t)) continue;          // (1)
        for (size_t j = 0; j < g.nodes.size(); j++) {
            const Node &add = g.nodes[j];
            if (j == i || removed[j] || add.op_type != "Add" || add.inputs.size() != 2) continue;
            const bool first = add.inputs[0] == t, second = add.inputs[1] == t;
            if (first == second) continue;                                          // (2)
            const int other = first ? add.inputs[1] : add.inputs[0];
            const Tensor *bias = g.initializer(other);
            if (!bias) continue;                                                     // (3) constant?
            const bool vec = bias->rank() == 1 || (bias->rank() == 2 && bias->dim(0) == 1);
            if (!vec) continue;                                                      // (3) vector-shaped?
            if (const Tensor *B = g.initializer(mm.inputs[1]))                       // (4)
                if (B->rank() == 2 && bias->numel() != 1 && bias->numel() != B->dim(1)) continue;

            Node gemm;
            gemm.op_type = "Gemm";
            gemm.name = "fused_" + mm.name + "_" + add.name;
            gemm.inputs = {mm.inputs[0], mm.inputs[1], other};
            gemm.outputs = add.outputs;
            gemm.attrs.set_float("alpha", 1.f);
            gemm.attrs.set_float("beta", 1.f);
            mm = gemm;                              // the MatMul slot becomes the Gemm
            removed[j] = 1;                         // the Add disappears
            fused++;
            break;
        }
    }
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); i++) if (!removed[i]) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    return fused;
}

// Walk BACKWARDS from the graph outputs, marking every value that is needed. A node is live
// if any of its outputs is needed. Everything else can be removed without changing outputs.
int eliminate_dead_nodes(Graph &g) {
    std::vector<int> producer = g.producers();
    std::vector<char> needed(size_t(g.num_values()), 0), live(g.nodes.size(), 0);
    std::vector<int> stack(g.outputs.begin(), g.outputs.end());
    while (!stack.empty()) {
        const int v = stack.back();
        stack.pop_back();
        if (v == kNoValue || needed[size_t(v)]) continue;
        needed[size_t(v)] = 1;
        const int p = producer[size_t(v)];
        if (p >= 0 && !live[size_t(p)]) {
            live[size_t(p)] = 1;
            for (int in : g.nodes[size_t(p)].inputs) stack.push_back(in);
        }
    }
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); i++) if (live[i]) kept.push_back(g.nodes[i]);
    const int removed = int(g.nodes.size() - kept.size());
    g.nodes = std::move(kept);
    return removed;
}

int drop_unused_weights(Graph &g) {
    std::set<std::string> used;
    for (auto &n : g.nodes) for (int v : n.inputs) if (v != kNoValue) used.insert(g.name(v));
    for (int v : g.outputs) used.insert(g.name(v));
    std::vector<std::string> drop;
    for (auto &kv : g.weights.all()) if (!used.count(kv.first)) drop.push_back(kv.first);
    for (auto &n : drop) g.weights.remove(n);
    return int(drop.size());
}

OptimizerStats optimize(Graph &g) {
    OptimizerStats s;
    s.identities_removed = eliminate_identity(g);
    s.transposes_folded = fold_gemm_transposes(g);
    s.gemms_fused = fuse_matmul_add(g);
    s.dead_nodes_removed = eliminate_dead_nodes(g);
    s.weights_dropped = drop_unused_weights(g);
    g.validate();                                // a pass must never produce an invalid graph
    return s;
}

} // namespace mo
