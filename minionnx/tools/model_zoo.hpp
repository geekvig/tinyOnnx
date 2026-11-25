// =====================================================================================
// tools/model_zoo.hpp  --  build small ONNX models in C++ (used by make_models, tests, bench)
// =====================================================================================
// This plays the role of torch.onnx.export(): it fills in a ModelProto by hand. Once saved,
// the file is a REAL .onnx file -- Netron can display it and onnxruntime can run it.
// =====================================================================================
#pragma once
#include "minionnx.hpp"

#include <cmath>
#include <random>

namespace zoo {
using namespace mo;

inline TensorProto tensor(const std::string &name, std::vector<int64_t> dims, std::vector<float> values) {
    TensorProto t;
    t.name = name;
    t.dims = std::move(dims);
    t.float_data = std::move(values);
    return t;
}

inline TensorProto random_tensor(const std::string &name, std::vector<int64_t> dims, float scale, std::mt19937 &rng) {
    TensorProto t;
    t.name = name;
    t.dims = dims;
    std::normal_distribution<float> dist(0.f, scale);
    t.float_data.resize(size_t(t.numel()));
    for (auto &v : t.float_data) v = dist(rng);
    return t;
}

// A value with shape [batch, n], where batch is symbolic ("N" = any batch size).
inline ValueInfo batch_input(const std::string &name, int64_t n) {
    ValueInfo vi;
    vi.name = name;
    vi.has_shape = true;
    vi.dims = {-1, n};
    vi.dim_names = {"N", ""};
    return vi;
}

inline NodeProto node(const std::string &op, const std::string &name, std::vector<std::string> in,
                      std::vector<std::string> out, std::vector<AttributeProto> attrs = {}) {
    NodeProto n;
    n.op_type = op;
    n.name = name;
    n.input = std::move(in);
    n.output = std::move(out);
    n.attribute = std::move(attrs);
    return n;
}

inline ModelProto empty_model(const std::string &graph_name) {
    ModelProto m;
    m.ir_version = 8;
    m.producer_name = "minionnx-model-zoo";
    m.producer_version = "1.0";
    m.opset_import = {{"", 13}};
    m.graph.name = graph_name;
    return m;
}

// The 2-8-1 XOR network trained by tinyPytorch (weights copied from the original xor.tinyonnx).
//   x[N,2] -> MatMul(W1) -> Add(b1) -> Relu -> MatMul(W2) -> Add(b2) -> Sigmoid -> out[N,1]
inline ModelProto make_xor() {
    ModelProto m = empty_model("xor_mlp");
    auto &g = m.graph;
    g.initializer = {
        tensor("W1", {2, 8}, {-0.569609702f, 1.78645205f, 2.15863872f, 1.17061973f, 1.50635457f, -0.0482498854f,
                              1.58416998f, -2.63292503f, -0.286451489f, -1.78645372f, -2.15849209f, 1.17061341f,
                              0.827136219f, -1.05826724f, 1.74275553f, 2.63304424f}),
        tensor("b1", {1, 8}, {0.f, -0.000183790122f, -6.98551376e-05f, -1.17060792f, -0.0322508104f, 0.f,
                              6.04682327e-07f, -0.000142836914f}),
        tensor("W2", {8, 1}, {-0.620104909f, 1.63407457f, 2.44251776f, -1.48730838f, 0.0101194866f, -0.0824610218f,
                              0.196818054f, 3.09076166f}),
        tensor("b2", {1, 1}, {-3.74593449f}),
    };
    g.node = {node("MatMul", "n1", {"x", "W1"}, {"z1"}), node("Add", "n2", {"z1", "b1"}, {"a1"}),
              node("Relu", "n3", {"a1"}, {"h"}),         node("MatMul", "n4", {"h", "W2"}, {"z2"}),
              node("Add", "n5", {"z2", "b2"}, {"a2"}),   node("Sigmoid", "n6", {"a2"}, {"out"})};
    g.input = {batch_input("x", 2)};
    g.output = {batch_input("out", 1)};
    return m;
}

// What PyTorch's nn.Linear exports: Gemm(x, W, b, transB=1) with W stored as [out, in].
//   x[N,in] -> Gemm -> Tanh -> Gemm -> Softmax -> probs[N,classes]
inline ModelProto make_classifier(int in, int hidden, int classes, unsigned seed = 3) {
    std::mt19937 rng(seed);
    ModelProto m = empty_model("classifier");
    auto &g = m.graph;
    g.initializer = {random_tensor("fc1.weight", {hidden, in}, 1.f / std::sqrt(float(in)), rng),
                     random_tensor("fc1.bias", {hidden}, 0.1f, rng),
                     random_tensor("fc2.weight", {classes, hidden}, 1.f / std::sqrt(float(hidden)), rng),
                     random_tensor("fc2.bias", {classes}, 0.1f, rng)};
    g.node = {node("Gemm", "fc1", {"x", "fc1.weight", "fc1.bias"}, {"h1"}, {make_attr("transB", int64_t(1))}),
              node("Tanh", "act", {"h1"}, {"h2"}),
              node("Gemm", "fc2", {"h2", "fc2.weight", "fc2.bias"}, {"logits"}, {make_attr("transB", int64_t(1))}),
              node("Softmax", "softmax", {"logits"}, {"probs"}, {make_attr("axis", int64_t(-1))})};
    g.input = {batch_input("x", in)};
    g.output = {batch_input("probs", classes)};
    return m;
}

// B independent branches  MatMul -> Add(bias) -> Relu -> MatMul -> Add(bias),  merged by a
// balanced tree of same-shape Adds, then Sigmoid. The branches do not depend on each other:
// exactly the graph shape the PARALLEL executor can speed up.
inline ModelProto make_branchy(int branches, int in_dim, int hidden, unsigned seed = 1) {
    std::mt19937 rng(seed);
    ModelProto m = empty_model("branchy");
    auto &g = m.graph;
    std::vector<std::string> outs;
    for (int b = 0; b < branches; b++) {
        const std::string s = std::to_string(b);
        g.initializer.push_back(random_tensor("Wa" + s, {in_dim, hidden}, 1.f / std::sqrt(float(in_dim)), rng));
        g.initializer.push_back(random_tensor("ba" + s, {1, hidden}, 0.1f, rng));
        g.initializer.push_back(random_tensor("Wb" + s, {hidden, in_dim}, 1.f / std::sqrt(float(hidden)), rng));
        g.initializer.push_back(random_tensor("bb" + s, {1, in_dim}, 0.1f, rng));
        g.node.push_back(node("MatMul", "mm_a" + s, {"x", "Wa" + s}, {"za" + s}));
        g.node.push_back(node("Add", "add_a" + s, {"za" + s, "ba" + s}, {"pa" + s}));
        g.node.push_back(node("Relu", "relu" + s, {"pa" + s}, {"h" + s}));
        g.node.push_back(node("MatMul", "mm_b" + s, {"h" + s, "Wb" + s}, {"zb" + s}));
        g.node.push_back(node("Add", "add_b" + s, {"zb" + s, "bb" + s}, {"y" + s}));
        outs.push_back("y" + s);
    }
    int k = 0;
    while (outs.size() > 1) {
        std::vector<std::string> next;
        for (size_t i = 0; i + 1 < outs.size(); i += 2) {
            const std::string name = "s" + std::to_string(k++);
            g.node.push_back(node("Add", "merge_" + name, {outs[i], outs[i + 1]}, {name}));
            next.push_back(name);
        }
        if (outs.size() % 2) next.push_back(outs.back());
        outs = next;
    }
    g.node.push_back(node("Sigmoid", "sigmoid", {outs[0]}, {"out"}));
    g.input = {batch_input("x", in_dim)};
    g.output = {batch_input("out", in_dim)};
    return m;
}

// A straight line of Relu nodes: nothing to parallelize; perfect for checking memory freeing.
inline ModelProto make_chain(int length, int width = 16) {
    ModelProto m = empty_model("chain");
    auto &g = m.graph;
    std::string prev = "x";
    for (int i = 0; i < length; i++) {
        const std::string out = (i + 1 == length) ? "out" : "t" + std::to_string(i);
        g.node.push_back(node("Relu", "relu" + std::to_string(i), {prev}, {out}));
        prev = out;
    }
    g.input = {batch_input("x", width)};
    g.output = {batch_input("out", width)};
    return m;
}

// Deterministic "random" input: same numbers on every machine (used to compare with ORT).
inline Tensor pattern_input(Shape shape, float phase = 0.f) {
    Tensor t = Tensor::empty(shape);
    for (int64_t i = 0; i < t.numel(); i++) t[i] = std::sin(0.37f * float(i) + phase);
    return t;
}

} // namespace zoo
