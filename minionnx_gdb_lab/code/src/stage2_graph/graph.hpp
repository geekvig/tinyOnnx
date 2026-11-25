// =====================================================================================
// STAGE 2 · graph.hpp  --  the in-memory graph (IR) the optimizer and runtime work on
// =====================================================================================
// Stage 1 gave us a ModelProto where nodes refer to tensors by NAME. Strings are slow to
// compare and easy to get wrong, so Stage 2 "interns" every name: each distinct tensor
// name gets a small integer id ("value id").
//
//     "x" -> 0   "W1" -> 1   "z1" -> 2   ...       nodes now hold ints, not strings
//
// Then it VALIDATES the graph (every input has exactly one producer, no cycles, ...) and can
// compute a TOPOLOGICAL ORDER: an order in which every node comes after the nodes that
// produce its inputs (like getting dressed: socks before shoes).
// =====================================================================================
#pragma once
#include "stage1_format/onnx_model.hpp"
#include "stage1_format/weights.hpp"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace mo {

class Attributes {
public:
    void set(const AttributeProto &a) { map_[a.name] = a; }
    bool has(const std::string &n) const { return map_.count(n) != 0; }
    float get_float(const std::string &n, float def) const {
        auto it = map_.find(n);
        return it == map_.end() ? def : it->second.f;
    }
    int64_t get_int(const std::string &n, int64_t def) const {
        auto it = map_.find(n);
        return it == map_.end() ? def : it->second.i;
    }
    void set_int(const std::string &n, int64_t v) { set(make_attr(n, v)); }
    void set_float(const std::string &n, float v) { set(make_attr(n, v)); }
    const std::map<std::string, AttributeProto> &all() const { return map_; }

private:
    std::map<std::string, AttributeProto> map_;
};

constexpr int kNoValue = -1;                 // an omitted optional input (empty name "" in ONNX)

struct Node {
    std::string op_type, name, domain;
    std::vector<int> inputs, outputs;        // value ids (kNoValue for an omitted optional input)
    Attributes attrs;
    std::string ep;                          // execution provider chosen for this node (Stage 3)
};

class Graph {
public:
    // Build from a parsed model: intern names, decode weights, keep declared input shapes.
    static Graph from_model(const ModelProto &m, const std::string &base_dir = ".");

    // --- names <-> ids ---
    int intern(const std::string &name);                     // get or create an id
    int find(const std::string &name) const;                 // -1 if unknown
    const std::string &name(int id) const { return names_.at(size_t(id)); }
    int num_values() const { return int(names_.size()); }

    bool is_initializer(int v) const { return v >= 0 && weights.find(names_[size_t(v)]) != nullptr; }
    const Tensor *initializer(int v) const { return v >= 0 ? weights.find(names_[size_t(v)]) : nullptr; }

    // --- structure queries (recomputed on demand; the graph is small) ---
    std::vector<int> producers() const;                      // value id -> node index or -1
    std::vector<int> consumer_counts() const;                // value id -> #input slots reading it (+1 per graph output)

    void validate() const;                                   // throws std::runtime_error with a clear message
    std::vector<int> topological_order() const;              // Kahn's algorithm; throws on a cycle
    std::string to_string() const;                           // human-readable dump

    std::vector<Node> nodes;
    std::vector<int> inputs;                                 // graph inputs the caller must feed
    std::vector<int> outputs;                                // graph outputs Run() returns
    std::map<int, ValueInfo> input_info;                     // declared type/shape of each input
    WeightStore weights;                                     // the initializers (by name)
    std::string graph_name;
    int64_t opset = 0;

private:
    std::vector<std::string> names_;
    std::unordered_map<std::string, int> ids_;
};

} // namespace mo
