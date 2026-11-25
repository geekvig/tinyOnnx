// =====================================================================================
// tools/inspect_onnx.cpp  --  look inside an .onnx file
// =====================================================================================
//   inspect_onnx model.onnx                 summary: inputs, outputs, nodes, weights
//   inspect_onnx model.onnx --wire [lines]  "protobuf X-ray": every field, byte by byte
//   inspect_onnx model.onnx --plan [eps]    what the optimizer + partitioner make of it
//
// The --wire mode is the best way to SEE the protobuf wire format from Stage 1: it prints
// the byte offset, field number, wire type and (using a tiny copy of onnx.proto's field
// numbers) the field name of everything in the file.
// =====================================================================================
#include "minionnx.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

using namespace mo;

namespace {

enum class Msg { None, Model, Graph, Node, Attr, Tensor, Entry, ValueInfo, Type, TypeTensor, Shape, Dim, Opset };
enum class Kind { Int, Str, Bytes, Float, Sub };
struct FieldInfo { const char *name; Kind kind; Msg sub; };

// field numbers copied from onnx/onnx.proto
const std::map<std::pair<Msg, uint32_t>, FieldInfo> &schema() {
    static const std::map<std::pair<Msg, uint32_t>, FieldInfo> s = {
        {{Msg::Model, 1}, {"ir_version", Kind::Int, Msg::None}},
        {{Msg::Model, 2}, {"producer_name", Kind::Str, Msg::None}},
        {{Msg::Model, 3}, {"producer_version", Kind::Str, Msg::None}},
        {{Msg::Model, 7}, {"graph", Kind::Sub, Msg::Graph}},
        {{Msg::Model, 8}, {"opset_import", Kind::Sub, Msg::Opset}},
        {{Msg::Opset, 1}, {"domain", Kind::Str, Msg::None}},
        {{Msg::Opset, 2}, {"version", Kind::Int, Msg::None}},
        {{Msg::Graph, 1}, {"node", Kind::Sub, Msg::Node}},
        {{Msg::Graph, 2}, {"name", Kind::Str, Msg::None}},
        {{Msg::Graph, 5}, {"initializer", Kind::Sub, Msg::Tensor}},
        {{Msg::Graph, 11}, {"input", Kind::Sub, Msg::ValueInfo}},
        {{Msg::Graph, 12}, {"output", Kind::Sub, Msg::ValueInfo}},
        {{Msg::Node, 1}, {"input", Kind::Str, Msg::None}},
        {{Msg::Node, 2}, {"output", Kind::Str, Msg::None}},
        {{Msg::Node, 3}, {"name", Kind::Str, Msg::None}},
        {{Msg::Node, 4}, {"op_type", Kind::Str, Msg::None}},
        {{Msg::Node, 5}, {"attribute", Kind::Sub, Msg::Attr}},
        {{Msg::Node, 7}, {"domain", Kind::Str, Msg::None}},
        {{Msg::Attr, 1}, {"name", Kind::Str, Msg::None}},
        {{Msg::Attr, 2}, {"f", Kind::Float, Msg::None}},
        {{Msg::Attr, 3}, {"i", Kind::Int, Msg::None}},
        {{Msg::Attr, 4}, {"s", Kind::Str, Msg::None}},
        {{Msg::Attr, 8}, {"ints", Kind::Int, Msg::None}},
        {{Msg::Attr, 20}, {"type", Kind::Int, Msg::None}},
        {{Msg::Tensor, 1}, {"dims", Kind::Int, Msg::None}},
        {{Msg::Tensor, 2}, {"data_type", Kind::Int, Msg::None}},
        {{Msg::Tensor, 4}, {"float_data", Kind::Bytes, Msg::None}},
        {{Msg::Tensor, 8}, {"name", Kind::Str, Msg::None}},
        {{Msg::Tensor, 9}, {"raw_data", Kind::Bytes, Msg::None}},
        {{Msg::Tensor, 13}, {"external_data", Kind::Sub, Msg::Entry}},
        {{Msg::Tensor, 14}, {"data_location", Kind::Int, Msg::None}},
        {{Msg::Entry, 1}, {"key", Kind::Str, Msg::None}},
        {{Msg::Entry, 2}, {"value", Kind::Str, Msg::None}},
        {{Msg::ValueInfo, 1}, {"name", Kind::Str, Msg::None}},
        {{Msg::ValueInfo, 2}, {"type", Kind::Sub, Msg::Type}},
        {{Msg::Type, 1}, {"tensor_type", Kind::Sub, Msg::TypeTensor}},
        {{Msg::TypeTensor, 1}, {"elem_type", Kind::Int, Msg::None}},
        {{Msg::TypeTensor, 2}, {"shape", Kind::Sub, Msg::Shape}},
        {{Msg::Shape, 1}, {"dim", Kind::Sub, Msg::Dim}},
        {{Msg::Dim, 1}, {"dim_value", Kind::Int, Msg::None}},
        {{Msg::Dim, 2}, {"dim_param", Kind::Str, Msg::None}},
    };
    return s;
}

int lines_left = 200;

void xray(WireReader r, Msg msg, int depth) {
    uint32_t field;
    WireType wt;
    while (lines_left > 0) {
        const size_t at = r.offset();
        if (!r.next_field(field, wt)) return;
        auto it = schema().find({msg, field});
        const char *name = it != schema().end() ? it->second.name : "?";
        std::printf("@%06zu %*s#%-2u %-16s %-6s ", at, depth * 2, "", field, name, wire_type_name(wt));
        lines_left--;
        if (wt == WireType::Varint) { std::printf("= %llu\n", (unsigned long long)r.read_varint()); continue; }
        if (wt == WireType::I32) { std::printf("= %g\n", r.read_float()); continue; }
        if (wt == WireType::I64) { std::printf("= 0x%llx\n", (unsigned long long)r.read_fixed64()); continue; }
        // LEN: a string, raw bytes, a packed array, or a sub-message
        if (it != schema().end() && it->second.kind == Kind::Sub) {
            WireReader sub = r.read_submessage();
            std::printf("{ (%zu bytes)\n", sub.size());
            xray(sub, it->second.sub, depth + 1);
            if (lines_left > 0) std::printf("        %*s}\n", depth * 2, "");
            continue;
        }
        std::string_view v = r.read_bytes();
        bool printable = it != schema().end() && it->second.kind == Kind::Str;
        if (printable) {
            std::printf("= \"%.*s\"\n", int(std::min<size_t>(v.size(), 60)), v.data());
        } else {
            std::printf("%zu bytes:", v.size());
            for (size_t i = 0; i < std::min<size_t>(v.size(), 12); i++) std::printf(" %02x", unsigned(uint8_t(v[i])));
            std::printf(v.size() > 12 ? " ...\n" : "\n");
        }
    }
}

std::string dims_str(const ValueInfo &vi) {
    if (!vi.has_shape) return "?";
    std::string s = "[";
    for (size_t i = 0; i < vi.dims.size(); i++)
        s += (i ? "," : "") + (vi.dim_names[i].empty() ? std::to_string(vi.dims[i]) : vi.dim_names[i]);
    return s + "]";
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::printf("usage: inspect_onnx model.onnx [--wire [lines]] [--plan [ep,ep,...]]\n");
        return 2;
    }
    const std::string path = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "";
    try {
        if (mode == "--wire") {
            if (argc > 3) lines_left = std::stoi(argv[3]);
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("cannot open " + path);
            std::ostringstream ss;
            ss << f.rdbuf();
            const std::string bytes = ss.str();
            std::printf("protobuf X-ray of %s (%zu bytes)   offset  #field name  wire-type  value\n", path.c_str(), bytes.size());
            xray(WireReader(bytes), Msg::Model, 0);
            if (lines_left <= 0) std::printf("... (stopped; pass a larger line count to see more)\n");
            return 0;
        }
        std::string base;
        ModelProto m = load_model_file(path, &base);
        if (mode == "--plan") {
            SessionOptions o;
            o.verbose = true;
            o.providers = {"cuda", "x86", "arm", "cpu"};
            if (argc > 3) {
                o.providers.clear();
                std::stringstream ss(argv[3]);
                std::string t;
                while (std::getline(ss, t, ',')) o.providers.push_back(t);
            }
            InferenceSession s(o);
            s.load_model(m, base);
            std::printf("%s", s.graph().to_string().c_str());
            return 0;
        }
        std::printf("file        : %s\n", path.c_str());
        std::printf("ir_version  : %lld   producer: %s %s\n", (long long)m.ir_version, m.producer_name.c_str(), m.producer_version.c_str());
        for (auto &o : m.opset_import) std::printf("opset       : domain '%s' version %lld\n", o.domain.c_str(), (long long)o.version);
        std::printf("graph       : '%s'  %zu nodes, %zu initializers\n", m.graph.name.c_str(), m.graph.node.size(), m.graph.initializer.size());
        for (auto &vi : m.graph.input) std::printf("  input  %-12s %s\n", vi.name.c_str(), dims_str(vi).c_str());
        for (auto &vi : m.graph.output) std::printf("  output %-12s %s\n", vi.name.c_str(), dims_str(vi).c_str());
        std::printf("nodes:\n");
        for (auto &n : m.graph.node) {
            std::printf("  %-8s %-12s (", n.op_type.c_str(), n.name.c_str());
            for (size_t i = 0; i < n.input.size(); i++) std::printf("%s%s", i ? ", " : "", n.input[i].c_str());
            std::printf(") -> ");
            for (size_t i = 0; i < n.output.size(); i++) std::printf("%s%s", i ? ", " : "", n.output[i].c_str());
            for (auto &a : n.attribute) {
                if (a.type == AttrType::Int) std::printf("  %s=%lld", a.name.c_str(), (long long)a.i);
                else if (a.type == AttrType::Float) std::printf("  %s=%g", a.name.c_str(), a.f);
            }
            std::printf("\n");
        }
        WeightStore ws;
        ws.load(m.graph, base);
        std::printf("weights:\n%s", ws.summary().c_str());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
