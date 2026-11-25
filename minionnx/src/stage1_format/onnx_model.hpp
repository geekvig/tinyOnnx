// =====================================================================================
// STAGE 1 · onnx_model.hpp  --  the subset of onnx.proto that MiniOnnx understands
// =====================================================================================
// These structs mirror the messages in the official onnx/onnx.proto. The comment after each
// member is its FIELD NUMBER in that schema -- the number that is actually written in the
// file (see protobuf_wire.hpp). If you open onnx.proto you will find the same numbers.
//
//   ModelProto                     the whole file
//    ├── ir_version, opset_import  "which edition of the rules this file follows"
//    └── graph : GraphProto        the network
//         ├── node[]        : NodeProto       operations ("MatMul", "Relu", ...)
//         ├── initializer[] : TensorProto     the WEIGHTS (trained numbers)
//         ├── input[]       : ValueInfoProto  what the caller must feed in
//         └── output[]      : ValueInfoProto  what Run() returns
//
// Nodes are connected by tensor NAMES (strings), not pointers: node A writes "h1",
// node B reads "h1". Stage 2 turns those names into a real graph.
// =====================================================================================
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mo {

// TensorProto.DataType values (subset). MiniOnnx computes in float32 only.
enum class DataType : int32_t { Undefined = 0, Float = 1, Uint8 = 2, Int8 = 3, Int32 = 6, Int64 = 7, Double = 11 };

struct StringStringEntry {               // StringStringEntryProto
    std::string key;                     // 1
    std::string value;                   // 2
};

// A weight, as stored in the file. Exactly one of these holds the numbers:
//   float_data   : a packed list of floats                      (simple, used by small models)
//   raw_data     : the raw little-endian bytes, 4 per float     (what PyTorch writes; compact)
//   external     : the bytes live in ANOTHER file               (models > 2 GB; protobuf's limit)
struct TensorProto {
    std::vector<int64_t> dims;                    // 1
    DataType data_type = DataType::Float;         // 2
    std::vector<float> float_data;                // 4 (packed)
    std::string name;                             // 8
    std::string raw_data;                         // 9  (bytes)
    std::vector<StringStringEntry> external_data; // 13 (location, offset, length)
    int32_t data_location = 0;                    // 14 (0 = DEFAULT, 1 = EXTERNAL)

    int64_t numel() const {                       // number of elements implied by dims
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
};

enum class AttrType : int32_t { Undefined = 0, Float = 1, Int = 2, String = 3, Tensor = 4, Floats = 6, Ints = 7 };

struct AttributeProto {
    std::string name;                // 1
    float f = 0;                     // 2
    int64_t i = 0;                   // 3
    std::string s;                   // 4
    std::vector<float> floats;       // 7
    std::vector<int64_t> ints;       // 8
    AttrType type = AttrType::Undefined; // 20
};

struct NodeProto {
    std::vector<std::string> input;  // 1
    std::vector<std::string> output; // 2
    std::string name;                // 3
    std::string op_type;             // 4
    std::vector<AttributeProto> attribute; // 5
    std::string domain;              // 7  ("" = the standard ai.onnx operator set)
};

// ValueInfoProto with its TypeProto flattened: element type + shape.
// A dim may be a number (dim_value) or a symbolic name like "batch" (dim_param).
struct ValueInfo {
    std::string name;                // 1
    DataType elem_type = DataType::Float;   // type.tensor_type.elem_type
    std::vector<int64_t> dims;       // -1 where the dim is symbolic / unknown
    std::vector<std::string> dim_names;     // dim_param text ("" when numeric)
    bool has_shape = false;
};

struct GraphProto {
    std::vector<NodeProto> node;            // 1
    std::string name;                       // 2
    std::vector<TensorProto> initializer;   // 5
    std::vector<ValueInfo> input;           // 11
    std::vector<ValueInfo> output;          // 12
};

struct OperatorSetId {
    std::string domain;              // 1
    int64_t version = 0;             // 2
};

struct ModelProto {
    int64_t ir_version = 8;          // 1
    std::string producer_name;       // 2
    std::string producer_version;    // 3
    GraphProto graph;                // 7
    std::vector<OperatorSetId> opset_import; // 8
};

// ------------------------------- file I/O (onnx_io.cpp) ---------------------------------
// Parse an in-memory .onnx file. Throws FormatError on malformed bytes.
ModelProto parse_model(const std::string &bytes);
// Encode a model to bytes (the inverse of parse_model).
std::string serialize_model(const ModelProto &m);

// Read/write a file. `load_model_file` remembers the folder so external weights can be found.
ModelProto load_model_file(const std::string &path, std::string *base_dir = nullptr);

struct SaveOptions {
    bool use_raw_data = true;            // raw_data (true) or float_data (false)
    bool external_data = false;          // put weights in "<file>.data" next to the model
    int64_t external_threshold = 1024;   // only tensors with at least this many bytes go external
    int64_t external_alignment = 64;     // start each external tensor on a 64-byte boundary
};
void save_model_file(const ModelProto &m, const std::string &path, const SaveOptions &opt = {});

// Helpers to build attributes in code.
AttributeProto make_attr(const std::string &name, float f);
AttributeProto make_attr(const std::string &name, int64_t i);

} // namespace mo
