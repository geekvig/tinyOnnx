// =====================================================================================
// STAGE 1 · onnx_io.cpp  --  bytes  <->  ModelProto
// =====================================================================================
// Each parse_xxx() function reads ONE protobuf message. The pattern is always the same:
//
//     while (r.next_field(field, wt)) {
//         switch (field) {
//         case 1: ...read the value for field 1...; break;
//         case 2: ...
//         default: r.skip(wt);          // unknown field -> ignore it (forward compatible)
//         }
//     }
//
// Each serialize_xxx() does the reverse, writing only the fields MiniOnnx knows about.
// =====================================================================================
#include "stage1_format/onnx_model.hpp"
#include "stage1_format/protobuf_wire.hpp"

#include <fstream>
#include <sstream>

namespace mo {

namespace {

std::string to_str(std::string_view v) { return std::string(v.data(), v.size()); }

void expect(WireType got, WireType want, uint32_t field, const WireReader &r) {
    if (got != want)
        WireReader::fail("field " + std::to_string(field) + " has wire type " + wire_type_name(got) +
                             ", expected " + wire_type_name(want), r.offset());
}

// ---------------------------------------------------------------- parsing
StringStringEntry parse_entry(WireReader r) {
    StringStringEntry e;
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        if (f == 1) { expect(wt, WireType::Len, f, r); e.key = to_str(r.read_bytes()); }
        else if (f == 2) { expect(wt, WireType::Len, f, r); e.value = to_str(r.read_bytes()); }
        else r.skip(wt);
    }
    return e;
}

TensorProto parse_tensor(WireReader r) {
    TensorProto t;
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        switch (f) {
        case 1: r.read_repeated_int64(wt, t.dims); break;
        case 2: expect(wt, WireType::Varint, f, r); t.data_type = DataType(int32_t(r.read_varint())); break;
        case 4: r.read_repeated_float(wt, t.float_data); break;
        case 8: expect(wt, WireType::Len, f, r); t.name = to_str(r.read_bytes()); break;
        case 9: expect(wt, WireType::Len, f, r); t.raw_data = to_str(r.read_bytes()); break;
        case 13: expect(wt, WireType::Len, f, r); t.external_data.push_back(parse_entry(r.read_submessage())); break;
        case 14: expect(wt, WireType::Varint, f, r); t.data_location = int32_t(r.read_varint()); break;
        default: r.skip(wt);
        }
    }
    return t;
}

AttributeProto parse_attribute(WireReader r) {
    AttributeProto a;
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        switch (f) {
        case 1: expect(wt, WireType::Len, f, r); a.name = to_str(r.read_bytes()); break;
        case 2: expect(wt, WireType::I32, f, r); a.f = r.read_float(); break;
        case 3: expect(wt, WireType::Varint, f, r); a.i = int64_t(r.read_varint()); break;
        case 4: expect(wt, WireType::Len, f, r); a.s = to_str(r.read_bytes()); break;
        case 7: r.read_repeated_float(wt, a.floats); break;
        case 8: r.read_repeated_int64(wt, a.ints); break;
        case 20: expect(wt, WireType::Varint, f, r); a.type = AttrType(int32_t(r.read_varint())); break;
        default: r.skip(wt);
        }
    }
    return a;
}

NodeProto parse_node(WireReader r) {
    NodeProto n;
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        switch (f) {
        case 1: expect(wt, WireType::Len, f, r); n.input.push_back(to_str(r.read_bytes())); break;
        case 2: expect(wt, WireType::Len, f, r); n.output.push_back(to_str(r.read_bytes())); break;
        case 3: expect(wt, WireType::Len, f, r); n.name = to_str(r.read_bytes()); break;
        case 4: expect(wt, WireType::Len, f, r); n.op_type = to_str(r.read_bytes()); break;
        case 5: expect(wt, WireType::Len, f, r); n.attribute.push_back(parse_attribute(r.read_submessage())); break;
        case 7: expect(wt, WireType::Len, f, r); n.domain = to_str(r.read_bytes()); break;
        default: r.skip(wt);
        }
    }
    return n;
}

// TensorShapeProto.Dimension { dim_value = 1; dim_param = 2; }
void parse_dim(WireReader r, ValueInfo &vi) {
    int64_t value = -1;
    std::string param;
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        if (f == 1 && wt == WireType::Varint) value = int64_t(r.read_varint());
        else if (f == 2 && wt == WireType::Len) param = to_str(r.read_bytes());
        else r.skip(wt);
    }
    vi.dims.push_back(value);
    vi.dim_names.push_back(param);
}

// TypeProto { tensor_type = 1 : TypeProto.Tensor { elem_type = 1; shape = 2 : TensorShapeProto { dim = 1 } } }
void parse_type(WireReader r, ValueInfo &vi) {
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        if (f != 1 || wt != WireType::Len) { r.skip(wt); continue; }
        WireReader tt = r.read_submessage();               // TypeProto.Tensor
        uint32_t f2; WireType wt2;
        while (tt.next_field(f2, wt2)) {
            if (f2 == 1 && wt2 == WireType::Varint) vi.elem_type = DataType(int32_t(tt.read_varint()));
            else if (f2 == 2 && wt2 == WireType::Len) {
                vi.has_shape = true;
                WireReader shape = tt.read_submessage();
                uint32_t f3; WireType wt3;
                while (shape.next_field(f3, wt3)) {
                    if (f3 == 1 && wt3 == WireType::Len) parse_dim(shape.read_submessage(), vi);
                    else shape.skip(wt3);
                }
            } else tt.skip(wt2);
        }
    }
}

ValueInfo parse_value_info(WireReader r) {
    ValueInfo vi;
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        if (f == 1) { expect(wt, WireType::Len, f, r); vi.name = to_str(r.read_bytes()); }
        else if (f == 2) { expect(wt, WireType::Len, f, r); parse_type(r.read_submessage(), vi); }
        else r.skip(wt);
    }
    return vi;
}

GraphProto parse_graph(WireReader r) {
    GraphProto g;
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        switch (f) {
        case 1: expect(wt, WireType::Len, f, r); g.node.push_back(parse_node(r.read_submessage())); break;
        case 2: expect(wt, WireType::Len, f, r); g.name = to_str(r.read_bytes()); break;
        case 5: expect(wt, WireType::Len, f, r); g.initializer.push_back(parse_tensor(r.read_submessage())); break;
        case 11: expect(wt, WireType::Len, f, r); g.input.push_back(parse_value_info(r.read_submessage())); break;
        case 12: expect(wt, WireType::Len, f, r); g.output.push_back(parse_value_info(r.read_submessage())); break;
        default: r.skip(wt);
        }
    }
    return g;
}

OperatorSetId parse_opset(WireReader r) {
    OperatorSetId o;
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        if (f == 1 && wt == WireType::Len) o.domain = to_str(r.read_bytes());
        else if (f == 2 && wt == WireType::Varint) o.version = int64_t(r.read_varint());
        else r.skip(wt);
    }
    return o;
}

// ---------------------------------------------------------------- serializing
WireWriter ser_value_info(const ValueInfo &vi) {
    WireWriter shape;
    if (vi.has_shape)
        for (size_t i = 0; i < vi.dims.size(); i++) {
            WireWriter dim;
            const std::string &nm = i < vi.dim_names.size() ? vi.dim_names[i] : std::string();
            if (!nm.empty()) dim.field_bytes(2, nm);
            else if (vi.dims[i] >= 0) dim.field_varint(1, uint64_t(vi.dims[i]));
            shape.field_message(1, dim);
        }
    WireWriter tensor_type;
    tensor_type.field_varint(1, uint64_t(int32_t(vi.elem_type)));
    if (vi.has_shape) tensor_type.field_message(2, shape);
    WireWriter type;
    type.field_message(1, tensor_type);
    WireWriter w;
    w.field_bytes(1, vi.name);
    w.field_message(2, type);
    return w;
}

WireWriter ser_tensor(const TensorProto &t) {
    WireWriter w;
    for (auto d : t.dims) w.field_varint(1, uint64_t(d));       // dims: unpacked, like the official encoder
    w.field_varint(2, uint64_t(int32_t(t.data_type)));
    w.field_packed_floats(4, t.float_data);
    w.field_bytes(8, t.name);
    if (!t.raw_data.empty()) w.field_bytes(9, t.raw_data);
    for (auto &e : t.external_data) {
        WireWriter ew;
        ew.field_bytes(1, e.key);
        ew.field_bytes(2, e.value);
        w.field_message(13, ew);
    }
    if (t.data_location) w.field_varint(14, uint64_t(t.data_location));
    return w;
}

WireWriter ser_attribute(const AttributeProto &a) {
    WireWriter w;
    w.field_bytes(1, a.name);
    switch (a.type) {
    case AttrType::Float: w.field_float(2, a.f); break;
    case AttrType::Int: w.field_varint(3, uint64_t(a.i)); break;
    case AttrType::String: w.field_bytes(4, a.s); break;
    case AttrType::Floats: for (float x : a.floats) w.field_float(7, x); break;
    case AttrType::Ints: for (auto x : a.ints) w.field_varint(8, uint64_t(x)); break;
    default: break;
    }
    w.field_varint(20, uint64_t(int32_t(a.type)));
    return w;
}

WireWriter ser_node(const NodeProto &n) {
    WireWriter w;
    for (auto &s : n.input) w.field_bytes(1, s);
    for (auto &s : n.output) w.field_bytes(2, s);
    if (!n.name.empty()) w.field_bytes(3, n.name);
    w.field_bytes(4, n.op_type);
    for (auto &a : n.attribute) w.field_message(5, ser_attribute(a));
    if (!n.domain.empty()) w.field_bytes(7, n.domain);
    return w;
}

} // namespace

ModelProto parse_model(const std::string &bytes) {
    WireReader r(bytes);
    ModelProto m;
    m.ir_version = 0;
    bool saw_graph = false;
    uint32_t f; WireType wt;
    while (r.next_field(f, wt)) {
        switch (f) {
        case 1: expect(wt, WireType::Varint, f, r); m.ir_version = int64_t(r.read_varint()); break;
        case 2: expect(wt, WireType::Len, f, r); m.producer_name = to_str(r.read_bytes()); break;
        case 3: expect(wt, WireType::Len, f, r); m.producer_version = to_str(r.read_bytes()); break;
        case 7: expect(wt, WireType::Len, f, r); m.graph = parse_graph(r.read_submessage()); saw_graph = true; break;
        case 8: expect(wt, WireType::Len, f, r); m.opset_import.push_back(parse_opset(r.read_submessage())); break;
        default: r.skip(wt);
        }
    }
    if (!saw_graph) throw FormatError("file has no graph (is this really an .onnx file?)");
    return m;
}

std::string serialize_model(const ModelProto &m) {
    WireWriter g;
    for (auto &n : m.graph.node) g.field_message(1, ser_node(n));
    g.field_bytes(2, m.graph.name);
    for (auto &t : m.graph.initializer) g.field_message(5, ser_tensor(t));
    for (auto &vi : m.graph.input) g.field_message(11, ser_value_info(vi));
    for (auto &vi : m.graph.output) g.field_message(12, ser_value_info(vi));

    WireWriter w;
    w.field_varint(1, uint64_t(m.ir_version));
    if (!m.producer_name.empty()) w.field_bytes(2, m.producer_name);
    if (!m.producer_version.empty()) w.field_bytes(3, m.producer_version);
    w.field_message(7, g);
    for (auto &o : m.opset_import) {
        WireWriter ow;
        if (!o.domain.empty()) ow.field_bytes(1, o.domain);
        ow.field_varint(2, uint64_t(o.version));
        w.field_message(8, ow);
    }
    return w.take();
}

ModelProto load_model_file(const std::string &path, std::string *base_dir) {
    std::ifstream f(path, std::ios::binary);                       // binary! never text mode on Windows
    if (!f) throw std::runtime_error("cannot open model file '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    if (base_dir) {
        auto slash = path.find_last_of("/\\");
        *base_dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
    }
    return parse_model(ss.str());
}

void save_model_file(const ModelProto &model, const std::string &path, const SaveOptions &opt) {
    ModelProto m = model;                                            // we rewrite the weights' storage
    std::string data_file_name, data_bytes;
    if (opt.external_data) {
        auto slash = path.find_last_of("/\\");
        data_file_name = (slash == std::string::npos ? path : path.substr(slash + 1)) + ".data";
    }
    for (auto &t : m.graph.initializer) {
        // 1. Bring the numbers into raw little-endian bytes (if they are in float_data).
        if (t.raw_data.empty() && !t.float_data.empty()) {
            t.raw_data.resize(t.float_data.size() * 4);
            for (size_t i = 0; i < t.float_data.size(); i++) {
                uint32_t bits;
                std::memcpy(&bits, &t.float_data[i], 4);
                for (int b = 0; b < 4; b++) t.raw_data[i * 4 + b] = char((bits >> (8 * b)) & 0xFF);
            }
        }
        const bool go_external = opt.external_data && int64_t(t.raw_data.size()) >= opt.external_threshold;
        if (go_external) {
            // 2a. Append to the side file, padded so each tensor starts on an aligned offset.
            while (int64_t(data_bytes.size()) % opt.external_alignment) data_bytes.push_back('\0');
            const size_t offset = data_bytes.size();
            data_bytes += t.raw_data;
            t.external_data = {{"location", data_file_name},
                               {"offset", std::to_string(offset)},
                               {"length", std::to_string(t.raw_data.size())}};
            t.data_location = 1;
            t.raw_data.clear();
            t.float_data.clear();
        } else if (opt.use_raw_data) {
            t.float_data.clear();                                    // 2b. keep raw_data only
        } else {
            // 2c. keep float_data only (decode the raw bytes back)
            t.float_data.resize(t.raw_data.size() / 4);
            for (size_t i = 0; i < t.float_data.size(); i++) {
                uint32_t bits = 0;
                for (int b = 0; b < 4; b++) bits |= uint32_t(uint8_t(t.raw_data[i * 4 + b])) << (8 * b);
                std::memcpy(&t.float_data[i], &bits, 4);
            }
            t.raw_data.clear();
        }
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write '" + path + "'");
    const std::string bytes = serialize_model(m);
    f.write(bytes.data(), std::streamsize(bytes.size()));
    if (opt.external_data) {
        auto slash = path.find_last_of("/\\");
        std::string dir = slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
        std::ofstream d(dir + data_file_name, std::ios::binary);
        if (!d) throw std::runtime_error("cannot write external data file");
        d.write(data_bytes.data(), std::streamsize(data_bytes.size()));
    }
}

AttributeProto make_attr(const std::string &name, float f) {
    AttributeProto a; a.name = name; a.f = f; a.type = AttrType::Float; return a;
}
AttributeProto make_attr(const std::string &name, int64_t i) {
    AttributeProto a; a.name = name; a.i = i; a.type = AttrType::Int; return a;
}

} // namespace mo
