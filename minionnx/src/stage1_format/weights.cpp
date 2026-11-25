// =====================================================================================
// STAGE 1 · weights.cpp  --  decode initializers into the aligned arena
// =====================================================================================
#include "stage1_format/weights.hpp"
#include "stage1_format/protobuf_wire.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace mo {

namespace {

bool host_is_little_endian() {
    const uint32_t one = 1;
    uint8_t first;
    std::memcpy(&first, &one, 1);
    return first == 1;
}

size_t align_up(size_t x) { return (x + kAlignment - 1) / kAlignment * kAlignment; }

std::string ext_value(const TensorProto &t, const std::string &key) {
    for (auto &e : t.external_data) if (e.key == key) return e.value;
    return {};
}

// Validate an external "location": it must be a plain relative path inside the model folder.
// A malicious model could otherwise say location="../../secret" or "/etc/passwd".
void check_location(const std::string &loc, const std::string &tensor) {
    if (loc.empty()) throw FormatError("tensor '" + tensor + "': external data has no location");
    if (loc[0] == '/' || loc[0] == '\\' || loc.find(':') != std::string::npos || loc.find("..") != std::string::npos)
        throw FormatError("tensor '" + tensor + "': unsafe external data location '" + loc + "'");
}

} // namespace

void decode_le_floats(const char *src, size_t n, float *dst) {
    if (host_is_little_endian()) {
        if (n) std::memcpy(dst, src, n * 4);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = 0;
        for (int b = 0; b < 4; b++) bits |= uint32_t(uint8_t(src[i * 4 + b])) << (8 * b);
        std::memcpy(&dst[i], &bits, 4);
    }
}

void WeightStore::load(const GraphProto &g, const std::string &base_dir) {
    tensors_.clear();
    records_.clear();

    // ---- pass 1: validate every tensor and plan where it goes in the arena ----
    size_t total = 0;
    for (const TensorProto &t : g.initializer) {
        if (t.name.empty()) throw FormatError("initializer without a name");
        if (t.data_type != DataType::Float)
            throw FormatError("initializer '" + t.name + "': data_type " + std::to_string(int(t.data_type)) +
                              " not supported (MiniOnnx computes in float32 only)");
        for (int64_t d : t.dims)
            if (d < 0) throw FormatError("initializer '" + t.name + "': negative dimension");
        const int64_t n = t.numel();
        Record r;
        r.name = t.name;
        r.shape = t.dims;
        r.bytes = size_t(n) * 4;
        if (t.data_location == 1) {
            const std::string loc = ext_value(t, "location");
            check_location(loc, t.name);
            r.source = "external:" + loc + "@" + (ext_value(t, "offset").empty() ? "0" : ext_value(t, "offset"));
        } else if (!t.raw_data.empty()) {
            if (t.raw_data.size() != r.bytes)
                throw FormatError("initializer '" + t.name + "': raw_data has " + std::to_string(t.raw_data.size()) +
                                  " bytes but dims " + shape_str(t.dims) + " need " + std::to_string(r.bytes));
            r.source = "raw_data";
        } else {
            if (int64_t(t.float_data.size()) != n)
                throw FormatError("initializer '" + t.name + "': float_data has " + std::to_string(t.float_data.size()) +
                                  " values but dims " + shape_str(t.dims) + " need " + std::to_string(n));
            r.source = "float_data";
        }
        r.arena_offset = total;
        total = align_up(total + r.bytes);
        records_.push_back(r);
    }

    // ---- pass 2: one allocation, then copy every tensor into its slot ----
    arena_ = aligned_bytes(total);
    arena_bytes_ = total;
    std::map<std::string, std::string> file_cache;              // external file -> contents
    for (size_t i = 0; i < g.initializer.size(); i++) {
        const TensorProto &t = g.initializer[i];
        Record &r = records_[i];
        float *dst = reinterpret_cast<float *>(arena_.get() + r.arena_offset);
        const size_t n = r.bytes / 4;
        if (r.source == "float_data") {
            if (n) std::memcpy(dst, t.float_data.data(), r.bytes);
        } else if (r.source == "raw_data") {
            decode_le_floats(t.raw_data.data(), n, dst);
        } else {
            const std::string loc = ext_value(t, "location");
            auto it = file_cache.find(loc);
            if (it == file_cache.end()) {
                std::ifstream f(base_dir + "/" + loc, std::ios::binary);
                if (!f) throw FormatError("initializer '" + t.name + "': cannot open external file '" + loc + "'");
                std::ostringstream ss;
                ss << f.rdbuf();
                it = file_cache.emplace(loc, ss.str()).first;
            }
            const std::string &blob = it->second;
            const std::string off_s = ext_value(t, "offset"), len_s = ext_value(t, "length");
            const size_t off = off_s.empty() ? 0 : size_t(std::stoull(off_s));
            const size_t len = len_s.empty() ? r.bytes : size_t(std::stoull(len_s));
            if (len != r.bytes)
                throw FormatError("initializer '" + t.name + "': external length " + std::to_string(len) +
                                  " != " + std::to_string(r.bytes) + " bytes implied by dims");
            if (off > blob.size() || blob.size() - off < len)
                throw FormatError("initializer '" + t.name + "': external range runs past end of '" + loc + "'");
            decode_le_floats(blob.data() + off, n, dst);
        }
        tensors_[t.name] = Tensor::view(t.dims, arena_, dst);
    }
}

std::string WeightStore::summary() const {
    std::string s;
    char line[256];
    std::snprintf(line, sizeof line, "  %-14s %-14s %10s %10s  %s\n", "name", "shape", "offset", "bytes", "source");
    s += line;
    for (auto &r : records_) {
        std::snprintf(line, sizeof line, "  %-14s %-14s %10zu %10zu  %s\n", r.name.c_str(), shape_str(r.shape).c_str(),
                      r.arena_offset, r.bytes, r.source.c_str());
        s += line;
    }
    std::snprintf(line, sizeof line, "  arena: %zu bytes, every tensor 64-byte aligned\n", arena_bytes_);
    s += line;
    return s;
}

} // namespace mo
