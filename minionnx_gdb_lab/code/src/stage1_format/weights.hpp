// =====================================================================================
// STAGE 1 · weights.hpp  --  WeightStore: where the trained numbers live at run time
// =====================================================================================
// A model's weights ("initializers") arrive in one of three encodings (see TensorProto).
// The WeightStore decodes all of them into ONE big, 64-byte-aligned block of memory
// (an "arena"), and hands out Tensor VIEWS into it.
//
//   arena:  | W1 ......... |pad| b1 .. |pad| W2 ............ |pad| b2 |pad|
//           ^0             ^64*k        ^64*j
//
// Why one arena instead of one allocation per weight?
//   * one malloc instead of hundreds (faster load, less fragmentation)
//   * every tensor starts on a cache-line boundary (good for SIMD)
//   * the weights are read-only after load, so all threads can share them with no locks
//   * this is exactly the layout a memory-mapped weight file would have
// =====================================================================================
#pragma once
#include "core/tensor.hpp"
#include "stage1_format/onnx_model.hpp"

#include <map>
#include <string>

namespace mo {

class WeightStore {
public:
    struct Record {                 // bookkeeping, printed by summary()
        std::string name;
        Shape shape;
        std::string source;         // "float_data" | "raw_data" | "external:<file>@<offset>"
        size_t arena_offset = 0;
        size_t bytes = 0;
    };

    // Decode every initializer of `g`. `base_dir` is the folder of the .onnx file
    // (external data paths are relative to it). Throws on any inconsistency.
    void load(const GraphProto &g, const std::string &base_dir = ".");

    const Tensor *find(const std::string &name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    const std::map<std::string, Tensor> &all() const { return tensors_; }
    // Add a tensor produced at load time (e.g. by constant folding in the optimizer).
    void add(const std::string &name, Tensor t) { tensors_[name] = std::move(t); }
    void remove(const std::string &name) { tensors_.erase(name); }

    size_t arena_bytes() const { return arena_bytes_; }
    const std::vector<Record> &records() const { return records_; }
    std::string summary() const;

private:
    std::shared_ptr<uint8_t> arena_;
    size_t arena_bytes_ = 0;
    std::map<std::string, Tensor> tensors_;
    std::vector<Record> records_;
};

// Decode `n` little-endian float32 values. On a little-endian CPU (x86, ARM in its usual
// mode) this is a plain memcpy; on a big-endian CPU every 4-byte group is reversed.
void decode_le_floats(const char *src, size_t n, float *dst);

} // namespace mo
