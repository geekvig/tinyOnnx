// =====================================================================================
// CORE · tensor.hpp  --  the runtime tensor: a shape + a 64-byte-aligned float buffer
// =====================================================================================
// Why 64-byte alignment?  A cache line is 64 bytes, and SIMD loads (AVX2 = 32 bytes,
// AVX-512 = 64 bytes, NEON = 16 bytes) are fastest when data starts on such a boundary.
// We use C++17 aligned operator new, which works on Linux, macOS AND Windows/MinGW
// (std::aligned_alloc does NOT exist on Windows -- a classic portability trap).
//
// The buffer is held by a shared_ptr, so copying a Tensor is cheap (it shares the data).
// A Tensor can also be a VIEW into memory owned by something else (e.g. the weight arena)
// using shared_ptr's "aliasing constructor": the view keeps the owner alive.
// =====================================================================================
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace mo {

using Shape = std::vector<int64_t>;
constexpr size_t kAlignment = 64;

inline int64_t shape_numel(const Shape &s) {
    int64_t n = 1;
    for (int64_t d : s) {
        if (d < 0) throw std::runtime_error("negative dimension in shape");
        n *= d;
    }
    return n;
}

inline std::string shape_str(const Shape &s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); i++) r += (i ? "," : "") + std::to_string(s[i]);
    return r + "]";
}

// One aligned allocation, freed with the matching aligned delete.
inline std::shared_ptr<uint8_t> aligned_bytes(size_t bytes) {
    bytes = std::max<size_t>(bytes, 1);
    bytes = (bytes + kAlignment - 1) / kAlignment * kAlignment;
    void *p = ::operator new(bytes, std::align_val_t(kAlignment));
    return std::shared_ptr<uint8_t>(static_cast<uint8_t *>(p),
                                    [](uint8_t *q) { ::operator delete(q, std::align_val_t(kAlignment)); });
}

class Tensor {
public:
    Tensor() = default;

    // Uninitialized storage for `shape`.
    static Tensor empty(Shape shape) {
        Tensor t;
        t.numel_ = shape_numel(shape);
        t.shape_ = std::move(shape);
        auto bytes = aligned_bytes(size_t(t.numel_) * sizeof(float));
        t.buf_ = std::shared_ptr<float>(bytes, reinterpret_cast<float *>(bytes.get()));  // aliasing ctor
        return t;
    }
    static Tensor zeros(Shape shape) {
        Tensor t = empty(std::move(shape));
        std::fill(t.data(), t.data() + t.numel_, 0.f);
        return t;
    }
    // Copy `values` in; the element count MUST match the shape (this check prevents the
    // heap-buffer-overflow found in the original tinyONNX, where a feed could lie about its size).
    static Tensor from(Shape shape, const std::vector<float> &values) {
        Tensor t = empty(std::move(shape));
        if (int64_t(values.size()) != t.numel_)
            throw std::runtime_error("Tensor::from: shape " + shape_str(t.shape_) + " needs " +
                                     std::to_string(t.numel_) + " values, got " + std::to_string(values.size()));
        if (t.numel_) std::memcpy(t.data(), values.data(), values.size() * sizeof(float));
        return t;
    }
    // A view of `numel(shape)` floats at `data`, kept alive by `owner`.
    static Tensor view(Shape shape, const std::shared_ptr<uint8_t> &owner, float *data) {
        Tensor t;
        t.numel_ = shape_numel(shape);
        t.shape_ = std::move(shape);
        t.buf_ = std::shared_ptr<float>(owner, data);
        return t;
    }

    const Shape &shape() const { return shape_; }
    int rank() const { return int(shape_.size()); }
    int64_t dim(int i) const { return shape_.at(size_t(i)); }
    int64_t numel() const { return numel_; }
    bool valid() const { return buf_ != nullptr || numel_ == 0; }

    float *data() { return buf_.get(); }
    const float *data() const { return buf_.get(); }
    float &operator[](int64_t i) { return buf_.get()[i]; }
    float operator[](int64_t i) const { return buf_.get()[i]; }

    // Matrix view used by the kernels: everything except the last dim is "rows".
    int64_t rows() const { return shape_.empty() ? 1 : numel_ / std::max<int64_t>(1, shape_.back()); }
    int64_t cols() const { return shape_.empty() ? 1 : shape_.back(); }

    std::vector<float> to_vector() const { return std::vector<float>(data(), data() + numel_); }

private:
    Shape shape_;
    std::shared_ptr<float> buf_;
    int64_t numel_ = 0;
};

} // namespace mo
