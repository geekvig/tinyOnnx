// =====================================================================================
// STAGE 1 · protobuf_wire.hpp  --  the Protocol Buffers *wire format*, by hand
// =====================================================================================
// A real .onnx file is a protobuf message (ModelProto). Protobuf is NOT text: it is a
// stream of (tag, value) pairs. Every pair starts with a "tag" varint:
//
//        tag = (field_number << 3) | wire_type
//
//   wire_type 0 = VARINT  : an integer, 7 bits per byte, high bit = "more bytes follow"
//   wire_type 1 = I64     : exactly 8 bytes, little-endian (double, fixed64)
//   wire_type 2 = LEN     : a varint length N, then N bytes (strings, bytes, sub-messages,
//                           and "packed" arrays of numbers)
//   wire_type 5 = I32     : exactly 4 bytes, little-endian (float, fixed32)
//
// Field NUMBERS (not names) are what is stored on disk. The names live only in onnx.proto.
// A reader that meets a field number it does not know simply SKIPS it -- that is how
// protobuf stays forward compatible, and why we can read real ONNX files while only
// understanding a small subset of the schema.
//
// Two tiny classes:  WireReader (bytes -> values)   and   WireWriter (values -> bytes).
// =====================================================================================
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mo {

// Thrown for any malformed file. Carries the byte offset so you can find the problem with
// a hex editor (or with tools/inspect_onnx --wire).
struct FormatError : std::runtime_error {
    explicit FormatError(const std::string &m) : std::runtime_error("format error: " + m) {}
};

enum class WireType : uint8_t { Varint = 0, I64 = 1, Len = 2, I32 = 5 };

inline const char *wire_type_name(WireType w) {
    switch (w) {
    case WireType::Varint: return "VARINT";
    case WireType::I64: return "I64";
    case WireType::Len: return "LEN";
    case WireType::I32: return "I32";
    }
    return "?";
}

// -------------------------------------------------------------------------------------
// WireReader: a cursor over a byte range. It never owns memory; sub-messages are just
// smaller WireReaders over a slice of the same buffer (zero copies).
// -------------------------------------------------------------------------------------
class WireReader {
public:
    WireReader(const uint8_t *begin, const uint8_t *end, size_t base_offset = 0)
        : begin_(begin), p_(begin), end_(end), base_(base_offset) {}
    explicit WireReader(std::string_view s)
        : WireReader(reinterpret_cast<const uint8_t *>(s.data()),
                     reinterpret_cast<const uint8_t *>(s.data()) + s.size()) {}
    // A reader never owns its bytes. Building one from a TEMPORARY string would leave it
    // pointing at freed memory (AddressSanitizer: stack-use-after-scope), so forbid it.
    explicit WireReader(std::string &&) = delete;

    bool at_end() const { return p_ >= end_; }
    size_t size() const { return size_t(end_ - begin_); }   // bytes in this message
    size_t offset() const { return base_ + size_t(p_ - begin_); }   // absolute file offset

    // Read the next tag. Returns false at the end of this message.
    bool next_field(uint32_t &field, WireType &wt) {
        if (at_end()) return false;
        const size_t at = offset();
        uint64_t tag = read_varint();
        field = uint32_t(tag >> 3);
        uint32_t w = uint32_t(tag & 7);
        if (field == 0) fail("field number 0 is illegal", at);
        if (w != 0 && w != 1 && w != 2 && w != 5)
            fail("unsupported wire type " + std::to_string(w) + " for field " + std::to_string(field), at);
        wt = WireType(w);
        return true;
    }

    // VARINT: little-endian base-128. Byte b contributes (b & 0x7F) << (7 * index).
    //   300 = 0b1_0010_1100  ->  bytes 0xAC 0x02
    //         0xAC = 1 0101100  (more follows, low 7 bits = 0101100)
    //         0x02 = 0 0000010  (last byte,    next 7 bits = 0000010)
    uint64_t read_varint() {
        uint64_t result = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (p_ >= end_) fail("truncated varint", offset());
            uint8_t b = *p_++;
            result |= uint64_t(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
        }
        fail("varint longer than 10 bytes", offset());
    }

    uint32_t read_fixed32() {
        need(4);
        uint32_t v = uint32_t(p_[0]) | uint32_t(p_[1]) << 8 | uint32_t(p_[2]) << 16 | uint32_t(p_[3]) << 24;
        p_ += 4;
        return v;
    }
    uint64_t read_fixed64() {
        uint64_t lo = read_fixed32(), hi = read_fixed32();
        return lo | (hi << 32);
    }
    float read_float() {                           // I32 wire type -> IEEE-754 float
        uint32_t bits = read_fixed32();
        float f;
        std::memcpy(&f, &bits, 4);                 // memcpy = the legal way to reinterpret bits
        return f;
    }

    // LEN: returns a view of the payload bytes and moves past them.
    std::string_view read_bytes() {
        uint64_t n = read_varint();
        if (n > uint64_t(end_ - p_)) fail("length " + std::to_string(n) + " runs past end of message", offset());
        std::string_view v(reinterpret_cast<const char *>(p_), size_t(n));
        p_ += n;
        return v;
    }
    // LEN holding a nested message: a reader over exactly those bytes.
    WireReader read_submessage() {
        size_t payload_offset;
        std::string_view v = read_bytes_at(payload_offset);
        auto *b = reinterpret_cast<const uint8_t *>(v.data());
        return WireReader(b, b + v.size(), payload_offset);
    }

    // Skip a field we do not understand (forward compatibility).
    void skip(WireType wt) {
        switch (wt) {
        case WireType::Varint: read_varint(); break;
        case WireType::I64: need(8); p_ += 8; break;
        case WireType::Len: read_bytes(); break;
        case WireType::I32: need(4); p_ += 4; break;
        }
    }

    // Repeated numeric fields may arrive "packed" (one LEN blob holding many values) or
    // "unpacked" (one tag per value). A correct reader must accept both.
    void read_repeated_int64(WireType wt, std::vector<int64_t> &out) {
        if (wt == WireType::Varint) { out.push_back(int64_t(read_varint())); return; }
        if (wt != WireType::Len) fail("bad wire type for repeated int64", offset());
        WireReader packed = read_submessage();
        while (!packed.at_end()) out.push_back(int64_t(packed.read_varint()));
    }
    void read_repeated_float(WireType wt, std::vector<float> &out) {
        if (wt == WireType::I32) { out.push_back(read_float()); return; }
        if (wt != WireType::Len) fail("bad wire type for repeated float", offset());
        WireReader packed = read_submessage();
        while (!packed.at_end()) out.push_back(packed.read_float());
    }

    [[noreturn]] static void fail(const std::string &msg, size_t at) {
        throw FormatError(msg + " (at byte " + std::to_string(at) + ")");
    }

private:
    void need(size_t n) const {
        if (size_t(end_ - p_) < n) fail("unexpected end of data", offset());
    }
    std::string_view read_bytes_at(size_t &payload_offset) {
        uint64_t n = read_varint();
        payload_offset = offset();
        if (n > uint64_t(end_ - p_)) fail("length " + std::to_string(n) + " runs past end of message", offset());
        std::string_view v(reinterpret_cast<const char *>(p_), size_t(n));
        p_ += n;
        return v;
    }

    const uint8_t *begin_, *p_, *end_;
    size_t base_;
};

// -------------------------------------------------------------------------------------
// WireWriter: appends encoded fields to a std::string (used as a byte buffer).
// Sub-messages are encoded into their own WireWriter first, because the LEN prefix must
// be known before the payload is written.
// -------------------------------------------------------------------------------------
class WireWriter {
public:
    void varint(uint64_t v) {
        while (v >= 0x80) { buf_.push_back(char(uint8_t(v) | 0x80)); v >>= 7; }
        buf_.push_back(char(uint8_t(v)));
    }
    void tag(uint32_t field, WireType wt) { varint((uint64_t(field) << 3) | uint32_t(wt)); }

    void field_varint(uint32_t f, uint64_t v) { tag(f, WireType::Varint); varint(v); }
    void field_float(uint32_t f, float v) {
        tag(f, WireType::I32);
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        for (int i = 0; i < 4; i++) buf_.push_back(char((bits >> (8 * i)) & 0xFF));
    }
    void field_bytes(uint32_t f, std::string_view s) {
        tag(f, WireType::Len);
        varint(s.size());
        buf_.append(s.data(), s.size());
    }
    void field_message(uint32_t f, const WireWriter &sub) { field_bytes(f, sub.buf_); }

    void field_packed_floats(uint32_t f, const std::vector<float> &v) {
        if (v.empty()) return;
        WireWriter tmp;
        for (float x : v) {
            uint32_t bits;
            std::memcpy(&bits, &x, 4);
            for (int i = 0; i < 4; i++) tmp.buf_.push_back(char((bits >> (8 * i)) & 0xFF));
        }
        field_bytes(f, tmp.buf_);
    }

    const std::string &bytes() const { return buf_; }
    std::string take() { return std::move(buf_); }

private:
    std::string buf_;
};

} // namespace mo
