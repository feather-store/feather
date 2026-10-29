#pragma once
// ── File format v10: the codec ──────────────────────────────────────────
//
// Pure encoding helpers for the append-only format. No DB state lives here;
// feather.h owns what goes into the sections and when they are written.
//
//   [0,    4096)  header slot 0
//   [4096, 8192)  header slot 1
//   [8192, end)   sections, appended by successive commits
//
// A commit appends its sections past the committed end, makes them durable,
// then writes a header with generation+1 into slot (generation+1) % 2 and
// makes that durable. The previous header stays intact in the other slot, so a
// crash at any point leaves at least one valid header whose sections are all
// on disk: a torn header fails its checksum, a torn section lies past every
// valid header's end. Nothing ever rewrites a byte a valid header covers.
//
// Header (64 bytes, rest of the slot zero):
//   0  u32 magic "FEAT"         4  u32 version = 10
//   8  u32 0xFFFFFFFF           -- read by v3-v9 builds as the metadata count,
//                                  which they reject against the file size
//                                  instead of misparsing the file
//   12 u32 slot                 16 u64 generation
//   24 u64 committed end        32 u64 data start (8192)
//   40 u64 unix time of commit  48 u64 reserved
//   56 u64 xxh64 of bytes [0,56)
//
// Section: [type u32][flags u32][payload_len u64][payload][xxh64 u64]
//
// VECTORS payload: [name str16][count u32][stride u32][pad u32][pad zero
// bytes][count x stride bytes] — pad places the vectors on a MAP_ALIGN
// boundary, so a read-only process maps them instead of copying them, and N
// readers share one copy. Written by a full rewrite; the GRAPH records of
// that image then carry VEC_IMAGE (no vector bytes of their own).
//
// FULLVEC payload: [name str16][dim u32][count u32] then count x
// [slot u32][float32 x dim] — the full-precision vector of a slot of an
// in-RAM int8 modality, read back from the file (never held in RAM) to re-rank
// int8 search candidates. Always written AFTER the GRAPH records of the same
// commit, and a GRAPH record carrying a slot's data voids any earlier FULLVEC
// for it, so a build that does not write FULLVEC cannot leave a stale one
// behind. Flagged SEC_OPTIONAL: a build without it loads the int8 index alone.
//
// VERSIONS payload: [clock u64] then [id u64][version u64] records — the
// compare-and-set version of each record put by the same commit's META
// sections (always written after them: a META put resets the version), and
// the DB's version clock, so a version is never reused even once the record
// that held the highest one is deleted. Flagged SEC_OPTIONAL: a build without
// it loads the data and restarts the versions.
//
// IDEMPOTENCY payload: records of [key str16][fingerprint str16][created i64]
// [count u32][count x (id u64, version u64)][note str32] — the result of a write batch
// that carried an idempotency key, so a retry returns it instead of applying
// the batch again. A full rewrite keeps only unexpired keys. SEC_OPTIONAL.
//
// MODALITY payloads may carry trailing fields, ignored by builds that predate
// them: [refine u8][int8_train u8] (see DB::set_int8_ram).
//   checksum = xxh64(payload, seed = xxh64(the 16 header bytes, 0))
#include <array>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>

namespace feather {
namespace v10 {

constexpr uint32_t MAGIC        = 0x46454154u;   // "FEAT"
constexpr uint32_t VERSION      = 10;
constexpr uint32_t LEGACY_GUARD = 0xFFFFFFFFu;
constexpr uint64_t SLOT_SIZE    = 4096;
constexpr uint64_t DATA_START   = 2 * SLOT_SIZE;
constexpr size_t   HEADER_LEN   = 64;
constexpr size_t   SECTION_HDR  = 16;
constexpr size_t   SECTION_TAIL = 8;
// Target payload size. Sections are chunked so neither a commit nor a load
// ever buffers more than this (plus one record) in memory.
constexpr size_t   CHUNK        = 8u << 20;

enum Section : uint32_t {
    SEC_MODALITY = 1,   // start of a full modality image: (re)create the index
    SEC_GRAPH    = 2,   // HNSW slots: full records and link-list-only records
    SEC_META     = 3,   // metadata puts and deletes
    SEC_SALIENCE = 4,   // recall_count / last_recalled_at only
    SEC_VECTORS  = 5,   // one modality's vectors, contiguous, in slot order
    SEC_FULLVEC  = 6,   // float32 originals of an in-RAM int8 modality (optional)
    SEC_VERSIONS = 7,   // compare-and-set versions + the version clock (optional)
    SEC_IDEMPOTENCY = 8,  // results of keyed write batches (optional)
};
// A VECTORS section's vector bytes start at a multiple of this in the file,
// so a reader can map them directly on 4K, 16K (Apple silicon) and 64K pages.
constexpr uint64_t MAP_ALIGN = 64u << 10;
// Section flag: a reader that does not know the type may skip it. Without the
// flag an unknown type means "written by a newer build" and the load fails.
constexpr uint32_t SEC_OPTIONAL = 1;

// GRAPH record kinds (bits). LINKS is always present.
constexpr uint8_t REC_LINKS = 0x01;
constexpr uint8_t REC_DATA  = 0x02;   // label, level, vector

// GRAPH vector encodings.
constexpr uint8_t VEC_RAW  = 0;   // the index's storage bytes, verbatim
constexpr uint8_t VEC_INT8 = 1;   // [scale f32][int8 x dim] — set_quantized()
constexpr uint8_t VEC_IMAGE = 2;  // none: the vector is in this image's VECTORS section

constexpr uint8_t META_PUT = 1;
constexpr uint8_t META_DEL = 2;

// ── XXH64 ────────────────────────────────────────────────────────────────
// The section checksum covers every byte a load reads, so its speed is load
// time. A bytewise CRC32 runs at ~0.5 GB/s, which would have added ~0.4 s to
// a 182 MB cold load; XXH64 runs near memory bandwidth. Standard XXH64, so
// any xxhash implementation can verify a file.
namespace detail {
constexpr uint64_t P1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t P2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t P3 = 0x165667B19E3779F9ULL;
constexpr uint64_t P4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t P5 = 0x27D4EB2F165667C5ULL;
inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t rd64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t round(uint64_t acc, uint64_t in) {
    acc += in * P2;
    acc  = rotl(acc, 31);
    return acc * P1;
}
inline uint64_t merge(uint64_t acc, uint64_t val) {
    acc ^= round(0, val);
    return acc * P1 + P4;
}
} // namespace detail

inline uint64_t xxh64(const void* data, size_t len, uint64_t seed = 0) {
    using namespace detail;
    const uint8_t* p   = static_cast<const uint8_t*>(data);
    const uint8_t* end = p + len;
    uint64_t h;
    if (len >= 32) {
        const uint8_t* limit = end - 32;
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        do {
            v1 = round(v1, rd64(p)); p += 8;
            v2 = round(v2, rd64(p)); p += 8;
            v3 = round(v3, rd64(p)); p += 8;
            v4 = round(v4, rd64(p)); p += 8;
        } while (p <= limit);
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
    } else {
        h = seed + P5;
    }
    h += static_cast<uint64_t>(len);
    while (p + 8 <= end) {
        h ^= round(0, rd64(p));
        h  = rotl(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= static_cast<uint64_t>(rd32(p)) * P1;
        h  = rotl(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<uint64_t>(*p) * P5;
        h  = rotl(h, 11) * P1;
        ++p;
    }
    h ^= h >> 33; h *= P2;
    h ^= h >> 29; h *= P3;
    h ^= h >> 32;
    return h;
}

// Streaming XXH64, same result as xxh64() over the concatenated input. The
// VECTORS section is hundreds of MB: it is hashed straight out of the mapping
// or the file, never through a copy of itself.
class Xxh64Stream {
    uint64_t v_[4], seed_, total_ = 0;
    uint8_t  buf_[32];
    size_t   used_ = 0;
public:
    explicit Xxh64Stream(uint64_t seed = 0) : seed_(seed) {
        using namespace detail;
        v_[0] = seed + P1 + P2; v_[1] = seed + P2; v_[2] = seed; v_[3] = seed - P1;
    }
    void update(const void* data, size_t len) {
        using namespace detail;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        const uint8_t* end = p + len;
        total_ += len;
        if (used_ + len < 32) { std::memcpy(buf_ + used_, p, len); used_ += len; return; }
        if (used_) {
            const size_t fill = 32 - used_;
            std::memcpy(buf_ + used_, p, fill);
            for (int i = 0; i < 4; ++i) v_[i] = round(v_[i], rd64(buf_ + 8 * i));
            p += fill;
            used_ = 0;
        }
        while (p + 32 <= end) {
            for (int i = 0; i < 4; ++i) v_[i] = round(v_[i], rd64(p + 8 * i));
            p += 32;
        }
        used_ = static_cast<size_t>(end - p);
        std::memcpy(buf_, p, used_);
    }
    uint64_t digest() const {
        using namespace detail;
        uint64_t h;
        if (total_ >= 32) {
            h = rotl(v_[0], 1) + rotl(v_[1], 7) + rotl(v_[2], 12) + rotl(v_[3], 18);
            for (int i = 0; i < 4; ++i) h = merge(h, v_[i]);
        } else {
            h = seed_ + P5;
        }
        h += total_;
        const uint8_t* p = buf_;
        const uint8_t* end = buf_ + used_;
        while (p + 8 <= end) { h ^= round(0, rd64(p)); h = rotl(h, 27) * P1 + P4; p += 8; }
        if (p + 4 <= end) { h ^= static_cast<uint64_t>(rd32(p)) * P1; h = rotl(h, 23) * P2 + P3; p += 4; }
        while (p < end) { h ^= static_cast<uint64_t>(*p) * P5; h = rotl(h, 11) * P1; ++p; }
        h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
        return h;
    }
};

inline uint64_t section_checksum(const char* hdr16, const char* payload, uint64_t len) {
    return xxh64(payload, static_cast<size_t>(len), xxh64(hdr16, SECTION_HDR, 0));
}

// ── Little-endian field helpers (the whole format is little-endian) ─────
// Slot counts are stored as 32 bits: they count hnswlib's internal ids,
// which are 32-bit, so a wider field could only ever hold a corrupt value.
inline uint32_t count32(uint64_t n) {
    if (n > UINT32_MAX) throw std::length_error("more than 2^32 vectors in one modality");
    return static_cast<uint32_t>(n);
}

template <class T> inline void put(std::string& out, T v) {
    out.append(reinterpret_cast<const char*>(&v), sizeof(T));
}
inline void put_str16(std::string& out, const std::string& s) {
    put<uint16_t>(out, static_cast<uint16_t>(s.size()));
    out.append(s);
}

// Bounds-checked cursor over one section payload. Every read is checked
// against the bytes that are actually there: a count read off disk is a loop
// trip count, and a checksum only proves the bytes are the ones written.
struct Reader {
    const char* p;
    const char* end;
    Reader(const char* b, size_t n) : p(b), end(b + n) {}
    size_t left() const { return static_cast<size_t>(end - p); }
    void need(size_t n) const {
        if (left() < n) throw std::runtime_error("corrupt .feather: section record overruns its section");
    }
    template <class T> T get() {
        need(sizeof(T));
        T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T);
        return v;
    }
    const char* take(size_t n) { need(n); const char* q = p; p += n; return q; }
    std::string str16() {
        uint16_t n = get<uint16_t>();
        return std::string(take(n), n);
    }
};

// ── Header ──────────────────────────────────────────────────────────────
struct Header {
    uint32_t slot       = 0;
    uint64_t generation = 0;
    uint64_t end        = DATA_START;
    uint64_t time       = 0;
};

inline std::array<char, SLOT_SIZE> encode_header(const Header& h) {
    std::array<char, SLOT_SIZE> b{};
    auto w32 = [&](size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); };
    auto w64 = [&](size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); };
    w32(0, MAGIC); w32(4, VERSION); w32(8, LEGACY_GUARD); w32(12, h.slot);
    w64(16, h.generation); w64(24, h.end); w64(32, DATA_START); w64(40, h.time);
    w64(48, 0);
    w64(56, xxh64(b.data(), 56, 0));
    return b;
}

enum class HeaderState { NOT_FEATHER, NEWER, INVALID, VALID };

// `file_size` bounds the committed end: a header claiming bytes the file does
// not have is not a usable header (the file was truncated after the commit).
inline HeaderState decode_header(const char* b, size_t n, uint64_t file_size, Header& out) {
    if (n < HEADER_LEN) return HeaderState::NOT_FEATHER;
    uint32_t magic, version;
    std::memcpy(&magic, b, 4);
    std::memcpy(&version, b + 4, 4);
    if (magic != MAGIC) return HeaderState::NOT_FEATHER;
    uint64_t sum;
    std::memcpy(&sum, b + 56, 8);
    if (xxh64(b, 56, 0) != sum) return HeaderState::INVALID;
    if (version > VERSION) return HeaderState::NEWER;
    if (version != VERSION) return HeaderState::INVALID;
    uint64_t start;
    std::memcpy(&out.slot, b + 12, 4);
    std::memcpy(&out.generation, b + 16, 8);
    std::memcpy(&out.end, b + 24, 8);
    std::memcpy(&start, b + 32, 8);
    std::memcpy(&out.time, b + 40, 8);
    if (start != DATA_START || out.end < DATA_START || out.end > file_size)
        return HeaderState::INVALID;
    return HeaderState::VALID;
}

// ── Stream adapters ─────────────────────────────────────────────────────
// Metadata::serialize/deserialize speak std::ostream/istream. These let them
// write straight into a section buffer and read straight out of one, without
// an ostringstream copy per record.
class StringSink : public std::streambuf {
    std::string& s_;
public:
    explicit StringSink(std::string& s) : s_(s) {}
protected:
    std::streamsize xsputn(const char* p, std::streamsize n) override {
        s_.append(p, static_cast<size_t>(n));
        return n;
    }
    int_type overflow(int_type c) override {
        if (c != traits_type::eof()) s_.push_back(static_cast<char>(c));
        return c;
    }
};

class MemSource : public std::streambuf {
public:
    MemSource(const char* p, size_t n) {
        char* b = const_cast<char*>(p);
        setg(b, b, b + n);
    }
};

} // namespace v10
} // namespace feather
