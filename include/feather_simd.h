#pragma once
// Runtime-dispatched distance kernels.
//
// Why this exists: the upstream hnswlib kernels only get AVX when the WHOLE
// build is compiled with -mavx, which also lets the compiler emit AVX anywhere
// else and makes the binary crash on older CPUs. Published wheels were
// therefore built SSE-only, and pip users never got AVX. Here each wide kernel
// is compiled for its own ISA with a per-function target attribute, and the
// best one is chosen at runtime from CPUID. The same binary runs everywhere
// and uses AVX2+FMA / AVX-512 where the CPU has it. arm64 gets NEON (always
// present there); the scalar loop was not auto-vectorised, because a float
// reduction can't be reordered without -ffast-math.
//
// FEATHER_SIMD_RUNTIME=scalar|sse|avx2|avx512 caps the level (A/B benchmarks,
// tests). It can only lower the level, never enable an ISA the CPU lacks.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if !defined(NO_MANUAL_VECTORIZATION) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#  define FEATHER_SIMD_X86 1
#  include <immintrin.h>
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#    define FEATHER_TARGET(t)          /* MSVC compiles intrinsics without flags */
#  else
#    include <cpuid.h>
#    define FEATHER_TARGET(t) __attribute__((target(t)))
#  endif
#  if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    define FEATHER_HAVE_SSE2 1
#  endif
#elif !defined(NO_MANUAL_VECTORIZATION) && (defined(__aarch64__) || defined(_M_ARM64))
#  define FEATHER_SIMD_NEON 1
#  include <arm_neon.h>
#endif

namespace feather_simd {

enum Level : int { SCALAR = 0, SSE = 1, AVX2 = 2, AVX512 = 3, NEON = 10 };

// ── CPU detection ──────────────────────────────────────────────────────────
#if defined(FEATHER_SIMD_X86)
inline void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
#  if defined(_MSC_VER) && !defined(__clang__)
    int t[4];
    __cpuidex(t, static_cast<int>(leaf), static_cast<int>(sub));
    for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(t[i]);
#  else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#  endif
}
inline unsigned long long xgetbv0() {
#  if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#  else
    unsigned eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#  endif
}
inline int detect_x86() {
    unsigned r[4];
    cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    if (max_leaf < 1) return SCALAR;
    cpuid(1, 0, r);
    const bool sse2    = (r[3] >> 26) & 1;
    const bool fma     = (r[2] >> 12) & 1;
    const bool osxsave = (r[2] >> 27) & 1;
    const bool avx     = (r[2] >> 28) & 1;
    int level = SCALAR;
#  if defined(FEATHER_HAVE_SSE2)
    if (sse2) level = SSE;
#  else
    (void)sse2;
#  endif
    if (!(osxsave && avx) || max_leaf < 7) return level;
    const unsigned long long xcr0 = xgetbv0();
    const bool os_ymm = (xcr0 & 0x6) == 0x6;              // XMM + YMM state
    const bool os_zmm = (xcr0 & 0xe6) == 0xe6;            // + opmask, ZMM
    cpuid(7, 0, r);
    const bool avx2    = (r[1] >> 5) & 1;
    const bool avx512f = (r[1] >> 16) & 1;
    if (os_ymm && avx2 && fma) level = AVX2;
    if (level == AVX2 && os_zmm && avx512f) level = AVX512;
    return level;
}
#endif

inline int detected_level() {
#if defined(FEATHER_SIMD_X86)
    return detect_x86();
#elif defined(FEATHER_SIMD_NEON)
    return NEON;
#else
    return SCALAR;
#endif
}

// Detected level, optionally capped by FEATHER_SIMD_RUNTIME. Read once.
inline int level() {
    static const int lvl = [] {
        int l = detected_level();
        if (const char* e = std::getenv("FEATHER_SIMD_RUNTIME")) {
            int cap = l;
            if      (!std::strcmp(e, "scalar")) cap = SCALAR;
            else if (!std::strcmp(e, "sse"))    cap = SSE;
            else if (!std::strcmp(e, "avx2"))   cap = AVX2;
            else if (!std::strcmp(e, "avx512")) cap = AVX512;
            if (l == NEON) l = (cap == SCALAR) ? SCALAR : NEON;
            else if (cap < l) l = cap;
        }
        return l;
    }();
    return lvl;
}

inline const char* level_name(int l) {
    switch (l) {
        case SSE:    return "sse2";
        case AVX2:   return "avx2+fma";
        case AVX512: return "avx512f";
        case NEON:   return "neon";
        default:     return "scalar";
    }
}

// ── float32 squared L2 ─────────────────────────────────────────────────────
inline float l2_scalar(const float* a, const float* b, size_t n) {
    float r = 0.0f;
    for (size_t i = 0; i < n; ++i) { float d = a[i] - b[i]; r += d * d; }
    return r;
}

#if defined(FEATHER_SIMD_X86) && defined(FEATHER_HAVE_SSE2)
inline float l2_sse(const float* a, const float* b, size_t n) {
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128 d0 = _mm_sub_ps(_mm_loadu_ps(a + i),     _mm_loadu_ps(b + i));
        __m128 d1 = _mm_sub_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4));
        s0 = _mm_add_ps(s0, _mm_mul_ps(d0, d0));
        s1 = _mm_add_ps(s1, _mm_mul_ps(d1, d1));
    }
    for (; i + 4 <= n; i += 4) {
        __m128 d = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
        s0 = _mm_add_ps(s0, _mm_mul_ps(d, d));
    }
    __m128 s = _mm_add_ps(s0, s1);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s) + l2_scalar(a + i, b + i, n - i);
}
#endif

#if defined(FEATHER_SIMD_X86)
FEATHER_TARGET("avx2,fma")
inline float l2_avx2(const float* a, const float* b, size_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {       // two independent accumulators hide FMA latency
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        s0 = _mm256_fmadd_ps(d, d, s0);
    }
    __m256 s8 = _mm256_add_ps(s0, s1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    float r = _mm_cvtss_f32(s);
    for (; i < n; ++i) { float d = a[i] - b[i]; r += d * d; }
    return r;
}

FEATHER_TARGET("avx512f")
inline float l2_avx512(const float* a, const float* b, size_t n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        s0 = _mm512_fmadd_ps(d0, d0, s0);
        s1 = _mm512_fmadd_ps(d1, d1, s1);
    }
    for (; i + 16 <= n; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        s0 = _mm512_fmadd_ps(d, d, s0);
    }
    if (i < n) {                          // masked tail: no scalar loop
        const __mmask16 m = static_cast<__mmask16>((1u << (n - i)) - 1u);
        __m512 d = _mm512_sub_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i));
        s1 = _mm512_fmadd_ps(d, d, s1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
}
#endif

#if defined(FEATHER_SIMD_NEON)
inline float l2_neon(const float* a, const float* b, size_t n) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i),     vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s0 = vfmaq_f32(s0, d0, d0);
        s1 = vfmaq_f32(s1, d1, d1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        s0 = vfmaq_f32(s0, d, d);
    }
    return vaddvq_f32(vaddq_f32(s0, s1)) + l2_scalar(a + i, b + i, n - i);
}
#endif

typedef float (*L2Fn)(const float*, const float*, size_t);

inline L2Fn l2_fn_for(int l) {
    switch (l) {
#if defined(FEATHER_SIMD_X86)
        case AVX512: return l2_avx512;
        case AVX2:   return l2_avx2;
#  if defined(FEATHER_HAVE_SSE2)
        case SSE:    return l2_sse;
#  endif
#endif
#if defined(FEATHER_SIMD_NEON)
        case NEON:   return l2_neon;
#endif
        default:     return l2_scalar;
    }
}

inline L2Fn l2_fn() {
    static const L2Fn f = l2_fn_for(level());
    return f;
}

// ── int8 squared L2 (in-RAM int8 modality) ─────────────────────────────────
// |a_i - b_i| <= 254, so each 16-bit madd pair adds <= 129,032 to an int32
// lane. Blocks of 8192 dims keep every lane far below 2^31, then flush into
// an int64 total, so any dim up to the file format's 2^20 cap is exact.
inline int64_t i8_l2_scalar(const int8_t* a, const int8_t* b, size_t n) {
    int64_t acc = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        acc += static_cast<int64_t>(d) * d;
    }
    return acc;
}

#if defined(FEATHER_SIMD_X86)
// 32 bytes per step: |a-b| is computed EXACTLY as uint8 (bias both sides by
// 0x80, then OR the two saturating differences), zero-extended to 16 bits and
// squared with madd. Half the widening conversions of a 16-byte sign-extend
// loop: measured 1.45x faster at 1536-d, same results bit for bit.
FEATHER_TARGET("avx2")
inline int64_t i8_l2_avx2(const int8_t* a, const int8_t* b, size_t n) {
    const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));
    const __m256i zero = _mm256_setzero_si256();
    int64_t total = 0;
    size_t i = 0;
    while (i + 32 <= n) {
        const size_t block_end = (n - i > 8192) ? i + 8192 : n;
        __m256i acc0 = zero, acc1 = zero;
        for (; i + 32 <= block_end; i += 32) {
            __m256i va = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)), bias);
            __m256i vb = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)), bias);
            __m256i d  = _mm256_or_si256(_mm256_subs_epu8(va, vb), _mm256_subs_epu8(vb, va));
            __m256i lo = _mm256_unpacklo_epi8(d, zero);
            __m256i hi = _mm256_unpackhi_epi8(d, zero);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(lo, lo));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(hi, hi));
        }
        __m256i acc = _mm256_add_epi32(acc0, acc1);
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        total += static_cast<int64_t>(_mm_cvtsi128_si32(s));
        if (block_end == n) break;
    }
    return total + i8_l2_scalar(a + i, b + i, n - i);
}
#endif

#if defined(FEATHER_SIMD_NEON)
inline int64_t i8_l2_neon(const int8_t* a, const int8_t* b, size_t n) {
    int64_t total = 0;
    size_t i = 0;
    while (i + 16 <= n) {
        const size_t block_end = (n - i > 8192) ? i + 8192 : n;
        int32x4_t acc = vdupq_n_s32(0);
        for (; i + 16 <= block_end; i += 16) {
            int8x16_t va = vld1q_s8(a + i), vb = vld1q_s8(b + i);
            int16x8_t dlo = vsubl_s8(vget_low_s8(va),  vget_low_s8(vb));
            int16x8_t dhi = vsubl_s8(vget_high_s8(va), vget_high_s8(vb));
            acc = vmlal_s16(acc, vget_low_s16(dlo),  vget_low_s16(dlo));
            acc = vmlal_s16(acc, vget_high_s16(dlo), vget_high_s16(dlo));
            acc = vmlal_s16(acc, vget_low_s16(dhi),  vget_low_s16(dhi));
            acc = vmlal_s16(acc, vget_high_s16(dhi), vget_high_s16(dhi));
        }
        total += static_cast<int64_t>(vaddvq_s32(acc));
        if (block_end == n) break;
    }
    return total + i8_l2_scalar(a + i, b + i, n - i);
}
#endif

// ── asymmetric: float32 query vs int8 row (x scale) ────────────────────────
// Used by the pre-filtered exact scan on in-RAM int8 modalities: the query
// stays float (no quantization error on the query side), and each stored row
// is dequantized on the fly inside the kernel instead of copied out first.
inline float l2_f32_i8_scalar(const float* q, const int8_t* v, float scale, size_t n) {
    float r = 0.0f;
    for (size_t i = 0; i < n; ++i) { float d = q[i] - static_cast<float>(v[i]) * scale; r += d * d; }
    return r;
}

#if defined(FEATHER_SIMD_X86)
FEATHER_TARGET("avx2,fma")
inline float l2_f32_i8_avx2(const float* q, const int8_t* v, float scale, size_t n) {
    const __m256 sc = _mm256_set1_ps(scale);
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i b  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(v + i));
        __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)), sc);
        __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b, 8))), sc);
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(q + i), f0);
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(q + i + 8), f1);
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
    }
    __m256 s8 = _mm256_add_ps(s0, s1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s) + l2_f32_i8_scalar(q + i, v + i, scale, n - i);
}
#endif

#if defined(FEATHER_SIMD_NEON)
inline float l2_f32_i8_neon(const float* q, const int8_t* v, float scale, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int16x8_t w = vmovl_s8(vld1_s8(v + i));
        float32x4_t f0 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), scale);
        float32x4_t f1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))), scale);
        float32x4_t d0 = vsubq_f32(vld1q_f32(q + i), f0);
        float32x4_t d1 = vsubq_f32(vld1q_f32(q + i + 4), f1);
        acc = vfmaq_f32(acc, d0, d0);
        acc = vfmaq_f32(acc, d1, d1);
    }
    return vaddvq_f32(acc) + l2_f32_i8_scalar(q + i, v + i, scale, n - i);
}
#endif

typedef float (*F32I8L2Fn)(const float*, const int8_t*, float, size_t);

inline F32I8L2Fn f32_i8_l2_fn_for(int l) {
#if defined(FEATHER_SIMD_X86)
    if (l == AVX2 || l == AVX512) return l2_f32_i8_avx2;
#endif
#if defined(FEATHER_SIMD_NEON)
    if (l == NEON) return l2_f32_i8_neon;
#endif
    (void)l;
    return l2_f32_i8_scalar;
}

inline F32I8L2Fn f32_i8_l2_fn() {
    static const F32I8L2Fn f = f32_i8_l2_fn_for(level());
    return f;
}

typedef int64_t (*I8L2Fn)(const int8_t*, const int8_t*, size_t);

inline I8L2Fn i8_l2_fn_for(int l) {
#if defined(FEATHER_SIMD_X86)
    if (l >= AVX2 && l != NEON) return i8_l2_avx2;
#endif
#if defined(FEATHER_SIMD_NEON)
    if (l == NEON) return i8_l2_neon;
#endif
    (void)l;
    return i8_l2_scalar;
}

inline I8L2Fn i8_l2_fn() {
    static const I8L2Fn f = i8_l2_fn_for(level());
    return f;
}

}  // namespace feather_simd
