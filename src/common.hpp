#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <cmath>
#include <bit>
#include <limits>
#include <type_traits>

#if defined(_MSC_VER) && !defined(__clang__)
inline int __builtin_clz(unsigned int x) { return std::countl_zero(x); }
inline int __builtin_ctz(unsigned int x) { return std::countr_zero(x); }
inline int __builtin_popcount(unsigned int x) { return std::popcount(x); }

template <typename T>
inline bool __builtin_add_overflow(T a, T b, T* res) {
    if constexpr (std::is_signed_v<T>) {
        using UT = std::make_unsigned_t<T>;
        *res = static_cast<T>(static_cast<UT>(a) + static_cast<UT>(b));
        if (b > 0 && a > std::numeric_limits<T>::max() - b) return true;
        if (b < 0 && a < std::numeric_limits<T>::min() - b) return true;
        return false;
    } else {
        *res = a + b;
        return *res < a;
    }
}

template <typename T>
inline bool __builtin_sub_overflow(T a, T b, T* res) {
    if constexpr (std::is_signed_v<T>) {
        using UT = std::make_unsigned_t<T>;
        *res = static_cast<T>(static_cast<UT>(a) - static_cast<UT>(b));
        if (b < 0 && a > std::numeric_limits<T>::max() + b) return true;
        if (b > 0 && a < std::numeric_limits<T>::min() + b) return true;
        return false;
    } else {
        *res = a - b;
        return a < b;
    }
}
#endif

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using s8  = std::int8_t;
using s16 = std::int16_t;
using s32 = std::int32_t;
using s64 = std::int64_t;

using f32 = float;
using f64 = double;

// Endianness swap helpers
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
#include <sys/endian.h>
#endif

#if defined(_MSC_VER)
#include <cstdlib>
inline u16 bswap16(u16 v) { return _byteswap_ushort(v); }
inline u32 bswap32(u32 v) { return _byteswap_ulong(v); }
inline u64 bswap64(u64 v) { return _byteswap_uint64(v); }
#elif defined(__OpenBSD__)
#include <sys/endian.h>
inline u16 bswap16(u16 v) { return swap16(v); }
inline u32 bswap32(u32 v) { return swap32(v); }
inline u64 bswap64(u64 v) { return swap64(v); }
#elif defined(bswap16)
// BSDs <sys/endian.h> defines bswap16/32/64 as macros for the same builtins.
#else
inline u16 bswap16(u16 v) { return __builtin_bswap16(v); }
inline u32 bswap32(u32 v) { return __builtin_bswap32(v); }
inline u64 bswap64(u64 v) { return __builtin_bswap64(v); }
#endif

// 64x64 -> 128-bit multiplication helpers (portable across 32-bit and 64-bit hosts)
inline void multu64_128(u64 a, u64 b, u64& hi, u64& lo) {
#if defined(__SIZEOF_INT128__)
    unsigned __int128 res = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    lo = static_cast<u64>(res);
    hi = static_cast<u64>(res >> 64);
#else
    const u64 a_lo = static_cast<u32>(a), a_hi = a >> 32;
    const u64 b_lo = static_cast<u32>(b), b_hi = b >> 32;
    const u64 p0 = a_lo * b_lo;
    const u64 p1 = a_lo * b_hi;
    const u64 p2 = a_hi * b_lo;
    const u64 p3 = a_hi * b_hi;
    const u64 m = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
    lo = (p0 & 0xFFFFFFFFULL) | (m << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (m >> 32);
#endif
}

inline void mults64_128(s64 a, s64 b, u64& hi, u64& lo) {
#if defined(__SIZEOF_INT128__)
    __int128 res = static_cast<__int128>(a) * static_cast<__int128>(b);
    lo = static_cast<u64>(res);
    hi = static_cast<u64>(res >> 64);
#else
    multu64_128(static_cast<u64>(a), static_cast<u64>(b), hi, lo);
    if (a < 0) hi -= static_cast<u64>(b);
    if (b < 0) hi -= static_cast<u64>(a);
#endif
}

// System constants
constexpr u64 CPU_CLOCK_RATE = 93750000ULL; // 93.75 MHz
constexpr u64 CYCLES_PER_FRAME = CPU_CLOCK_RATE / 60; // 1,562,500 cycles per frame at 60Hz
constexpr u64 AI_VIDEO_CLOCK_NTSC = 48681812ULL; // Video clock used to derive the AI DAC sample rate
constexpr u32 RDRAM_SIZE = 8 * 1024 * 1024; // 8 MB RDRAM (Expansion Pak)
constexpr u32 DMEM_SIZE = 4096;
constexpr u32 IMEM_SIZE = 4096;
constexpr u32 PIF_ROM_SIZE = 2048;
constexpr u32 PIF_RAM_SIZE = 64;

// Sign extension helpers
inline s64 sign_extend_32_64(s32 val) {
    return static_cast<s64>(val);
}
inline s64 sign_extend_16_64(s16 val) {
    return static_cast<s64>(val);
}
inline s64 sign_extend_8_64(s8 val) {
    return static_cast<s64>(val);
}
