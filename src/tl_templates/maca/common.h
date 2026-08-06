// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights
// reserved.

#pragma once

#include "atomic.h"
#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
#include <cute/arch/mma.hpp>
#include <cute/underscore.hpp>
#include <mcr/mc_runtime.h>
#include <mctlass/fast_math.h>

#define MACART_INF_F __int_as_float(0x7f800000)
#define MACART_NEGINF_F __int_as_float(0xff800000)
#define MACART_NAN_F __int_as_float(0x7fffffff)
#define MACART_MIN_DENORM_F __int_as_float(0x00000001)
#define MACART_MAX_NORMAL_F __int_as_float(0x7f7fffff)
#define MACART_NEG_ZERO_F __int_as_float(0x80000000)
#define MACART_ZERO_F 0.0f
#define MACART_ONE_F 1.0f

/* double precision constants */
#define MACART_INF __hiloint2double(0x7ff00000, 0x00000000)
#define MACART_NAN __hiloint2double(0xfff80000, 0x00000000)

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__
#define TL_PATCH

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    mcError_t __err = (stmt);                                                  \
    if (__err != mcSuccess) {                                                  \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, mcGetErrorName(__err), mcGetErrorString(__err));      \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    mcError_t __err = mcGetLastError();                                        \
    if (__err != mcSuccess) {                                                  \
      snprintf(error_buf, ERROR_BUF_SIZE, "kernel_name: %s - %s",              \
               mcGetErrorName(__err), mcGetErrorString(__err));                \
      return -1;                                                               \
    }                                                                          \
  } while (0)

using half_t = __half;

using bfloat16_t = maca_bfloat16;

struct bfloat16x2 {
  bfloat16_t data[2];
};

struct bfloat16x4 {
  bfloat16_t data[4];
};

struct bfloat16x8 {
  bfloat16_t data[8];
};

struct bfloat16x16 {
  bfloat16_t data[16];
};

using int4_t = int4;

using float16_t = _Float16;
using float16x2 =
    __attribute__((__vector_size__(2 * sizeof(float16_t)))) float16_t;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using float16x8 =
    __attribute__((__vector_size__(8 * sizeof(float16_t)))) float16_t;
using float16x16 =
    __attribute__((__vector_size__(16 * sizeof(float16_t)))) float16_t;

using float32x2 = __attribute__((__vector_size__(2 * sizeof(float)))) float;

typedef
    __attribute__((__vector_size__(4 * sizeof(short)))) short bfloat16x4_vec;

using int32x2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float64x4 = __attribute__((__vector_size__(4 * sizeof(double)))) double;
using int8x4 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;

namespace platform {

/// Numeric limits
template <> struct numeric_limits<__half> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = true;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 10;

  /// Least positive value
  __device__ static __half min() {
    uint16_t val = 0x0001;
    return *reinterpret_cast<__half *>(&val);
  }

  /// Minimum finite value
  __device__ static __half lowest() {
    uint16_t val = 0xfbff;
    return *reinterpret_cast<__half *>(&val);
  }

  /// Maximum finite value
  __device__ static __half max() {
    uint16_t val = 0x7bff;
    return *reinterpret_cast<__half *>(&val);
  }

  /// Returns smallest finite value
  __device__ static __half epsilon() {
    uint16_t val = 0x1800;
    return *reinterpret_cast<__half *>(&val);
  }

  /// Returns maximum rounding error
  __device__ static __half round_error() { return __half(0.5f); }

  /// Returns positive infinity value
  __device__ static __half infinity() {
    uint16_t val = 0x7c00;
    return *reinterpret_cast<__half *>(&val);
  }

  /// Returns quiet NaN value
  __device__ static __half quiet_NaN() {
    uint16_t val = 0x7fff;
    return *reinterpret_cast<__half *>(&val);
  }

  /// Returns signaling NaN value
  __device__ static __half signaling_NaN() {
    uint16_t val = 0x0001;
    return *reinterpret_cast<__half *>(&val);
  }

  /// Returns smallest positive subnormal value
  __device__ static __half denorm_min() {
    uint16_t val = 0x0001;
    return *reinterpret_cast<__half *>(&val);
  }
};

template <> struct numeric_limits<maca_bfloat16> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 7;

  /// Least positive value
  __device__ static maca_bfloat16 min() {
    return maca_bfloat16(__maca_bfloat16_raw{0x01});
  }

  /// Minimum finite value
  __device__ static maca_bfloat16 lowest() {
    return maca_bfloat16(__maca_bfloat16_raw{0xff7f});
  }

  /// Maximum finite value
  __device__ static maca_bfloat16 max() {
    return maca_bfloat16(__maca_bfloat16_raw{0x7f7f});
  }

  /// Returns smallest finite value
  __device__ static maca_bfloat16 epsilon() {
    return maca_bfloat16(__maca_bfloat16_raw{0x1000});
  }

  /// Returns maximum rounding error
  __device__ static maca_bfloat16 round_error() { return maca_bfloat16(0.5f); }
  /// Returns positive infinity value
  __device__ static maca_bfloat16 infinity() {
    return maca_bfloat16(__maca_bfloat16_raw{0x7f80});
  }
  /// Returns quiet NaN value
  __device__ static maca_bfloat16 quiet_NaN() {
    return maca_bfloat16(__maca_bfloat16_raw{0x7fff});
  }
  /// Returns signaling NaN value
  __device__ static maca_bfloat16 signaling_NaN() {
    return maca_bfloat16(__maca_bfloat16_raw{0x7fff});
  }
  /// Returns smallest positive subnormal value
  __device__ static maca_bfloat16 denorm_min() {
    return maca_bfloat16(__maca_bfloat16_raw{0x1});
  }
};
} // namespace platform

// Pack four char values. Build the 32-bit pattern from unsigned bytes: a
// negative signed char would otherwise signed-extend and flood the other lanes
// through the OR.
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  const unsigned char b0 = static_cast<unsigned char>(x0);
  const unsigned char b1 = static_cast<unsigned char>(x1);
  const unsigned char b2 = static_cast<unsigned char>(x2);
  const unsigned char b3 = static_cast<unsigned char>(x3);
  return static_cast<int>((b3 << 24) | (b2 << 16) | (b1 << 8) | b0);
}

// Pack eight char values.
TL_DEVICE int2 make_int2(signed char x0, signed char x1, signed char x2,
                         signed char x3, signed char y0, signed char y1,
                         signed char y2, signed char y3) {
  int2 result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  return result;
}

// Pack sixteen char values.
TL_DEVICE int4_t make_int4(signed char x0, signed char x1, signed char x2,
                           signed char x3, signed char y0, signed char y1,
                           signed char y2, signed char y3, signed char z0,
                           signed char z1, signed char z2, signed char z3,
                           signed char w0, signed char w1, signed char w2,
                           signed char w3) {
  int4_t result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  result.z = make_int(z0, z1, z2, z3);
  result.w = make_int(w0, w1, w2, w3);
  return result;
}

TL_DEVICE int4_t make_int4(short x0, short x1, short y0, short y1, short z0,
                           short z1, short w0, short w1) {
  int4_t result;
  *((short2 *)&result.x) = make_short2(x0, x1);
  *((short2 *)&result.y) = make_short2(y0, y1);
  *((short2 *)&result.z) = make_short2(z0, z1);
  *((short2 *)&result.w) = make_short2(w0, w1);
  return result;
}

// Pack four char values.
TL_DEVICE unsigned int make_uint(unsigned char x0, unsigned char x1,
                                 unsigned char x2, unsigned char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
}

template <typename T> TL_DEVICE unsigned int pack_b8x4(T x0, T x1, T x2, T x3) {
  return make_uint(*reinterpret_cast<unsigned char *>(&x0),
                   *reinterpret_cast<unsigned char *>(&x1),
                   *reinterpret_cast<unsigned char *>(&x2),
                   *reinterpret_cast<unsigned char *>(&x3));
}

// Pack eight char values.
TL_DEVICE uint2 make_uint2(unsigned char x0, unsigned char x1, unsigned char x2,
                           unsigned char x3, unsigned char y0, unsigned char y1,
                           unsigned char y2, unsigned char y3) {
  uint2 result;
  result.x = make_uint(x0, x1, x2, x3);
  result.y = make_uint(y0, y1, y2, y3);
  return result;
}

// Pack sixteen char values.
TL_DEVICE uint4 make_uint4(unsigned char x0, unsigned char x1, unsigned char x2,
                           unsigned char x3, unsigned char y0, unsigned char y1,
                           unsigned char y2, unsigned char y3, unsigned char z0,
                           unsigned char z1, unsigned char z2, unsigned char z3,
                           unsigned char w0, unsigned char w1, unsigned char w2,
                           unsigned char w3) {
  uint4 result;
  result.x = make_uint(x0, x1, x2, x3);
  result.y = make_uint(y0, y1, y2, y3);
  result.z = make_uint(z0, z1, z2, z3);
  result.w = make_uint(w0, w1, w2, w3);
  return result;
}

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_maca_bfloat162(const bfloat16_t x,
                                         const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

namespace tl {
TL_DEVICE float fast_rcp(float x) { return __fdividef(1.0f, x); }
} // namespace tl

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val, int memory_order = 0) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  (void)memory_order;
  atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
}

template <typename T> TL_DEVICE void AtomicAdd(_Float16 *address, T val) {
  atomicAdd(reinterpret_cast<__half *>(address), static_cast<__half>(val));
}

TL_DEVICE half_t max(const half_t a, const half_t b) {
  return half_t(__hmax(a, b));
}

TL_DEVICE half_t min(const half_t a, const half_t b) {
  return half_t(__hmin(a, b));
}

TL_DEVICE bfloat16_t max(const bfloat16_t a, const bfloat16_t b) {
  return __hmax(a, b);
}

TL_DEVICE bfloat16_t min(const bfloat16_t a, const bfloat16_t b) {
  return __hmin(a, b);
}

// DP4A
TL_DEVICE int __dp4a(int srcA, int srcB, int c) {
  int4 v_srca{(signed char)(srcA & 0xff), (signed char)((srcA >> 8) & 0xff),
              (signed char)((srcA >> 16) & 0xff),
              (signed char)((srcA >> 24) & 0xff)};
  int4 v_srcb{(signed char)(srcB & 0xff), (signed char)((srcB >> 8) & 0xff),
              (signed char)((srcB >> 16) & 0xff),
              (signed char)((srcB >> 24) & 0xff)};

  return v_srca.x * v_srcb.x + v_srca.y * v_srcb.y + v_srca.z * v_srcb.z +
         v_srca.w * v_srcb.w + c;
}

// Pack eight int values.
TL_DEVICE longlong4 make_longlong4(int x0, int x1, int y0, int y1, int z0,
                                   int z1, int w0, int w1) {
  longlong4 result;
  *((int2 *)&result.x) = make_int2(x0, x1);
  *((int2 *)&result.y) = make_int2(y0, y1);
  *((int2 *)&result.z) = make_int2(z0, z1);
  *((int2 *)&result.w) = make_int2(w0, w1);
  return result;
}

// Pack thirty-two char values.
TL_DEVICE longlong4
make_longlong4(signed char x0, signed char x1, signed char x2, signed char x3,
               signed char x4, signed char x5, signed char x6, signed char x7,
               signed char y0, signed char y1, signed char y2, signed char y3,
               signed char y4, signed char y5, signed char y6, signed char y7,
               signed char z0, signed char z1, signed char z2, signed char z3,
               signed char z4, signed char z5, signed char z6, signed char z7,
               signed char w0, signed char w1, signed char w2, signed char w3,
               signed char w4, signed char w5, signed char w6, signed char w7) {
  longlong4 result;
  *((int2 *)&result.x) = make_int2(x0, x1, x2, x3, x4, x5, x6, x7);
  *((int2 *)&result.y) = make_int2(y0, y1, y2, y3, y4, y5, y6, y7);
  *((int2 *)&result.z) = make_int2(z0, z1, z2, z3, z4, z5, z6, z7);
  *((int2 *)&result.w) = make_int2(w0, w1, w2, w3, w4, w5, w6, w7);
  return result;
}

// Pack thirty-two unsigned char values.
TL_DEVICE ulonglong4 make_ulonglong4(
    unsigned char x0, unsigned char x1, unsigned char x2, unsigned char x3,
    unsigned char x4, unsigned char x5, unsigned char x6, unsigned char x7,
    unsigned char y0, unsigned char y1, unsigned char y2, unsigned char y3,
    unsigned char y4, unsigned char y5, unsigned char y6, unsigned char y7,
    unsigned char z0, unsigned char z1, unsigned char z2, unsigned char z3,
    unsigned char z4, unsigned char z5, unsigned char z6, unsigned char z7,
    unsigned char w0, unsigned char w1, unsigned char w2, unsigned char w3,
    unsigned char w4, unsigned char w5, unsigned char w6, unsigned char w7) {
  ulonglong4 result;
  *((uint2 *)&result.x) = make_uint2(x0, x1, x2, x3, x4, x5, x6, x7);
  *((uint2 *)&result.y) = make_uint2(y0, y1, y2, y3, y4, y5, y6, y7);
  *((uint2 *)&result.z) = make_uint2(z0, z1, z2, z3, z4, z5, z6, z7);
  *((uint2 *)&result.w) = make_uint2(w0, w1, w2, w3, w4, w5, w6, w7);
  return result;
}

// Helper to cast SMEM pointer to unsigned
TL_DEVICE uint32_t smem_ptr_to_uint(void const *const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <typename InDatatype, typename OutDatatype>
TL_DEVICE void DP4A(InDatatype *a, InDatatype *b, OutDatatype *c) {
  const int a_int = *((int *)a);
  const int b_int = *((int *)b);
  const int c_int = *((int *)c);
  *c = __dp4a(a_int, b_int, c_int);
}

namespace tl {
// Any
template <typename T> TL_DEVICE bool Any(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

// All
template <typename T> TL_DEVICE bool All(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

// Pow of int
template <int y = 1, typename T> TL_DEVICE T pow_of_int(T x) {
  T result = x;
  for (int i = 1; i < y; i++) {
    result *= x;
  }
  return result;
}

template <int barrier_id = 0, int thread_count = 0>
TL_DEVICE void __sync_thread_partial() {
  // INFO: all threads will sync in a warp in maca, does not need partial
  // version
}

} // namespace tl

//
// Type-safe warp shuffle helpers for 16-bit float types
// These wrappers avoid relying on implicit conversions that may be disallowed
// (e.g., converting float -> mctlass::bfloat16_t) by explicitly promoting to
// float for the shuffle and then down-converting.
//
namespace tl {

template <int LaneMask> TL_DEVICE uint32_t shfl_xor_u32_imm(uint32_t v) {
  static_assert(LaneMask == 1 || LaneMask == 2 || LaneMask == 4 ||
                    LaneMask == 8,
                "unsupported row-local XOR lane mask");
  if constexpr (LaneMask == 1) {
    // Quad permutation: [0, 1, 2, 3] -> [1, 0, 3, 2].
    return static_cast<uint32_t>(__builtin_mxc_update_shfl(
        static_cast<int>(v), static_cast<int>(v), 0x0b1, 0xf, 0xf, false));
  } else if constexpr (LaneMask == 2) {
    // Quad permutation: [0, 1, 2, 3] -> [2, 3, 0, 1].
    return static_cast<uint32_t>(__builtin_mxc_update_shfl(
        static_cast<int>(v), static_cast<int>(v), 0x04e, 0xf, 0xf, false));
  } else if constexpr (LaneMask == 4) {
    // XOR 4 within a 16-lane row is two masked row shifts:
    // banks 0/2 read from +4, banks 1/3 read from -4.
    uint32_t out = v;
    out = static_cast<uint32_t>(__builtin_mxc_update_shfl(
        static_cast<int>(out), static_cast<int>(v), 0x104, 0xf, 0x5, false));
    out = static_cast<uint32_t>(__builtin_mxc_update_shfl(
        static_cast<int>(out), static_cast<int>(v), 0x114, 0xf, 0xa, false));
    return out;
  } else if constexpr (LaneMask == 8) {
    // Row rotate right by 8 is equivalent to XOR 8 within a 16-lane row.
    return static_cast<uint32_t>(__builtin_mxc_update_shfl(
        static_cast<int>(v), static_cast<int>(v), 0x128, 0xf, 0xf, false));
  } else {
    return v;
  }
}

template <int LaneMask, typename T> TL_DEVICE T shfl_xor_imm(T val) {
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    uint32_t bits = 0;
    __builtin_memcpy(&bits, &val, sizeof(T));
    bits = shfl_xor_u32_imm<LaneMask>(bits);
    T out;
    __builtin_memcpy(&out, &bits, sizeof(T));
    return out;
  } else if constexpr (sizeof(T) == 2 * sizeof(uint32_t)) {
    uint32_t bits[2];
    __builtin_memcpy(bits, &val, sizeof(T));
    bits[0] = shfl_xor_u32_imm<LaneMask>(bits[0]);
    bits[1] = shfl_xor_u32_imm<LaneMask>(bits[1]);
    T out;
    __builtin_memcpy(&out, bits, sizeof(T));
    return out;
  } else {
    return val;
  }
}

template <typename T>
TL_DEVICE T shfl_xor_sync_fallback(uint64_t mask, T val, int laneMask) {
  return __shfl_xor_sync(mask, val, laneMask);
}

template <>
TL_DEVICE half_t shfl_xor_sync_fallback(uint64_t mask, half_t val,
                                        int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor_sync(mask, f, laneMask);
  return half_t(r);
}

template <>
TL_DEVICE uint1 shfl_xor_sync_fallback(uint64_t mask, uint1 val, int laneMask) {
  unsigned long raw = static_cast<unsigned long>(val.x);
  unsigned long result = __shfl_xor_sync(mask, raw, laneMask);
  return uint1{static_cast<unsigned int>(result)};
}

template <typename T>
TL_DEVICE T shfl_xor_sync(uint64_t mask, T val, int laneMask) {
  if constexpr (sizeof(T) <= 2 * sizeof(uint32_t)) {
    if (mask == uint64_t(-1)) {
      switch (laneMask) {
      case 1:
        return shfl_xor_imm<1>(val);
      case 2:
        return shfl_xor_imm<2>(val);
      case 4:
        return shfl_xor_imm<4>(val);
      case 8:
        return shfl_xor_imm<8>(val);
      default:
        break;
      }
    }
  }
  return shfl_xor_sync_fallback(mask, val, laneMask);
}

} // namespace tl

namespace tl {

using uint1 = unsigned int;

template <typename T> TL_DEVICE T from_uint1(::uint1 v) {
  T r;
  __builtin_memcpy(&r, &v, sizeof(T));
  return r;
}

template <typename T> TL_DEVICE ::uint1 to_uint1(T v) {
  ::uint1 r;
  __builtin_memcpy(&r, &v, sizeof(::uint1));
  return r;
}

// =========================================================================
// Packed x2 element-wise math helpers
//
// Each operation (add2, sub2, mul2, fma2, max2, min2, abs2) is provided for:
//   1. float2          (FP32x2, scalar lanes -- no MACA f32 packed SDK API)
//   2. maca_bfloat162  (BF16x2, MACA SDK __h*2 intrinsics)
//   3. half2           (FP16x2, MACA SDK __h*2 intrinsics)
// =========================================================================
// Pack two half_t into a uint1
TL_DEVICE uint1 pack_half2(half_t a, half_t b) {
  return __pack_half2(static_cast<__half>(a), static_cast<__half>(b));
}

// Helper to extract half_t from float16x2 (which uses _Float16)
TL_DEVICE half_t extract_half_from_float16x2(float16x2 v, int lane) {
  // float16x2 is a vector of _Float16, convert to half_t via float
  return half_t(static_cast<float>(v[lane]));
}

// Helper to extract bfloat16_t from bfloat16x2
TL_DEVICE bfloat16_t extract_bfloat16_from_bfloat16x2(bfloat16x2 v, int lane) {
  return v.data[lane];
}

// --- add2 ----------------------------------------------------------------

TL_DEVICE float2 add2(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

TL_DEVICE maca_bfloat162 add2(maca_bfloat162 a, maca_bfloat162 b) {
  return __hadd2(a, b);
}

TL_DEVICE half2 add2(half2 a, half2 b) { return __hadd2(a, b); }

// --- sub2 ----------------------------------------------------------------

TL_DEVICE float2 sub2(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

TL_DEVICE maca_bfloat162 sub2(maca_bfloat162 a, maca_bfloat162 b) {
  return __hsub2(a, b);
}

TL_DEVICE half2 sub2(half2 a, half2 b) { return __hsub2(a, b); }

// --- mul2 ----------------------------------------------------------------

TL_DEVICE float2 mul2(float2 a, float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

TL_DEVICE maca_bfloat162 mul2(maca_bfloat162 a, maca_bfloat162 b) {
  return __hmul2(a, b);
}

TL_DEVICE half2 mul2(half2 a, half2 b) { return __hmul2(a, b); }

// --- fma2 ----------------------------------------------------------------

TL_DEVICE float2 fma2(float2 a, float2 b, float2 c) {
  return make_float2(a.x * b.x + c.x, a.y * b.y + c.y);
}

TL_DEVICE maca_bfloat162 fma2(maca_bfloat162 a, maca_bfloat162 b,
                              maca_bfloat162 c) {
  return __hfma2(a, b, c);
}

TL_DEVICE half2 fma2(half2 a, half2 b, half2 c) { return __hfma2(a, b, c); }

// --- max2 ----------------------------------------------------------------

TL_DEVICE float2 max2(float2 a, float2 b) {
  return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

TL_DEVICE maca_bfloat162 max2(maca_bfloat162 a, maca_bfloat162 b) {
  return __hmax2(a, b);
}

TL_DEVICE half2 max2(half2 a, half2 b) { return __hmax2(a, b); }

// --- min2 ----------------------------------------------------------------

TL_DEVICE float2 min2(float2 a, float2 b) {
  return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

TL_DEVICE maca_bfloat162 min2(maca_bfloat162 a, maca_bfloat162 b) {
  return __hmin2(a, b);
}

TL_DEVICE half2 min2(half2 a, half2 b) { return __hmin2(a, b); }

// --- max2_nan ------------------------------------------------------------

TL_DEVICE maca_bfloat162 max2_nan(maca_bfloat162 a, maca_bfloat162 b) {
  return __hmax2_nan(a, b);
}

TL_DEVICE half2 max2_nan(half2 a, half2 b) { return __hmax2_nan(a, b); }

// --- min2_nan ------------------------------------------------------------

TL_DEVICE maca_bfloat162 min2_nan(maca_bfloat162 a, maca_bfloat162 b) {
  return __hmin2_nan(a, b);
}

TL_DEVICE half2 min2_nan(half2 a, half2 b) { return __hmin2_nan(a, b); }

// --- abs2 ----------------------------------------------------------------

TL_DEVICE float2 abs2(float2 a) { return make_float2(fabsf(a.x), fabsf(a.y)); }

TL_DEVICE maca_bfloat162 abs2(maca_bfloat162 a) { return __habs2(a); }

TL_DEVICE half2 abs2(half2 a) { return __habs2(a); }

} // namespace tl

// MACA vector types are aggregate structs, so they do not accept the
// function-style construction emitted by the generic C code generator (for
// example, `longlong4(a, b, c, d)`).  The MACA SDK already provides the
// corresponding make_* helpers.  Function-like macros only rewrite calls;
// uses such as `longlong4 value` remain type declarations.
#ifndef TILELANG_MACA_VECTOR_CONSTRUCTOR_COMPAT
#define TILELANG_MACA_VECTOR_CONSTRUCTOR_COMPAT
#define longlong2(...) make_longlong2(__VA_ARGS__)
#define longlong3(...) make_longlong3(__VA_ARGS__)
#define longlong4(...) make_longlong4(__VA_ARGS__)
#define ulonglong2(...) make_ulonglong2(__VA_ARGS__)
#define ulonglong3(...) make_ulonglong3(__VA_ARGS__)
#define ulonglong4(...) make_ulonglong4(__VA_ARGS__)
#endif
