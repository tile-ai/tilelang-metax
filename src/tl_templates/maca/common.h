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

#define __float2half_rn(x) half(x)

#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16

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
} // namespace platform

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

typedef
    __attribute__((__vector_size__(4 * sizeof(short)))) short bfloat16x4_vec;

using int32x2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float64x4 = __attribute__((__vector_size__(4 * sizeof(double)))) double;
using int8x4 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;

// Pack four char values.
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
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

// Pack eight char values.
TL_DEVICE uint2 make_uint2(unsigned char x0, unsigned char x1, unsigned char x2,
                           unsigned char x3, unsigned char y0, unsigned char y1,
                           unsigned char y2, unsigned char y3) {
  uint2 result;
  result.x = make_uint(x0, x1, x2, x3);
  result.y = make_uint(y0, y1, y2, y3);
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
  return mctlass::fast_max(a, b);
}

TL_DEVICE half_t min(const half_t a, const half_t b) {
  return mctlass::fast_min(a, b);
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

// Generic passthroughs
template <typename T>
TL_DEVICE T shfl_xor_sync(uint64_t mask, T val, int laneMask) {
  return __shfl_xor_sync(mask, val, laneMask);
}

// Specializations for mctlass::half_t
template <>
TL_DEVICE half_t shfl_xor_sync(uint64_t mask, half_t val, int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor_sync(mask, f, laneMask);
  return half_t(r);
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

//  --- add2
//  --------------------------------------------------------------------------

TL_DEVICE float2 add2(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
TL_DEVICE float16x2 add2(float16x2 a, float16x2 b) { return a + b; }

TL_DEVICE bfloat16x2 add2(bfloat16x2 a, bfloat16x2 b) {
  bfloat16x2 out;
  out.data[0] = a.data[0] + b.data[0];
  out.data[1] = a.data[1] + b.data[1];
  return out;
}

//  --- sub2
//  --------------------------------------------------------------------------

TL_DEVICE float2 sub2(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

TL_DEVICE float16x2 sub2(float16x2 a, float16x2 b) { return a - b; }

TL_DEVICE bfloat16x2 sub2(bfloat16x2 a, bfloat16x2 b) {
  bfloat16x2 out;
  out.data[0] = a.data[0] - b.data[0];
  out.data[1] = a.data[1] - b.data[1];
  return out;
}

//  --- mul2
//  --------------------------------------------------------------------------

TL_DEVICE float2 mul2(float2 a, float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

TL_DEVICE float16x2 mul2(float16x2 a, float16x2 b) { return a * b; }

TL_DEVICE bfloat16x2 mul2(bfloat16x2 a, bfloat16x2 b) {
  bfloat16x2 out;
  out.data[0] = a.data[0] * b.data[0];
  out.data[1] = a.data[1] * b.data[1];
  return out;
}

//  --- fma2
//  --------------------------------------------------------------------------

TL_DEVICE float2 fma2(float2 a, float2 b, float2 c) {
  return make_float2(a.x * b.x + c.x, a.y * b.y + c.y);
}
TL_DEVICE float16x2 fma2(float16x2 a, float16x2 b, float16x2 c) {
  return a * b + c;
}
TL_DEVICE bfloat16x2 fma2(bfloat16x2 a, bfloat16x2 b, bfloat16x2 c) {
  bfloat16x2 out;
  out.data[0] = a.data[0] * b.data[0] + c.data[0];
  out.data[1] = a.data[1] * b.data[1] + c.data[1];
  return out;
}

//  --- max2
//  --------------------------------------------------------------------------

TL_DEVICE float2 max2(float2 a, float2 b) {
  return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

TL_DEVICE float16x2 max2(float16x2 a, float16x2 b) {
  float16x2 out;
  out[0] = a[0] > b[0] ? a[0] : b[0];
  out[1] = a[1] > b[1] ? a[1] : b[1];
  return out;
}

TL_DEVICE bfloat16x2 max2(bfloat16x2 a, bfloat16x2 b) {
  bfloat16x2 out;
  out.data[0] = (float(a.data[0]) > float(b.data[0])) ? a.data[0] : b.data[0];
  out.data[1] = (float(a.data[1]) > float(b.data[1])) ? a.data[1] : b.data[1];
  return out;
}

//  --- min2
//  --------------------------------------------------------------------------

TL_DEVICE float2 min2(float2 a, float2 b) {
  return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

TL_DEVICE float16x2 min2(float16x2 a, float16x2 b) {
  float16x2 out;
  out[0] = (float(a[0]) < float(b[0])) ? a[0] : b[0];
  out[1] = (float(a[1]) < float(b[1])) ? a[1] : b[1];
  return out;
}

TL_DEVICE bfloat16x2 min2(bfloat16x2 a, bfloat16x2 b) {
  bfloat16x2 out;
  out.data[0] = (float(a.data[0]) < float(b.data[0])) ? a.data[0] : b.data[0];
  out.data[1] = (float(a.data[1]) < float(b.data[1])) ? a.data[1] : b.data[1];
  return out;
}

//  --- abs2
//  --------------------------------------------------------------------------

TL_DEVICE float2 abs2(float2 a) { return make_float2(fabsf(a.x), fabsf(a.y)); }

TL_DEVICE float16x2 abs2(float16x2 a) {
  float16x2 out;
  out[0] = (a[0] < _Float16(0.0f)) ? -a[0] : a[0];
  out[1] = (a[1] < _Float16(0.0f)) ? -a[1] : a[1];
  return out;
}

TL_DEVICE bfloat16x2 abs2(bfloat16x2 a) {
  bfloat16x2 out;
  out.data[0] = (float(a.data[0]) < 0.0f) ? -a.data[0] : a.data[0];
  out.data[1] = (float(a.data[1]) < 0.0f) ? -a.data[1] : a.data[1];
  return out;
}

} // namespace tl
