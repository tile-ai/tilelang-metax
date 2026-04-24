#pragma once

#include "common.h"

#ifndef __MACA_ALIGN__
#define __MACA_ALIGN__(N) __attribute__((aligned(N)))
#endif

namespace tl {
namespace detail {

TL_DEVICE float decode_fp4_e2m1(unsigned char bits) {
  bits &= 0x0F;
  const float sign = (bits & 0x8U) ? -1.0f : 1.0f;
  const unsigned int exponent = (bits >> 1U) & 0x3U;
  const unsigned int mantissa = bits & 0x1U;

  // e2m1: bias=1, no inf/nan encoding.
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return 0.0f;
    }
    // subnormal: sign * 0.5
    return sign * 0.5f;
  }

  // normal: sign * (1 + mantissa/2) * 2^(exponent-1)
  return sign * ldexpf(1.0f + 0.5f * static_cast<float>(mantissa),
                       static_cast<int>(exponent) - 1);
}

TL_DEVICE unsigned char encode_fp4_e2m1(float x) {
  if (x == 0.0f) {
    return signbit(x) ? 0x8U : 0x0U;
  }

  const bool neg = signbit(x);
  float ax = fabsf(x);

  // Saturate to maximum finite representable value: 6.0
  if (ax > 6.0f) {
    ax = 6.0f;
  }

  const float candidates[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const unsigned char codes[8] = {0x0U, 0x1U, 0x2U, 0x3U,
                                  0x4U, 0x5U, 0x6U, 0x7U};

  float best_diff = fabsf(ax - candidates[0]);
  unsigned int best_idx = 0U;
  for (unsigned int i = 1U; i < 8U; ++i) {
    float diff = fabsf(ax - candidates[i]);
    // Tie-break toward value with even LSB to emulate RN-even.
    if (diff < best_diff || (diff == best_diff && ((codes[i] & 1U) == 0U) &&
                             ((codes[best_idx] & 1U) != 0U))) {
      best_diff = diff;
      best_idx = i;
    }
  }

  unsigned char out = codes[best_idx];
  if (neg) {
    out |= 0x8U;
  }
  return out;
}

} // namespace detail
} // namespace tl

struct fp4_e2_t {
  unsigned char __x;

  TL_DEVICE fp4_e2_t() = default;
  TL_DEVICE explicit fp4_e2_t(unsigned char x) : __x(x & 0x0FU) {}
  TL_DEVICE explicit fp4_e2_t(float x) : __x(tl::detail::encode_fp4_e2m1(x)) {}
  TL_DEVICE explicit fp4_e2_t(double x)
      : __x(tl::detail::encode_fp4_e2m1(static_cast<float>(x))) {}
  TL_DEVICE explicit fp4_e2_t(half_t x)
      : __x(tl::detail::encode_fp4_e2m1(static_cast<float>(x))) {}
  TL_DEVICE explicit fp4_e2_t(bfloat16_t x)
      : __x(tl::detail::encode_fp4_e2m1(static_cast<float>(x))) {}

  TL_DEVICE operator float() const { return tl::detail::decode_fp4_e2m1(__x); }
  TL_DEVICE operator half_t() const { return half_t(float(*this)); }
};

class fp4_e2_2_t {
public:
  unsigned char __x;

  TL_DEVICE fp4_e2_2_t() = default;
  TL_DEVICE explicit fp4_e2_2_t(unsigned char data) : __x(data) {}

  TL_DEVICE fp4_e2_t x() const {
    return fp4_e2_t(static_cast<unsigned char>(__x & 0x0FU));
  }
  TL_DEVICE fp4_e2_t y() const {
    return fp4_e2_t(static_cast<unsigned char>((__x >> 4U) & 0x0FU));
  }

  TL_DEVICE void set_x(fp4_e2_t val) {
    __x = (__x & 0xF0U) | (val.__x & 0x0FU);
  }
  TL_DEVICE void set_y(fp4_e2_t val) {
    __x = (__x & 0x0FU) | ((val.__x & 0x0FU) << 4U);
  }

  template <typename T> TL_DEVICE void set_x(T val) {
    set_x(fp4_e2_t(static_cast<float>(val)));
  }
  template <typename T> TL_DEVICE void set_y(T val) {
    set_y(fp4_e2_t(static_cast<float>(val)));
  }
};

struct __MACA_ALIGN__(2) fp4_e2_4_t {
  fp4_e2_2_t x;
  fp4_e2_2_t y;
};

struct __MACA_ALIGN__(4) fp4_e2_8_t {
  fp4_e2_4_t x;
  fp4_e2_4_t y;
};

struct __MACA_ALIGN__(8) fp4_e2_16_t {
  fp4_e2_8_t x;
  fp4_e2_8_t y;
};

struct __MACA_ALIGN__(16) fp4_e2_32_t {
  fp4_e2_16_t x;
  fp4_e2_16_t y;

  TL_DEVICE fp4_e2_32_t &operator=(const ulonglong4 &rhs) {
    x.x = *(fp4_e2_8_t *)&rhs.x;
    x.y = *(fp4_e2_8_t *)&rhs.y;
    y.x = *(fp4_e2_8_t *)&rhs.z;
    y.y = *(fp4_e2_8_t *)&rhs.w;
    return *this;
  }
};

struct __MACA_ALIGN__(32) fp4_e2_64_t {
  fp4_e2_32_t x;
  fp4_e2_32_t y;
};

TL_DEVICE fp4_e2_2_t make_fp4_e2_2_t(fp4_e2_t x, fp4_e2_t y) {
  return fp4_e2_2_t((x.__x & 0x0FU) | ((y.__x & 0x0FU) << 4U));
}

TL_DEVICE fp4_e2_4_t make_fp4_e2_4_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                     fp4_e2_t x3) {
  fp4_e2_4_t result;
  result.x = make_fp4_e2_2_t(x0, x1);
  result.y = make_fp4_e2_2_t(x2, x3);
  return result;
}

TL_DEVICE fp4_e2_8_t make_fp4_e2_8_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                     fp4_e2_t x3, fp4_e2_t x4, fp4_e2_t x5,
                                     fp4_e2_t x6, fp4_e2_t x7) {
  fp4_e2_8_t result;
  result.x = make_fp4_e2_4_t(x0, x1, x2, x3);
  result.y = make_fp4_e2_4_t(x4, x5, x6, x7);
  return result;
}

TL_DEVICE fp4_e2_16_t make_fp4_e2_16_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                       fp4_e2_t x3, fp4_e2_t x4, fp4_e2_t x5,
                                       fp4_e2_t x6, fp4_e2_t x7, fp4_e2_t y0,
                                       fp4_e2_t y1, fp4_e2_t y2, fp4_e2_t y3,
                                       fp4_e2_t y4, fp4_e2_t y5, fp4_e2_t y6,
                                       fp4_e2_t y7) {
  fp4_e2_16_t result;
  result.x = make_fp4_e2_8_t(x0, x1, x2, x3, x4, x5, x6, x7);
  result.y = make_fp4_e2_8_t(y0, y1, y2, y3, y4, y5, y6, y7);
  return result;
}

TL_DEVICE fp4_e2_32_t make_fp4_e2_32_t(
    fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2, fp4_e2_t x3, fp4_e2_t x4,
    fp4_e2_t x5, fp4_e2_t x6, fp4_e2_t x7, fp4_e2_t x8, fp4_e2_t x9,
    fp4_e2_t x10, fp4_e2_t x11, fp4_e2_t x12, fp4_e2_t x13, fp4_e2_t x14,
    fp4_e2_t x15, fp4_e2_t y0, fp4_e2_t y1, fp4_e2_t y2, fp4_e2_t y3,
    fp4_e2_t y4, fp4_e2_t y5, fp4_e2_t y6, fp4_e2_t y7, fp4_e2_t y8,
    fp4_e2_t y9, fp4_e2_t y10, fp4_e2_t y11, fp4_e2_t y12, fp4_e2_t y13,
    fp4_e2_t y14, fp4_e2_t y15) {
  fp4_e2_32_t result;
  result.x = make_fp4_e2_16_t(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                              x12, x13, x14, x15);
  result.y = make_fp4_e2_16_t(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11,
                              y12, y13, y14, y15);
  return result;
}

// fp4_e2m1 -> half
TL_DEVICE half_t __tl_cvt_fp4_to_half(const unsigned char src) {
  return half_t(tl::detail::decode_fp4_e2m1(src));
}

// fp4_e2m1x2 -> half2
TL_DEVICE half2 __tl_cvt_fp4x2_to_half2(const unsigned char src) {
  half2 result;
  result.x = __tl_cvt_fp4_to_half(src & 0x0FU);
  result.y = __tl_cvt_fp4_to_half((src >> 4U) & 0x0FU);
  return result;
}

// half -> fp4_e2m1
TL_DEVICE unsigned char __tl_cvt_half_to_fp4(const half_t src) {
  return tl::detail::encode_fp4_e2m1(static_cast<float>(src));
}

// half2 -> fp4_e2m1x2
TL_DEVICE unsigned char __tl_cvt_half2_to_fp4x2(const half2 src) {
  const unsigned char lo = __tl_cvt_half_to_fp4(src.x) & 0x0FU;
  const unsigned char hi = __tl_cvt_half_to_fp4(src.y) & 0x0FU;
  return lo | (hi << 4U);
}

// fp4_e2m1 -> float
TL_DEVICE float __tl_cvt_fp4_to_float(const unsigned char src) {
  return tl::detail::decode_fp4_e2m1(src);
}

// fp4_e2m1x2 -> float2
TL_DEVICE float2 __tl_cvt_fp4x2_to_float2(const unsigned char src) {
  float2 result;
  result.x = __tl_cvt_fp4_to_float(src & 0x0FU);
  result.y = __tl_cvt_fp4_to_float((src >> 4U) & 0x0FU);
  return result;
}

// float -> fp4_e2m1
TL_DEVICE unsigned char __tl_cvt_float_to_fp4(const float src) {
  return tl::detail::encode_fp4_e2m1(src);
}

// float2 -> fp4_e2m1x2
TL_DEVICE unsigned char __tl_cvt_float2_to_fp4x2(const float2 src) {
  const unsigned char lo = __tl_cvt_float_to_fp4(src.x) & 0x0FU;
  const unsigned char hi = __tl_cvt_float_to_fp4(src.y) & 0x0FU;
  return lo | (hi << 4U);
}

// fp4_e2m1 -> double
TL_DEVICE double __tl_cvt_fp4_to_double(const unsigned char src) {
  return static_cast<double>(__tl_cvt_fp4_to_float(src));
}

// fp4_e2m1x2 -> double2
TL_DEVICE double2 __tl_cvt_fp4x2_to_double2(const unsigned char src) {
  float2 tmp = __tl_cvt_fp4x2_to_float2(src);
  double2 result;
  result.x = static_cast<double>(tmp.x);
  result.y = static_cast<double>(tmp.y);
  return result;
}

// double -> fp4_e2m1
TL_DEVICE unsigned char __tl_cvt_double_to_fp4(const double src) {
  return tl::detail::encode_fp4_e2m1(static_cast<float>(src));
}

// double2 -> fp4_e2m1x2
TL_DEVICE unsigned char __tl_cvt_double2_to_fp4x2(const double2 src) {
  const unsigned char lo = __tl_cvt_double_to_fp4(src.x) & 0x0FU;
  const unsigned char hi = __tl_cvt_double_to_fp4(src.y) & 0x0FU;
  return lo | (hi << 4U);
}

// fp4_e2m1 -> bfloat16
TL_DEVICE bfloat16_t __tl_cvt_fp4_to_bfloat16(const unsigned char src) {
  return bfloat16_t(__tl_cvt_fp4_to_float(src));
}

// fp4_e2m1x2 -> bfloat162
TL_DEVICE __maca_bfloat162
__tl_cvt_fp4x2_to_bfloat162(const unsigned char src) {
  float2 tmp = __tl_cvt_fp4x2_to_float2(src);
  return __floats2bfloat162_rn(tmp.x, tmp.y);
}

// bfloat16 -> fp4_e2m1
TL_DEVICE unsigned char __tl_cvt_bfloat16_to_fp4(const bfloat16_t src) {
  return tl::detail::encode_fp4_e2m1(static_cast<float>(src));
}

// bfloat162 -> fp4_e2m1x2
TL_DEVICE unsigned char
__tl_cvt_bfloat162_to_fp4x2(const __maca_bfloat162 src) {
  float2 tmp = __bfloat1622float2(src);
  const unsigned char lo = __tl_cvt_float_to_fp4(tmp.x) & 0x0FU;
  const unsigned char hi = __tl_cvt_float_to_fp4(tmp.y) & 0x0FU;
  return lo | (hi << 4U);
}

TL_DEVICE fp4_e2_t tl_fp4_packed_load(fp4_e2_2_t *packed, int idx) {
  return (idx & 1) ? packed[idx >> 1].y() : packed[idx >> 1].x();
}

TL_DEVICE void tl_fp4_packed_store(fp4_e2_2_t *packed, int idx, fp4_e2_t val) {
  if (idx & 1) {
    packed[idx >> 1].set_y(val);
  } else {
    packed[idx >> 1].set_x(val);
  }
}
