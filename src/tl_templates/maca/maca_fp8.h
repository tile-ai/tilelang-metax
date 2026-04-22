#pragma once

#include "common.h"
#include <maca_fp8.h>

TL_DEVICE half_t __cvt_fp8_e4m3_to_half(__maca_fp8_e4m3 x) {
  uint16_t bits = (uint16_t)x.__x;
  bits = (uint16_t)(bits << 8U);

  uint16_t sign = bits & 0x8000U;
  uint16_t exponent = (uint16_t)(((bits & 0x7800U) >> 1U) + 0x2000U);
  uint16_t mantissa = (bits & 0x0700U) >> 1U;
  unsigned char absx = 0x7FU & (unsigned char)x.__x;

  if (absx == 0x7FU) { // NaN
    bits = 0x7FFFU;    // fp16 canonical NaN, discard sign
  } else if (exponent == 0x2000U) {
    // zero or denormal
    if (mantissa != 0U) {
      // normalize
      mantissa = (uint16_t)(mantissa << 1U);
      while ((mantissa & 0x0400U) == 0U) {
        mantissa = (uint16_t)(mantissa << 1U);
        exponent = (uint16_t)(exponent - 0x0400U);
      }
      // discard implicit leading bit
      mantissa &= 0x03FFU;
    } else { // Zero
      exponent = 0U;
    }
    bits = (sign | exponent) | mantissa;
  } else {
    bits = (sign | exponent) | mantissa;
  }

  return *reinterpret_cast<half_t *>(&bits);
}

TL_DEVICE float __cvt_fp8_e4m3_to_float(__maca_fp8_e4m3 x) {
  uint32_t bits;
  uint32_t sign = ((uint32_t)x.__x & 0x80U) << 24;
  uint32_t exp4 = ((uint32_t)x.__x >> 3) & 0x0FU;
  uint32_t man3 = (uint32_t)x.__x & 0x07U;

  if (exp4 == 0U) {
    /* ±0 or subnormal: value = (-1)^s * man3 * 2^-9 */
    if (man3 == 0U) {
      bits = sign; /* ±0 */
    } else {
      switch (man3) {
      case 1U:                      /* 1 * 2^-9  */
        bits = sign | (118U << 23); /* 1.0 * 2^-9 */
        break;
      case 2U:                      /* 2 * 2^-9  */
        bits = sign | (119U << 23); /* 1.0 * 2^-8 */
        break;
      case 3U:                                  /* 3 * 2^-9  */
        bits = sign | (119U << 23) | 0x400000U; /* 1.5 * 2^-8 */
        break;
      case 4U:                      /* 4 * 2^-9  */
        bits = sign | (120U << 23); /* 1.0 * 2^-7 */
        break;
      case 5U:                                  /* 5 * 2^-9  */
        bits = sign | (120U << 23) | 0x200000U; /* 1.25 * 2^-7 */
        break;
      case 6U:                                  /* 6 * 2^-9  */
        bits = sign | (120U << 23) | 0x400000U; /* 1.5 * 2^-7 */
        break;
      case 7U:                                  /* 7 * 2^-9  */
        bits = sign | (120U << 23) | 0x600000U; /* 1.75 * 2^-7 */
        break;
      default:
        bits = sign;
        break;
      }
    }
  } else if (exp4 == 15U) {
    if (man3 == 7U) {
      /* quiet NaN */
      bits = sign | 0x7FC00000U;
    } else {
      /* normal decode: value = (-1)^s * (1 + man3/8) * 2^(exp4-7)
      float32: exp32 = (exp4-7)+127 = exp4+120, mant32 = man3 << 20 */
      bits = sign | ((exp4 + 120U) << 23) | (man3 << 20);
    }
  } else {
    /* 1 <= exp4 <= 14：normal decode */
    bits = sign | ((exp4 + 120U) << 23) | (man3 << 20);
  }

  return *reinterpret_cast<float *>(&bits);
}

struct fp8_e4_t {
  using value_t = __maca_fp8_e4m3;
  value_t v;

  TL_DEVICE constexpr fp8_e4_t() : v{} {}

  TL_DEVICE explicit fp8_e4_t(value_t x) : v(x) {}

  template <class T,
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, fp8_e4_t> &&
                                 std::is_constructible_v<value_t, T>,
                             int> = 0>
  TL_DEVICE explicit fp8_e4_t(T &&x) : v(std::forward<T>(x)) {}

  TL_DEVICE operator half_t() const { return __cvt_fp8_e4m3_to_half(v); }

  TL_DEVICE operator float() const { return __cvt_fp8_e4m3_to_float(v); }

private:
  template <class To, class = void>
  struct is_static_castable : std::false_type {};
  template <class To>
  struct is_static_castable<
      To, std::void_t<decltype(static_cast<To>(std::declval<value_t>()))>>
      : std::true_type {};

public:
  template <class To,
            std::enable_if_t<!std::is_same_v<std::decay_t<To>, half_t> &&
                                 !std::is_same_v<std::decay_t<To>, float> &&
                                 !std::is_same_v<std::decay_t<To>, value_t> &&
                                 !std::is_same_v<std::decay_t<To>, fp8_e4_t> &&
                                 is_static_castable<To>::value,
                             int> = 0>
  TL_DEVICE operator To() const {
    return static_cast<To>(v);
  }
};

TL_DEVICE half_t __cvt_fp8_e5m2_to_half(__maca_fp8_e5m2 x) {
  uint16_t bits = (uint16_t)x.__x;
  bits = (uint16_t)(bits << 8U);

  // FP8 e5m2 -> FP16 half bits
  uint16_t sign = bits & 0x8000U;     // bit15
  uint16_t exponent = bits & 0x7C00U; // bits14..10 (5-bit exp)
  uint16_t mantissa = bits & 0x0300U; // bits9..8 (2-bit mantissa)

  if ((exponent == 0x7C00U) && (mantissa != 0)) {
    mantissa |= 0x0200U; // quiet bit: half mantissa bit9
  }

  bits = (sign | exponent) | mantissa;

  return *reinterpret_cast<half_t *>(&bits);
}

TL_DEVICE float __cvt_fp8_e5m2_to_float(__maca_fp8_e5m2 x) {
  uint32_t bits;
  uint32_t sign = ((uint32_t)x.__x & 0x80U) << 24;
  uint32_t exp8 = ((uint32_t)x.__x >> 2) & 0x1FU;
  uint32_t man8 = (uint32_t)x.__x & 0x03U;

  if (exp8 == 0U) {
    if (man8 == 0U) {
      bits = sign;
    } else {
      uint32_t exp32, man32;
      switch (man8) {
      case 1U:              /* 1 * 2^-16 */
        exp32 = 127U - 16U; /* 111 */
        man32 = 0U;
        break;
      case 2U:              /* 2 * 2^-16 = 2^-15 */
        exp32 = 127U - 15U; /* 112 */
        man32 = 0U;
        break;
      case 3U:              /* 3 * 2^-16 = 1.5 * 2^-15 */
        exp32 = 127U - 15U; /* 112 */
        man32 = 0x400000U;
        break;
      default:
        exp32 = 0U;
        man32 = 0U;
        break;
      }
      bits = sign | (exp32 << 23) | man32;
    }
  } else if (exp8 == 31U) {
    if (man8 == 0U) {
      bits = sign | 0x7F800000U; /* ±Inf */
    } else {
      bits = sign | 0x7F800000U | (man8 << 21);
    }
  } else {
    uint32_t exp32 = exp8 + 112U;
    uint32_t man32 = man8 << 21U;
    bits = sign | (exp32 << 23) | man32;
  }

  return *reinterpret_cast<float *>(&bits);
}

struct fp8_e5_t {
  using value_t = __maca_fp8_e5m2;
  value_t v;

  TL_DEVICE constexpr fp8_e5_t() : v{} {}

  TL_DEVICE explicit fp8_e5_t(value_t x) : v(x) {}

  template <class T,
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, fp8_e5_t> &&
                                 std::is_constructible_v<value_t, T>,
                             int> = 0>
  TL_DEVICE explicit fp8_e5_t(T &&x) : v(std::forward<T>(x)) {}

  TL_DEVICE operator half_t() const { return __cvt_fp8_e5m2_to_half(v); }

  TL_DEVICE operator float() const { return __cvt_fp8_e5m2_to_float(v); }

private:
  template <class To, class = void>
  struct is_static_castable : std::false_type {};
  template <class To>
  struct is_static_castable<
      To, std::void_t<decltype(static_cast<To>(std::declval<value_t>()))>>
      : std::true_type {};

public:
  template <class To,
            std::enable_if_t<!std::is_same_v<std::decay_t<To>, half_t> &&
                                 !std::is_same_v<std::decay_t<To>, float> &&
                                 !std::is_same_v<std::decay_t<To>, value_t> &&
                                 !std::is_same_v<std::decay_t<To>, fp8_e5_t> &&
                                 is_static_castable<To>::value,
                             int> = 0>
  TL_DEVICE operator To() const {
    return static_cast<To>(v);
  }
};

struct fp8_e8_t {
  unsigned char data;
};

struct __MACA_ALIGN__(2) fp8_e4_2_t {
  fp8_e4_t x;
  fp8_e4_t y;
};

struct __MACA_ALIGN__(4) fp8_e4_4_t {
  fp8_e4_t x;
  fp8_e4_t y;
  fp8_e4_t z;
  fp8_e4_t w;
};

struct __MACA_ALIGN__(8) fp8_e4_8_t {
  fp8_e4_4_t x;
  fp8_e4_4_t y;
};

struct __MACA_ALIGN__(16) fp8_e4_16_t {
  fp8_e4_8_t x;
  fp8_e4_8_t y;
};

struct __MACA_ALIGN__(32) fp8_e4_32_t {
  fp8_e4_16_t x;
  fp8_e4_16_t y;

  TL_DEVICE fp8_e4_32_t &operator=(const ulonglong4 &rhs) {
    x.x = *(fp8_e4_8_t *)&rhs.x;
    x.y = *(fp8_e4_8_t *)&rhs.y;
    y.x = *(fp8_e4_8_t *)&rhs.z;
    y.y = *(fp8_e4_8_t *)&rhs.w;
    return *this;
  }
};

struct __MACA_ALIGN__(2) fp8_e5_2_t {
  fp8_e5_t x;
  fp8_e5_t y;
};

struct __MACA_ALIGN__(4) fp8_e5_4_t {
  fp8_e5_t x;
  fp8_e5_t y;
  fp8_e5_t z;
  fp8_e5_t w;
};

struct __MACA_ALIGN__(8) fp8_e5_8_t {
  fp8_e5_4_t x;
  fp8_e5_4_t y;
};

struct __MACA_ALIGN__(16) fp8_e5_16_t {
  fp8_e5_8_t x;
  fp8_e5_8_t y;
};

struct __MACA_ALIGN__(32) fp8_e5_32_t {
  fp8_e5_16_t x;
  fp8_e5_16_t y;

  TL_DEVICE fp8_e5_32_t &operator=(const ulonglong4 &rhs) {
    x.x = *(fp8_e5_8_t *)&rhs.x;
    x.y = *(fp8_e5_8_t *)&rhs.y;
    y.x = *(fp8_e5_8_t *)&rhs.z;
    y.y = *(fp8_e5_8_t *)&rhs.w;
    return *this;
  }
};

struct __MACA_ALIGN__(2) fp8_e8_2_t {
  fp8_e8_t x;
  fp8_e8_t y;
};

struct __MACA_ALIGN__(4) fp8_e8_4_t {
  fp8_e8_t x;
  fp8_e8_t y;
  fp8_e8_t z;
  fp8_e8_t w;
};

struct __MACA_ALIGN__(8) fp8_e8_8_t {
  fp8_e8_4_t x;
  fp8_e8_4_t y;
};

struct __MACA_ALIGN__(16) fp8_e8_16_t {
  fp8_e8_8_t x;
  fp8_e8_8_t y;
};

struct __MACA_ALIGN__(32) fp8_e8_32_t {
  fp8_e8_16_t x;
  fp8_e8_16_t y;

  TL_DEVICE fp8_e8_32_t &operator=(const ulonglong4 &rhs) {
    x.x = *(fp8_e8_8_t *)&rhs.x;
    x.y = *(fp8_e8_8_t *)&rhs.y;
    y.x = *(fp8_e8_8_t *)&rhs.z;
    y.y = *(fp8_e8_8_t *)&rhs.w;
    return *this;
  }
};

// Pack two fp8_e4_t values.
TL_DEVICE fp8_e4_2_t make_fp8_e4_2_t(fp8_e4_t x, fp8_e4_t y) {
  fp8_e4_2_t result;
  result.x = x;
  result.y = y;
  return result;
}

// Pack four fp8_e4_t values.
TL_DEVICE fp8_e4_4_t make_fp8_e4_4_t(fp8_e4_t x0, fp8_e4_t x1, fp8_e4_t x2,
                                     fp8_e4_t x3) {
  fp8_e4_4_t result;
  result.x = x0;
  result.y = x1;
  result.z = x2;
  result.w = x3;
  return result;
}

// Pack eight fp8_e4_t values.
TL_DEVICE fp8_e4_8_t make_fp8_e4_8_t(fp8_e4_t x0, fp8_e4_t x1, fp8_e4_t x2,
                                     fp8_e4_t x3, fp8_e4_t x4, fp8_e4_t x5,
                                     fp8_e4_t x6, fp8_e4_t x7) {
  fp8_e4_8_t result;
  result.x = make_fp8_e4_4_t(x0, x1, x2, x3);
  result.y = make_fp8_e4_4_t(x4, x5, x6, x7);
  return result;
}

// Pack sixteen fp8_e4_t values.
TL_DEVICE fp8_e4_16_t make_fp8_e4_16_t(fp8_e4_t x0, fp8_e4_t x1, fp8_e4_t x2,
                                       fp8_e4_t x3, fp8_e4_t x4, fp8_e4_t x5,
                                       fp8_e4_t x6, fp8_e4_t x7, fp8_e4_t y0,
                                       fp8_e4_t y1, fp8_e4_t y2, fp8_e4_t y3,
                                       fp8_e4_t y4, fp8_e4_t y5, fp8_e4_t y6,
                                       fp8_e4_t y7) {
  fp8_e4_16_t result;
  result.x = make_fp8_e4_8_t(x0, x1, x2, x3, x4, x5, x6, x7);
  result.y = make_fp8_e4_8_t(y0, y1, y2, y3, y4, y5, y6, y7);
  return result;
}

// Pack thirty-two fp8_e4_t values.
TL_DEVICE fp8_e4_32_t make_fp8_e4_32_t(
    fp8_e4_t x0, fp8_e4_t x1, fp8_e4_t x2, fp8_e4_t x3, fp8_e4_t x4,
    fp8_e4_t x5, fp8_e4_t x6, fp8_e4_t x7, fp8_e4_t x8, fp8_e4_t x9,
    fp8_e4_t x10, fp8_e4_t x11, fp8_e4_t x12, fp8_e4_t x13, fp8_e4_t x14,
    fp8_e4_t x15, fp8_e4_t y0, fp8_e4_t y1, fp8_e4_t y2, fp8_e4_t y3,
    fp8_e4_t y4, fp8_e4_t y5, fp8_e4_t y6, fp8_e4_t y7, fp8_e4_t y8,
    fp8_e4_t y9, fp8_e4_t y10, fp8_e4_t y11, fp8_e4_t y12, fp8_e4_t y13,
    fp8_e4_t y14, fp8_e4_t y15) {
  fp8_e4_32_t result;
  result.x = make_fp8_e4_16_t(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                              x12, x13, x14, x15);
  result.y = make_fp8_e4_16_t(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11,
                              y12, y13, y14, y15);
  return result;
}

// Pack two fp8_e5_t values.
TL_DEVICE fp8_e5_2_t make_fp8_e5_2_t(fp8_e5_t x, fp8_e5_t y) {
  fp8_e5_2_t result;
  result.x = x;
  result.y = y;
  return result;
}

// Pack four fp8_e5_t values.
TL_DEVICE fp8_e5_4_t make_fp8_e5_4_t(fp8_e5_t x0, fp8_e5_t x1, fp8_e5_t x2,
                                     fp8_e5_t x3) {
  fp8_e5_4_t result;
  result.x = x0;
  result.y = x1;
  result.z = x2;
  result.w = x3;
  return result;
}

// Pack eight fp8_e5_t values.
TL_DEVICE fp8_e5_8_t make_fp8_e5_8_t(fp8_e5_t x0, fp8_e5_t x1, fp8_e5_t x2,
                                     fp8_e5_t x3, fp8_e5_t x4, fp8_e5_t x5,
                                     fp8_e5_t x6, fp8_e5_t x7) {
  fp8_e5_8_t result;
  result.x = make_fp8_e5_4_t(x0, x1, x2, x3);
  result.y = make_fp8_e5_4_t(x4, x5, x6, x7);
  return result;
}

// Pack sixteen fp8_e5_t values.
TL_DEVICE fp8_e5_16_t make_fp8_e5_16_t(fp8_e5_t x0, fp8_e5_t x1, fp8_e5_t x2,
                                       fp8_e5_t x3, fp8_e5_t x4, fp8_e5_t x5,
                                       fp8_e5_t x6, fp8_e5_t x7, fp8_e5_t y0,
                                       fp8_e5_t y1, fp8_e5_t y2, fp8_e5_t y3,
                                       fp8_e5_t y4, fp8_e5_t y5, fp8_e5_t y6,
                                       fp8_e5_t y7) {
  fp8_e5_16_t result;
  result.x = make_fp8_e5_8_t(x0, x1, x2, x3, x4, x5, x6, x7);
  result.y = make_fp8_e5_8_t(y0, y1, y2, y3, y4, y5, y6, y7);
  return result;
}

// Pack thirty-two fp8_e5_t values.
TL_DEVICE fp8_e5_32_t make_fp8_e5_32_t(
    fp8_e5_t x0, fp8_e5_t x1, fp8_e5_t x2, fp8_e5_t x3, fp8_e5_t x4,
    fp8_e5_t x5, fp8_e5_t x6, fp8_e5_t x7, fp8_e5_t x8, fp8_e5_t x9,
    fp8_e5_t x10, fp8_e5_t x11, fp8_e5_t x12, fp8_e5_t x13, fp8_e5_t x14,
    fp8_e5_t x15, fp8_e5_t y0, fp8_e5_t y1, fp8_e5_t y2, fp8_e5_t y3,
    fp8_e5_t y4, fp8_e5_t y5, fp8_e5_t y6, fp8_e5_t y7, fp8_e5_t y8,
    fp8_e5_t y9, fp8_e5_t y10, fp8_e5_t y11, fp8_e5_t y12, fp8_e5_t y13,
    fp8_e5_t y14, fp8_e5_t y15) {
  fp8_e5_32_t result;
  result.x = make_fp8_e5_16_t(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                              x12, x13, x14, x15);
  result.y = make_fp8_e5_16_t(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11,
                              y12, y13, y14, y15);
  return result;
}

// Pack two fp8_e8_t values.
TL_DEVICE fp8_e8_2_t make_fp8_e8_2_t(fp8_e8_t x, fp8_e8_t y) {
  fp8_e8_2_t result;
  result.x = x;
  result.y = y;
  return result;
}

// Pack four fp8_e8_t values.
TL_DEVICE fp8_e8_4_t make_fp8_e8_4_t(fp8_e8_t x0, fp8_e8_t x1, fp8_e8_t x2,
                                     fp8_e8_t x3) {
  fp8_e8_4_t result;
  result.x = x0;
  result.y = x1;
  result.z = x2;
  result.w = x3;
  return result;
}

// Pack eight fp8_e8_t values.
TL_DEVICE fp8_e8_8_t make_fp8_e8_8_t(fp8_e8_t x0, fp8_e8_t x1, fp8_e8_t x2,
                                     fp8_e8_t x3, fp8_e8_t x4, fp8_e8_t x5,
                                     fp8_e8_t x6, fp8_e8_t x7) {
  fp8_e8_8_t result;
  result.x = make_fp8_e8_4_t(x0, x1, x2, x3);
  result.y = make_fp8_e8_4_t(x4, x5, x6, x7);
  return result;
}

// Pack sixteen fp8_e8_t values.
TL_DEVICE fp8_e8_16_t make_fp8_e8_16_t(fp8_e8_t x0, fp8_e8_t x1, fp8_e8_t x2,
                                       fp8_e8_t x3, fp8_e8_t x4, fp8_e8_t x5,
                                       fp8_e8_t x6, fp8_e8_t x7, fp8_e8_t y0,
                                       fp8_e8_t y1, fp8_e8_t y2, fp8_e8_t y3,
                                       fp8_e8_t y4, fp8_e8_t y5, fp8_e8_t y6,
                                       fp8_e8_t y7) {
  fp8_e8_16_t result;
  result.x = make_fp8_e8_8_t(x0, x1, x2, x3, x4, x5, x6, x7);
  result.y = make_fp8_e8_8_t(y0, y1, y2, y3, y4, y5, y6, y7);
  return result;
}

// Pack thirty-two fp8_e8_t values.
TL_DEVICE fp8_e8_32_t make_fp8_e8_32_t(
    fp8_e8_t x0, fp8_e8_t x1, fp8_e8_t x2, fp8_e8_t x3, fp8_e8_t x4,
    fp8_e8_t x5, fp8_e8_t x6, fp8_e8_t x7, fp8_e8_t x8, fp8_e8_t x9,
    fp8_e8_t x10, fp8_e8_t x11, fp8_e8_t x12, fp8_e8_t x13, fp8_e8_t x14,
    fp8_e8_t x15, fp8_e8_t y0, fp8_e8_t y1, fp8_e8_t y2, fp8_e8_t y3,
    fp8_e8_t y4, fp8_e8_t y5, fp8_e8_t y6, fp8_e8_t y7, fp8_e8_t y8,
    fp8_e8_t y9, fp8_e8_t y10, fp8_e8_t y11, fp8_e8_t y12, fp8_e8_t y13,
    fp8_e8_t y14, fp8_e8_t y15) {
  fp8_e8_32_t result;
  result.x = make_fp8_e8_16_t(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                              x12, x13, x14, x15);
  result.y = make_fp8_e8_16_t(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11,
                              y12, y13, y14, y15);
  return result;
}

TL_DEVICE float __tl_decode_fp8_e4m3_to_float(unsigned char bits) {
  const float sign = (bits & 0x80) ? -1.0f : 1.0f;
  const int exponent = (bits >> 3) & 0x0f;
  const int mantissa = bits & 0x07;

  if (exponent == 0) {
    if (mantissa == 0) {
      return (bits & 0x80) ? MACART_NEG_ZERO_F : 0.0f;
    }
    return sign * ldexpf(static_cast<float>(mantissa), -9);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return MACART_NAN_F;
  }

  return sign * ldexpf(1.0f + static_cast<float>(mantissa) * (1.0f / 8.0f),
                       exponent - 7);
}

TL_DEVICE float __tl_decode_fp8_e5m2_to_float(unsigned char bits) {
  const float sign = (bits & 0x80) ? -1.0f : 1.0f;
  const int exponent = (bits >> 2) & 0x1f;
  const int mantissa = bits & 0x03;
  if (exponent == 0) {
    if (mantissa == 0) {
      return (bits & 0x80) ? MACART_NEG_ZERO_F : 0.0f;
    }
    return sign * ldexpf(static_cast<float>(mantissa), -16);
  }
  if (exponent == 0x1f) {
    if (mantissa == 0) {
      return (bits & 0x80) ? MACART_NEGINF_F : MACART_INF_F;
    }
    return MACART_NAN_F;
  }
  return sign *
         ldexpf(1.0f + static_cast<float>(mantissa) * 0.25f, exponent - 15);
}

// e4m3x2 -> float2
TL_DEVICE float2
__tl_cvt_fp8x2_to_float2(const __maca_fp8x2_storage_t x,
                         const __maca_fp8_interpretation_t fp8_interpretation) {
  const unsigned char lo = static_cast<unsigned char>(x & 0xff);
  const unsigned char hi = static_cast<unsigned char>((x >> 8) & 0xff);
  float2 result;
  if (fp8_interpretation == __MACA_E4M3) {
    result.x = __tl_decode_fp8_e4m3_to_float(lo);
    result.y = __tl_decode_fp8_e4m3_to_float(hi);
  } else {
    result.x = __tl_decode_fp8_e5m2_to_float(lo);
    result.y = __tl_decode_fp8_e5m2_to_float(hi);
  }
  return result;
}
