#pragma once

#include "common.h"
#include <maca_fp8.h>

TL_DEVICE half_t __cvt_fp8_e4m3_to_half(__maca_fp8_e4m3 x) {
  unsigned char bits = x.__x;
  uint16_t sign = (bits & 0x80) ? 0x8000U : 0U;
  uint16_t exp4 = (bits >> 3) & 0x0FU;
  uint16_t man3 = bits & 0x07U;

  // NaN: E4M3FN encodes NaN as 0x7F (exp=15, mant=7)
  if ((bits & 0x7F) == 0x7F) {
    uint16_t nan_bits = 0x7FFFU;
    return *reinterpret_cast<half_t *>(&nan_bits);
  }

  uint16_t fp16_bits;

  if (exp4 == 0) {
    // Zero or denormal E4M3
    if (man3 == 0) {
      fp16_bits = sign; // Zero
    } else {
      // E4M3 denormal: value = man3 * 2^-9
      // Convert to FP16: man3 * 64 = 2^exp16 * (1 + man16/1024)
      uint16_t scale = man3 << 6;

      int msb_pos = 0;
      uint16_t temp = scale;
      while (temp > 1) {
        temp >>= 1;
        msb_pos++;
      }

      uint16_t fp16_exp = msb_pos;
      uint16_t fp16_man = ((scale - (1U << msb_pos)) * 1024) >> msb_pos;
      fp16_bits = sign | (fp16_exp << 10) | (fp16_man & 0x03FF);
    }
  } else {
    // Normal E4M3: exp16 = exp4 + 8, man16 = man3 << 7
    uint16_t fp16_exp = exp4 + 8;
    uint16_t fp16_man = man3 << 7;

    if (fp16_exp >= 31) {
      fp16_bits = sign | 0x7C00U; // FP16 infinity
    } else {
      fp16_bits = sign | (fp16_exp << 10) | fp16_man;
    }
  }

  return *reinterpret_cast<half_t *>(&fp16_bits);
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

TL_DEVICE bool operator==(fp8_e4_t lhs, fp8_e4_t rhs) {
  return lhs.v.__x == rhs.v.__x;
}

TL_DEVICE bool operator!=(fp8_e4_t lhs, fp8_e4_t rhs) {
  return lhs.v.__x != rhs.v.__x;
}

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

TL_DEVICE bool operator==(fp8_e5_t lhs, fp8_e5_t rhs) {
  return lhs.v.__x == rhs.v.__x;
}

TL_DEVICE bool operator!=(fp8_e5_t lhs, fp8_e5_t rhs) {
  return lhs.v.__x != rhs.v.__x;
}

TL_DEVICE unsigned char __tl_cvt_float_to_e8m0(const float src);
TL_DEVICE unsigned char __tl_cvt_double_to_e8m0(const double src);
TL_DEVICE unsigned char __tl_cvt_int_to_e8m0(const int src);

struct fp8_e8_t {
  using value_t = unsigned char;
  value_t v;

  TL_DEVICE constexpr fp8_e8_t() : v{} {}

  // construct from underlying storage
  TL_DEVICE explicit fp8_e8_t(value_t x) : v(x) {}

  TL_DEVICE explicit fp8_e8_t(int x) : v(__tl_cvt_int_to_e8m0(x)) {}
  TL_DEVICE explicit fp8_e8_t(float x) : v(__tl_cvt_float_to_e8m0(x)) {}
  TL_DEVICE explicit fp8_e8_t(double x) : v(__tl_cvt_double_to_e8m0(x)) {}

  // allow construction from types that can initialize the storage (e.g. int)
  template <class T,
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, fp8_e8_t> &&
                                 std::is_constructible_v<value_t, T>,
                             int> = 0>
  TL_DEVICE explicit fp8_e8_t(T &&x)
      : v(static_cast<value_t>(std::forward<T>(x))) {}

  // assignment from underlying storage
  TL_DEVICE fp8_e8_t &operator=(value_t x) {
    v = x;
    return *this;
  }

  // implicit cast back to underlying storage when needed
  TL_DEVICE operator value_t() const { return v; }

private:
  template <class To, class = void>
  struct is_static_castable : std::false_type {};
  template <class To>
  struct is_static_castable<
      To, std::void_t<decltype(static_cast<To>(std::declval<value_t>()))>>
      : std::true_type {};

public:
  template <class To,
            std::enable_if_t<!std::is_same_v<std::decay_t<To>, value_t> &&
                                 !std::is_same_v<std::decay_t<To>, fp8_e8_t> &&
                                 is_static_castable<To>::value,
                             int> = 0>
  TL_DEVICE operator To() const {
    return static_cast<To>(v);
  }
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

// e4m3x2 -> half2
// Software implementation for E4M3FN to ensure correct denormal handling
TL_DEVICE half2
__tl_cvt_fp8x2_to_half2(const __maca_fp8x2_storage_t x,
                        const __maca_fp8_interpretation_t fp8_interpretation) {
  // Extract low and high bytes
  const unsigned char lo = static_cast<unsigned char>(x & 0xff);
  const unsigned char hi = static_cast<unsigned char>((x >> 8) & 0xff);

  // Helper: E4M3 -> FP16 bits
  auto cvt_e4m3_bits = [](unsigned char bits) -> uint16_t {
    uint16_t sign = (bits & 0x80) ? 0x8000U : 0U;
    uint16_t exp4 = (bits >> 3) & 0x0FU;
    uint16_t man3 = bits & 0x07U;

    if ((bits & 0x7F) == 0x7F) {
      return 0x7FFFU; // FP16 canonical NaN
    }

    if (exp4 == 0) {
      if (man3 == 0) {
        return sign;
      }
      uint16_t scale = man3 << 6;
      int msb_pos = 0;
      uint16_t temp = scale;
      while (temp > 1) {
        temp >>= 1;
        msb_pos++;
      }
      uint16_t fp16_exp = msb_pos;
      uint16_t fp16_man = ((scale - (1U << msb_pos)) * 1024) >> msb_pos;
      return sign | (fp16_exp << 10) | (fp16_man & 0x03FF);
    }

    uint16_t fp16_exp = exp4 + 8;
    uint16_t fp16_man = man3 << 7;
    if (fp16_exp >= 31) {
      return sign | 0x7C00U;
    }
    return sign | (fp16_exp << 10) | fp16_man;
  };

  // Helper: E5M2 -> FP16 bits (simple bit manipulation, SDK is correct for
  // E5M2)
  auto cvt_e5m2_bits = [](unsigned char bits) -> uint16_t {
    uint16_t sign = (bits & 0x80) ? 0x8000U : 0U;
    uint16_t be = static_cast<uint16_t>(bits) << 8;
    uint16_t exponent = be & 0x7C00U;
    uint16_t mantissa = be & 0x0300U;
    if (exponent == 0x7C00U && mantissa != 0) {
      mantissa |= 0x0200U; // Quiet NaN
    }
    return sign | exponent | mantissa;
  };

  __half2_raw raw;
  if (fp8_interpretation == __MACA_E4M3) {
    raw.x = cvt_e4m3_bits(lo);
    raw.y = cvt_e4m3_bits(hi);
  } else {
    raw.x = cvt_e5m2_bits(lo);
    raw.y = cvt_e5m2_bits(hi);
  }
  return *reinterpret_cast<half2 *>(&raw);
}

// Helper: Convert a single FP16 (as uint16_t bits) to E4M3FN byte.
// This follows the standard E4M3FN specification compatible with PyTorch:
// - No infinities (saturate to max finite value)
// - NaN encoded as 0x7F (exponent=15, mantissa=7)
// - Round-to-nearest-even rounding
static TL_DEVICE unsigned char __tl_cvt_half_bits_to_e4m3_fn(uint16_t h) {
  const uint16_t sign = h & 0x8000U;
  const uint16_t exp_h = (h >> 10) & 0x1FU;
  const uint16_t man_h = h & 0x03FFU;

  // Handle zero and denormals in FP16
  if (exp_h == 0) {
    // FP16 denormal/zero -> FP8 denormal/zero (with possible underflow to zero)
    // FP16 denormal range: 2^-24 to 2^-15 * (2 - 2^-10)
    // FP8 E4M3 denormal range: 2^-9 to 2^-7 * (1 + 7/8)
    // Very small FP16 denormals underflow to FP8 zero
    if (man_h == 0) {
      return static_cast<unsigned char>(sign >> 8); // Preserve sign of zero
    }
    // FP16 denormal to float, then convert
    // This is slow but correct for edge cases
    float f = __half2float(*reinterpret_cast<const half *>(&h));
    // Convert via float
    uint32_t bits = *reinterpret_cast<uint32_t *>(&f);
    uint32_t sign_f = bits & 0x80000000U;
    int32_t exp_f = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
    uint32_t man_f = bits & 0x007FFFFFU;

    if (exp_f < -9) {
      // Underflow to zero
      return static_cast<unsigned char>(sign >> 8);
    }
    // Compute E4M3 representation
    // Target: value = (-1)^s * 2^(exp4-7) * (1 + man3/8)
    int exp4 = exp_f + 7;
    uint32_t man3 = (man_f >> 20) & 0x7; // Take top 3 bits of mantissa
    // Handle rounding for denormals
    uint32_t round_bit = (man_f >> 19) & 1;
    uint32_t sticky = (man_f & 0x7FFFF) != 0 ? 1 : 0;
    if (round_bit && (sticky || (man3 & 1))) {
      man3++;
      if (man3 > 7) {
        man3 = 0;
        exp4++;
      }
    }
    if (exp4 <= 0) {
      return static_cast<unsigned char>(sign >> 8);
    }
    if (exp4 >= 15) {
      // Saturate to max finite value (no infinity in E4M3FN)
      return static_cast<unsigned char>((sign >> 8) | 0x7E);
    }
    return static_cast<unsigned char>((sign >> 8) | ((exp4 << 3) & 0x78) |
                                      man3);
  }

  // Handle infinity/NaN in FP16
  if (exp_h == 31) {
    if (man_h == 0) {
      // FP16 infinity -> saturate to max finite E4M3FN value
      return static_cast<unsigned char>((sign >> 8) | 0x7E);
    }
    // FP16 NaN -> E4M3FN NaN (0x7F)
    return 0x7F;
  }

  // Normal FP16 number
  // FP16: value = (-1)^s * 2^(exp_h-15) * (1 + man_h/1024)
  // E4M3FN: value = (-1)^s * 2^(exp4-7) * (1 + man3/8)
  // exp4 = exp_h - 15 + 7 = exp_h - 8
  int exp4 = static_cast<int>(exp_h) - 8;

  if (exp4 <= 0) {
    // Underflow: try to represent as denormal or zero
    // E4M3FN denormal range: 2^-9 to 2^-7 * (1 + 7/8)
    // Denormal E4M3FN uses effective exponent of -8 (exp4=0)
    // Value = man3/8 * 2^-9 = man3 * 2^-12
    // FP16 value = 2^(exp_h-15) * (1 + man_h/1024)

    if (exp_h < 4) {
      // Too small to represent even as denormal, round to zero
      return static_cast<unsigned char>(sign >> 8);
    }

    // Convert to denormal E4M3
    // We need: man3/8 * 2^-9 = 2^(exp_h-15) * (1 + man_h/1024)
    // man3 = (1 + man_h/1024) * 2^(exp_h-15+9) * 8
    //      = (1024 + man_h) * 2^(exp_h-15+9-10) = (1024 + man_h) * 2^(exp_h-16)
    // But for exp_h=6: man3 = (1024 + man_h) * 2^-10 = (1024 + man_h) / 1024 ≈
    // 1 For exp_h=7: man3 = (1024 + man_h) / 512 ≈ 2-4 So we need to shift
    // right by (16 - exp_h) bits to get mantissa

    int shift_right =
        16 - static_cast<int>(exp_h); // Number of bits to shift right
    // Extended mantissa: 11 bits (implicit 1 + 10 explicit)
    uint32_t extended_m = (1U << 10) | man_h;
    // We need 3 bits + guard bits for rounding
    // Shift right to get the final 3-bit mantissa
    uint32_t shifted =
        extended_m >> (shift_right - 1); // Keep 1 extra bit for rounding

    uint32_t man3 = (shifted >> 1) & 0x7;
    uint32_t guard_bit = shifted & 1;
    uint32_t sticky =
        (extended_m & ((1U << (shift_right - 1)) - 1)) != 0 ? 1 : 0;

    // Round to nearest even
    if (guard_bit && (sticky || (man3 & 1))) {
      man3++;
      if (man3 > 7) {
        // Rounding caused overflow to normal number
        exp4 = 1;
        man3 = 0;
        if (exp4 >= 15) {
          return static_cast<unsigned char>((sign >> 8) | 0x7E);
        }
        return static_cast<unsigned char>((sign >> 8) | ((exp4 << 3) & 0x78) |
                                          man3);
      }
    }

    return static_cast<unsigned char>((sign >> 8) | man3);
  }

  // Normal case: exp4 >= 1
  // Need to convert mantissa from 10 bits to 3 bits with rounding
  uint32_t man3 = (man_h >> 7) & 0x7; // Top 3 bits
  uint32_t round_bit = (man_h >> 6) & 1;
  uint32_t sticky = (man_h & 0x3F) != 0 ? 1 : 0;

  // Round to nearest even
  if (round_bit && (sticky || (man3 & 1))) {
    man3++;
    if (man3 > 7) {
      man3 = 0;
      exp4++;
    }
  }

  // Check for overflow after rounding
  if (exp4 >= 15) {
    // Saturate to max finite value (no infinity in E4M3FN)
    return static_cast<unsigned char>((sign >> 8) | 0x7E);
  }

  return static_cast<unsigned char>((sign >> 8) | ((exp4 << 3) & 0x78) | man3);
}

// Helper: Convert a single FP16 (as uint16_t bits) to E5M2 byte.
// E5M2 format: 1 sign, 5 exp, 2 mantissa (same bias as FP16 = 15)
// - Has infinities and NaN
// - Round-to-nearest-even rounding
static TL_DEVICE unsigned char __tl_cvt_half_bits_to_e5m2(uint16_t h) {
  const uint16_t sign = h & 0x8000U;
  const uint16_t exp_h = (h >> 10) & 0x1FU;
  const uint16_t man_h = h & 0x03FFU;

  // Handle zero and denormals in FP16
  if (exp_h == 0) {
    // FP16 denormal/zero -> FP8 E5M2 denormal/zero
    // E5M2 denormal: value = man2 * 2^-16 (exp=0)
    // FP16 denormal: value = man10 * 2^-24
    if (man_h == 0) {
      return static_cast<unsigned char>(sign >> 8); // Zero
    }
    // FP16 denormal values are very small, most will underflow to E5M2 zero
    // E5M2 denormal range: 2^-16 to 2^-15 * (1 + 3/4) ≈ 2^-16 to 1.75*2^-15
    // FP16 denormal range: 2^-24 to ~2^-15
    // Only FP16 denormals with value >= 2^-16 can be represented
    // That means man_h * 2^-24 >= 2^-16 -> man_h >= 2^8 = 256
    if (man_h < 256) {
      // Underflow to zero
      return static_cast<unsigned char>(sign >> 8);
    }
    // Convert FP16 denormal to E5M2 denormal
    // value = man_h * 2^-24 = man2 * 2^-16
    // man2 = man_h * 2^-8 = man_h >> 8
    // But we need to round properly
    uint32_t man2 = man_h >> 8;
    uint32_t round_bit = (man_h >> 7) & 1;
    uint32_t sticky = (man_h & 0x7F) != 0 ? 1 : 0;
    if (round_bit && (sticky || (man2 & 1))) {
      man2++;
      if (man2 > 3) {
        man2 = 0;
        // Would become normal with exp=1
        return static_cast<unsigned char>((sign >> 8) | (1 << 2));
      }
    }
    return static_cast<unsigned char>((sign >> 8) | man2);
  }

  // Handle infinity/NaN in FP16
  if (exp_h == 31) {
    if (man_h == 0) {
      // FP16 infinity -> E5M2 infinity (exp=31, mant=0)
      return static_cast<unsigned char>((sign >> 8) | 0x7C);
    }
    // FP16 NaN -> E5M2 NaN (exp=31, mant!=0)
    return static_cast<unsigned char>((sign >> 8) | 0x7D); // Quiet NaN
  }

  // Normal FP16 number
  // FP16: value = (-1)^s * 2^(exp_h-15) * (1 + man_h/1024)
  // E5M2: value = (-1)^s * 2^(exp5-15) * (1 + man2/4)
  // Since bias is the same, exp5 = exp_h for normal numbers
  // man2 = man_h >> 8 (take top 2 bits of mantissa)
  uint32_t exp5 = exp_h;
  uint32_t man2 = (man_h >> 8) & 0x3;
  uint32_t round_bit = (man_h >> 7) & 1;
  uint32_t sticky = (man_h & 0x7F) != 0 ? 1 : 0;

  // Round to nearest even
  if (round_bit && (sticky || (man2 & 1))) {
    man2++;
    if (man2 > 3) {
      man2 = 0;
      exp5++;
    }
  }

  // Check for overflow
  if (exp5 >= 31) {
    // Overflow to infinity
    return static_cast<unsigned char>((sign >> 8) | 0x7C);
  }

  return static_cast<unsigned char>((sign >> 8) | ((exp5 << 2) & 0x7C) | man2);
}

// half2 -> fp8x2 (software implementation for E4M3FN and E5M2 compatibility)
TL_DEVICE __maca_fp8x2_storage_t __tl_cvt_half2_to_fp8x2(
    const half2 src, const __maca_fp8_interpretation_t fp8_interpretation) {
  __half2_raw raw = *reinterpret_cast<const __half2_raw *>(&src);

  if (fp8_interpretation == __MACA_E4M3) {
    // Use software conversion for E4M3FN to match PyTorch semantics
    unsigned char lo = __tl_cvt_half_bits_to_e4m3_fn(raw.x);
    unsigned char hi = __tl_cvt_half_bits_to_e4m3_fn(raw.y);
    return static_cast<__maca_fp8x2_storage_t>(lo) |
           (static_cast<__maca_fp8x2_storage_t>(hi) << 8);
  } else {
    // E5M2: also use software conversion for consistency
    unsigned char lo = __tl_cvt_half_bits_to_e5m2(raw.x);
    unsigned char hi = __tl_cvt_half_bits_to_e5m2(raw.y);
    return static_cast<__maca_fp8x2_storage_t>(lo) |
           (static_cast<__maca_fp8x2_storage_t>(hi) << 8);
  }
}

// Scalar fp8 -> half (native MACA intrinsic; single cvt on supported HW).
TL_DEVICE half
__tl_cvt_fp8_to_half(const __maca_fp8_storage_t x,
                     const __maca_fp8_interpretation_t fp8_interpretation) {
  __half_raw raw = __maca_cvt_fp8_to_halfraw(x, fp8_interpretation);
  return *reinterpret_cast<half *>(&raw);
}

// Scalar half -> fp8 (software implementation for E4M3FN and E5M2
// compatibility).
TL_DEVICE __maca_fp8_storage_t __tl_cvt_half_to_fp8(
    const half src, const __maca_fp8_interpretation_t fp8_interpretation) {
  __half_raw raw = *reinterpret_cast<const __half_raw *>(&src);
  if (fp8_interpretation == __MACA_E4M3) {
    return __tl_cvt_half_bits_to_e4m3_fn(raw.x);
  } else {
    return __tl_cvt_half_bits_to_e5m2(raw.x);
  }
}

// Scalar bfloat16 -> fp8 (native MACA intrinsic; single cvt on supported HW).
TL_DEVICE __maca_fp8_storage_t
__tl_cvt_bfloat16_to_fp8(const __maca_bfloat16 src,
                         const __maca_fp8_interpretation_t fp8_interpretation) {
  __maca_bfloat16_raw raw =
      *reinterpret_cast<const __maca_bfloat16_raw *>(&src);
  return __maca_cvt_bfloat16raw_to_fp8(raw, __MACA_SATFINITE,
                                       fp8_interpretation);
}

// Scalar fp8 -> bfloat16.
TL_DEVICE __maca_bfloat16
__tl_cvt_fp8_to_bfloat16(const __maca_fp8_storage_t x,
                         const __maca_fp8_interpretation_t fp8_interpretation) {
  __half_raw hr = __maca_cvt_fp8_to_halfraw(x, fp8_interpretation);
  return __float2bfloat16(__half2float(*reinterpret_cast<half *>(&hr)));
}

// e4m3x2 -> bfloat162
TL_DEVICE __maca_bfloat162
__tl_cvt_e4m3x2_to_bfloat162(const __maca_fp8x2_storage_t x) {
  half2 tmp = __maca_cvt_fp8x2_to_halfraw2(x, __MACA_E4M3);
  return __float22bfloat162_rn(make_float2((float)tmp.x, (float)tmp.y));
}

// e5m2x2 -> bfloat162
TL_DEVICE __maca_bfloat162
__tl_cvt_e5m2x2_to_bfloat162(const __maca_fp8x2_storage_t x) {
  half2 tmp = __maca_cvt_fp8x2_to_halfraw2(x, __MACA_E5M2);
  return __float22bfloat162_rn(make_float2((float)tmp.x, (float)tmp.y));
}

// bfloat162 -> e4m3x2
TL_DEVICE __maca_fp8x2_storage_t __tl_cvt_bfloat162_to_fp8x2(
    const __maca_bfloat162 src,
    const __maca_fp8_interpretation_t fp8_interpretation) {
  __maca_bfloat162_raw raw =
      *reinterpret_cast<const __maca_bfloat162_raw *>(&src);
  return __maca_cvt_bfloat16raw2_to_fp8x2(raw, __MACA_SATFINITE,
                                          fp8_interpretation);
}

// ==========================================================================
// FP8 E8M0 Related Conversions
// ==========================================================================

// e8m0 -> float
TL_DEVICE float __tl_cvt_e8m0_to_float(const unsigned char src) {
  unsigned int bits = (static_cast<unsigned int>(src) << 23);
  return *reinterpret_cast<float *>(&bits);
}

// e8m0 -> bfloat16
TL_DEVICE bfloat16_t __tl_cvt_e8m0_to_bfloat16(const unsigned char src) {
  return static_cast<bfloat16_t>(__tl_cvt_e8m0_to_float(src));
}

// e8m0x2 -> bfloat16x2
TL_DEVICE __maca_bfloat162
__tl_cvt_e8m0x2_to_bfloat162(const __maca_fp8x2_storage_t src) {
  unsigned char lo = static_cast<unsigned char>(src & 0xFF);
  unsigned char hi = static_cast<unsigned char>((src >> 8) & 0xFF);
  float f_lo = __tl_cvt_e8m0_to_float(lo);
  float f_hi = __tl_cvt_e8m0_to_float(hi);
  return __floats2bfloat162_rn(f_lo, f_hi);
}

// int8 -> e8m0
TL_DEVICE unsigned char __tl_cvt_int8_to_e8m0(const int8_t src) {
  if (src == 0) {
    return 0U;
  }
  const float f_val = static_cast<float>(src > 0 ? src : -src);
  return __tl_cvt_float_to_e8m0(f_val);
}

// int -> e8m0
TL_DEVICE unsigned char __tl_cvt_int_to_e8m0(const int src) {
  if (src == 0) {
    return 0U;
  }
  const float f_val = static_cast<float>(src > 0 ? src : -src);
  return __tl_cvt_float_to_e8m0(f_val);
}

// float -> e8m0
TL_DEVICE unsigned char __tl_cvt_float_to_e8m0(const float src) {
  unsigned int bits = *reinterpret_cast<const unsigned int *>(&src);
  unsigned int exponent = (bits >> 23) & 0xFF;
  unsigned int mantissa = bits & 0x7FFFFF;
  if (mantissa > 0 && exponent < 0xFF) {
    exponent += 1;
  }
  return static_cast<unsigned char>(exponent);
}

// float2 -> e8m0x2
TL_DEVICE __maca_fp8x2_storage_t __tl_cvt_float2_to_e8m0x2(const float2 src) {
  unsigned char lo = __tl_cvt_float_to_e8m0(src.x);
  unsigned char hi = __tl_cvt_float_to_e8m0(src.y);
  return static_cast<__maca_fp8x2_storage_t>(lo) |
         (static_cast<__maca_fp8x2_storage_t>(hi) << 8);
}

// double -> e8m0
TL_DEVICE unsigned char __tl_cvt_double_to_e8m0(const double src) {
  return __tl_cvt_float_to_e8m0(static_cast<float>(src));
}

// double2 -> e8m0x2
TL_DEVICE __maca_fp8x2_storage_t __tl_cvt_double2_to_e8m0x2(const double2 src) {
  unsigned char lo = __tl_cvt_double_to_e8m0(static_cast<float>(src.x));
  unsigned char hi = __tl_cvt_double_to_e8m0(static_cast<float>(src.y));
  return static_cast<__maca_fp8x2_storage_t>(lo) |
         (static_cast<__maca_fp8x2_storage_t>(hi) << 8);
}

// bfloat16 -> e8m0
TL_DEVICE unsigned char __tl_cvt_bfloat16_to_e8m0(const bfloat16_t src) {
  return __tl_cvt_float_to_e8m0(static_cast<float>(src));
}

// bfloat162 -> e8m0x2
TL_DEVICE __maca_fp8x2_storage_t
__tl_cvt_bfloat162_to_e8m0x2(const __maca_bfloat162 src) {
  const bfloat16_t *val_ptr = reinterpret_cast<const bfloat16_t *>(&src);

  float low_f = static_cast<float>(val_ptr[0]);
  float high_f = static_cast<float>(val_ptr[1]);

  unsigned char lo = __tl_cvt_float_to_e8m0(low_f);
  unsigned char hi = __tl_cvt_float_to_e8m0(high_f);

  return static_cast<__maca_fp8x2_storage_t>(lo) |
         (static_cast<__maca_fp8x2_storage_t>(hi) << 8);
}
