#pragma once

#include "mc_runtime.h"
#include <atomic>
#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
#include <mctlass/numeric_types.h>
#include <type_traits>

using half_t = __half;

using bfloat16_t = maca_bfloat16;

#define TL_DEVICE __forceinline__ __device__
#define TL_NOT_IMPLEMENTED()                                                   \
  do {                                                                         \
    printf("%s not implemented\n", __PRETTY_FUNCTION__);                       \
    __trap();                                                                  \
  } while (0)

template <typename T> struct normalize_atomic_type {
  using type = std::remove_cv_t<T>;
};

template <> struct normalize_atomic_type<half_t> {
  using type = half;
};

template <> struct normalize_atomic_type<bfloat16_t> {
  using type = maca_bfloat16;
};

template <> struct normalize_atomic_type<int64_t> {
  using type = unsigned long long;
};

template <typename T1, typename T2> TL_DEVICE T1 maca_cast(T2 val) {
  return T1(val);
}

template <> TL_DEVICE half maca_cast<half, float>(float val) {
  return __float2half(val);
}

template <> TL_DEVICE maca_bfloat16 maca_cast<maca_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 *address, T2 val,
                          int memory_order = int(std::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, float> || std::is_same_v<NT1, int>) {
    return static_cast<T1>(
        atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
  } else if constexpr (std::is_same_v<NT1, half> ||
                       std::is_same_v<NT1, maca_bfloat16>) {
    if (memory_order == int(std::memory_order_relaxed)) {
      return static_cast<T1>(
          atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
    } else {
      if constexpr (std::is_same_v<NT1, half>) {
        // fp16
        __half ret_val;
        unsigned short ret_val_cast =
            *reinterpret_cast<unsigned short *>(&ret_val);
        unsigned long long *ref_address =
            reinterpret_cast<unsigned long long *>(address);
        unsigned short val_cast = *reinterpret_cast<unsigned short *>(&val);
        ret_val_cast = atomicAdd(ref_address, val_cast);
        return static_cast<T1>(*reinterpret_cast<__half *>(&ret_val_cast));
      } else if constexpr (std::is_same_v<NT1, maca_bfloat16>) {
        // bf16
        maca_bfloat16 ret_val;
        unsigned short ret_val_cast =
            *reinterpret_cast<unsigned short *>(&ret_val);
        unsigned long long *ref_address =
            reinterpret_cast<unsigned long long *>(address);
        unsigned short val_cast = *reinterpret_cast<unsigned short *>(&val);
        ret_val_cast = atomicAdd(ref_address, val_cast);
        return static_cast<T1>(
            *reinterpret_cast<maca_bfloat16 *>(&ret_val_cast));
      }
    }
  } else {
    TL_NOT_IMPLEMENTED();
  }
}

// add max min
template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  atomicMax(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  return atomicMax(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  atomicMin(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  return atomicMin(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

TL_DEVICE inline void AtomicMax(float *address, float val,
                                int memory_order = 0) {
  (void)memory_order;
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    float f_assumed = __int_as_float(assumed);
    float f_max = (f_assumed > val) ? f_assumed : val;
    old = atomicCAS(address_as_i, assumed, __float_as_int(f_max));
  } while (assumed != old);
}

TL_DEVICE inline float AtomicMaxRet(float *address, float val,
                                    int memory_order = 0) {
  (void)memory_order;
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    float f_assumed = __int_as_float(assumed);
    float f_max = (f_assumed > val) ? f_assumed : val;
    old = atomicCAS(address_as_i, assumed, __float_as_int(f_max));
  } while (assumed != old);
  return __int_as_float(old);
}

TL_DEVICE inline void AtomicMin(float *address, float val,
                                int memory_order = 0) {
  (void)memory_order;
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    float f_assumed = __int_as_float(assumed);
    float f_min = (f_assumed < val) ? f_assumed : val;
    old = atomicCAS(address_as_i, assumed, __float_as_int(f_min));
  } while (assumed != old);
}

TL_DEVICE inline float AtomicMinRet(float *address, float val,
                                    int memory_order = 0) {
  (void)memory_order;
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    float f_assumed = __int_as_float(assumed);
    float f_min = (f_assumed < val) ? f_assumed : val;
    old = atomicCAS(address_as_i, assumed, __float_as_int(f_min));
  } while (assumed != old);
  return __int_as_float(old);
}

// add x2   x4
template <typename T1, typename T2>
TL_DEVICE void AtomicAddx2(T1 *ref, T2 *val, int memory_order = 0) {
  (void)memory_order;
  atomicAdd(reinterpret_cast<T1 *>(ref), static_cast<T1>(val[0]));
  atomicAdd(reinterpret_cast<T1 *>(ref + 1), static_cast<T1>(val[1]));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAddx2Ret(T1 *ref, T2 *val, T1 *ret, int memory_order = 0) {
  (void)memory_order;
  ret[0] = atomicAdd(reinterpret_cast<T1 *>(ref), static_cast<T1>(val[0]));
  ret[1] = atomicAdd(reinterpret_cast<T1 *>(ref + 1), static_cast<T1>(val[1]));
}
template <typename T1, typename T2>
TL_DEVICE void AtomicAddx4(T1 *ref, T2 *val, int memory_order = 0) {
  (void)memory_order;
  atomicAdd(reinterpret_cast<T1 *>(ref), static_cast<T1>(val[0]));
  atomicAdd(reinterpret_cast<T1 *>(ref + 1), static_cast<T1>(val[1]));
  atomicAdd(reinterpret_cast<T1 *>(ref + 2), static_cast<T1>(val[2]));
  atomicAdd(reinterpret_cast<T1 *>(ref + 3), static_cast<T1>(val[3]));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAddx4Ret(T1 *ref, T2 *val, T1 *ret, int memory_order = 0) {
  (void)memory_order;
  ret[0] = atomicAdd(reinterpret_cast<T1 *>(ref), static_cast<T1>(val[0]));
  ret[1] = atomicAdd(reinterpret_cast<T1 *>(ref + 1), static_cast<T1>(val[1]));
  ret[2] = atomicAdd(reinterpret_cast<T1 *>(ref + 2), static_cast<T1>(val[2]));
  ret[3] = atomicAdd(reinterpret_cast<T1 *>(ref + 3), static_cast<T1>(val[3]));
}

// For vectorized AtomicAdd, we maintain two versions of interfaces:
// 1. AtomicAddxN(dst_type* ref, src_type *val) // Pass pointer
// 2. AtomicAddxN(dst_type* ref, src_type val) // Pass value
template <typename T> TL_DEVICE half2 ToHalf2(T *val) {
  return *reinterpret_cast<const half2 *>(val);
}

template <typename T> TL_DEVICE half2 ToHalf2(T val) {
  return static_cast<half2>(*reinterpret_cast<const half2 *>(&val));
}

TL_DEVICE half2 ToHalf2(half2 val) { return val; }

TL_DEVICE half2 ToHalf2(float2 val) {
  half2 ret;
  ret.x = static_cast<half_t>(val.x);
  ret.y = static_cast<half_t>(val.y);
  return ret;
}

// Here ValType can be either value or value* (pointer)

template <typename ValType>
TL_DEVICE void AtomicAddx2(half_t *ref, ValType val,
                           int memory_order = int(std::memory_order_relaxed)) {
  (void)memory_order;
  half2 add_val = ToHalf2(val);
  atomicAdd(reinterpret_cast<half2 *>(ref), add_val);
}
template <typename T> TL_DEVICE maca_bfloat162 ToBfloat162(T *val) {
  return *reinterpret_cast<const maca_bfloat162 *>(val);
}

template <typename T> TL_DEVICE maca_bfloat162 ToBfloat162(T val) {
  return static_cast<maca_bfloat162>(
      *reinterpret_cast<const maca_bfloat162 *>(&val));
}

TL_DEVICE maca_bfloat162 ToBfloat162(maca_bfloat162 val) { return val; }

template <typename ValType>
TL_DEVICE void AtomicAddx2(bfloat16_t *ref, ValType val,
                           int memory_order = int(std::memory_order_relaxed)) {
  (void)memory_order;
  maca_bfloat162 add_val = ToBfloat162(val);
  atomicAdd(reinterpret_cast<maca_bfloat162 *>(ref), add_val);
}
