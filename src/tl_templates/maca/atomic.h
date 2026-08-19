#pragma once

#include "mc_runtime.h"
#include <atomic>
#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
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

template <typename T> TL_DEVICE unsigned short PackBits16(const T &val) {
  return *reinterpret_cast<const unsigned short *>(&val);
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
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, maca_bfloat16>) {
    // No native atomicMax for half/bf16 on MACA, use atomicCAS loop
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    NT1 val_cast = maca_cast<NT1>(val);
    unsigned short val_as_ushort = PackBits16(val_cast);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val_cast > *reinterpret_cast<NT1 *>(&old_val_ushort)) {
      unsigned short assumed = old_val_ushort;
      old_val_ushort = atomicCAS(address_as_ushort, assumed, val_as_ushort);
      if (assumed == old_val_ushort)
        break;
    }
  } else {
    atomicMax(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, maca_bfloat16>) {
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    NT1 val_cast = maca_cast<NT1>(val);
    unsigned short val_as_ushort = PackBits16(val_cast);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val_cast > *reinterpret_cast<NT1 *>(&old_val_ushort)) {
      unsigned short assumed = old_val_ushort;
      old_val_ushort = atomicCAS(address_as_ushort, assumed, val_as_ushort);
      if (assumed == old_val_ushort)
        break;
    }
    return static_cast<T1>(*reinterpret_cast<T1 *>(&old_val_ushort));
  } else {
    return atomicMax(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, maca_bfloat16>) {
    // No native atomicMin for half/bf16 on MACA, use atomicCAS loop
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    NT1 val_cast = maca_cast<NT1>(val);
    unsigned short val_as_ushort = PackBits16(val_cast);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val_cast < *reinterpret_cast<NT1 *>(&old_val_ushort)) {
      unsigned short assumed = old_val_ushort;
      old_val_ushort = atomicCAS(address_as_ushort, assumed, val_as_ushort);
      if (assumed == old_val_ushort)
        break;
    }
  } else {
    atomicMin(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, maca_bfloat16>) {
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    NT1 val_cast = maca_cast<NT1>(val);
    unsigned short val_as_ushort = PackBits16(val_cast);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val_cast < *reinterpret_cast<NT1 *>(&old_val_ushort)) {
      unsigned short assumed = old_val_ushort;
      old_val_ushort = atomicCAS(address_as_ushort, assumed, val_as_ushort);
      if (assumed == old_val_ushort)
        break;
    }
    return static_cast<T1>(*reinterpret_cast<T1 *>(&old_val_ushort));
  } else {
    return atomicMin(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
  }
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

TL_DEVICE half2 ToHalf2(float2 val) { return __floats2half2_rn(val.x, val.y); }
TL_DEVICE half2 ToHalf2(const float *val) {
  return __floats2half2_rn(val[0], val[1]);
}
TL_DEVICE half2 ToHalf2(float *val) {
  return ToHalf2(static_cast<const float *>(val));
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

TL_DEVICE maca_bfloat162 ToBfloat162(float2 val) {
  return __floats2bfloat162_rn(val.x, val.y);
}
TL_DEVICE maca_bfloat162 ToBfloat162(const float *val) {
  return __floats2bfloat162_rn(val[0], val[1]);
}
TL_DEVICE maca_bfloat162 ToBfloat162(float *val) {
  return ToBfloat162(static_cast<const float *>(val));
}

template <typename ValType>
TL_DEVICE void AtomicAddx2(bfloat16_t *ref, ValType val,
                           int memory_order = int(std::memory_order_relaxed)) {
  (void)memory_order;
  maca_bfloat162 add_val = ToBfloat162(val);
  atomicAdd(reinterpret_cast<maca_bfloat162 *>(ref), add_val);
}

// Float2/Float4 helpers for vectorized atomic add
template <typename T> TL_DEVICE float2 ToFloat2(T *val) {
  return *reinterpret_cast<const float2 *>(val);
}

TL_DEVICE float2 ToFloat2(float2 val) { return val; }
template <typename T> TL_DEVICE float4 ToFloat4(T *val) {
  return *reinterpret_cast<const float4 *>(val);
}

TL_DEVICE float4 ToFloat4(float4 val) { return val; }

// Scalar fallback for float AtomicAddx2
template <typename ValType>
TL_DEVICE void AtomicAddx2(float *ref, ValType val, int memory_order = 0) {
  (void)memory_order;
  float2 add_val = ToFloat2(val);
  atomicAdd(ref + 0, add_val.x);
  atomicAdd(ref + 1, add_val.y);
}

template <typename ValType>
TL_DEVICE float2 AtomicAddx2Ret(float *ref, ValType val, int memory_order = 0) {
  (void)memory_order;
  float2 add_val = ToFloat2(val);
  float2 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  return ret;
}

template <typename ValType>
TL_DEVICE half2 AtomicAddx2Ret(half_t *ref, ValType val, int memory_order = 0) {
  (void)memory_order;
  half2 add_val = ToHalf2(val);
  half2 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  return ret;
}

template <typename ValType>
TL_DEVICE __maca_bfloat162 AtomicAddx2Ret(bfloat16_t *ref, ValType val,
                                          int memory_order = 0) {
  (void)memory_order;
  __maca_bfloat162 add_val = *reinterpret_cast<const __maca_bfloat162 *>(val);
  __maca_bfloat162 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  return ret;
}

// Scalar fallback for float AtomicAddx4
template <typename dst_dtype, typename ValType>
TL_DEVICE void AtomicAddx4(dst_dtype *ref, ValType val, int memory_order = 0) {
  (void)memory_order;
  float4 add_val = ToFloat4(val);
  atomicAdd(ref + 0, add_val.x);
  atomicAdd(ref + 1, add_val.y);
  atomicAdd(ref + 2, add_val.z);
  atomicAdd(ref + 3, add_val.w);
}

template <typename SrcType>
TL_DEVICE void AtomicAddx4(half_t *ref, SrcType *val, int memory_order = 0) {
  AtomicAddx2(ref, val, memory_order);
  AtomicAddx2(ref + 2, val + 2, memory_order);
}

template <typename SrcType>
TL_DEVICE void AtomicAddx4(bfloat16_t *ref, SrcType *val,
                           int memory_order = 0) {
  AtomicAddx2(ref, val, memory_order);
  AtomicAddx2(ref + 2, val + 2, memory_order);
}

template <typename dst_dtype, typename ValType>
TL_DEVICE float4 AtomicAddx4Ret(dst_dtype *ref, ValType val,
                                int memory_order = 0) {
  (void)memory_order;
  float4 add_val = ToFloat4(val);
  float4 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  ret.z = atomicAdd(ref + 2, add_val.z);
  ret.w = atomicAdd(ref + 3, add_val.w);
  return ret;
}

// No single-atomic fp16x4 exists, so this is two per-pair AtomicAddx2Ret
// (per-pair atomic, like the fp32-x4 fallback). Returns uint2 (the half4 store
// type): the two half2 packed.
template <typename SrcType>
TL_DEVICE uint2 AtomicAddx4Ret(half_t *ref, SrcType *val,
                               int memory_order = 0) {
  half2 prev_lo = AtomicAddx2Ret(ref, val, memory_order);
  half2 prev_hi = AtomicAddx2Ret(ref + 2, val + 2, memory_order);
  uint2 ret;
  ret.x = *reinterpret_cast<const unsigned int *>(&prev_lo);
  ret.y = *reinterpret_cast<const unsigned int *>(&prev_hi);
  return ret;
}

// bf16 counterpart of the fp16 AtomicAddx4Ret above.
template <typename SrcType>
TL_DEVICE uint2 AtomicAddx4Ret(bfloat16_t *ref, SrcType *val,
                               int memory_order = 0) {
  __maca_bfloat162 prev_lo = AtomicAddx2Ret(ref, val, memory_order);
  __maca_bfloat162 prev_hi = AtomicAddx2Ret(ref + 2, val + 2, memory_order);
  uint2 ret;
  ret.x = *reinterpret_cast<const unsigned int *>(&prev_lo);
  ret.y = *reinterpret_cast<const unsigned int *>(&prev_hi);
  return ret;
}

// AtomicLoad / AtomicStore
template <typename T> TL_DEVICE T AtomicLoad(T *ref, int memory_order) {
  (void)memory_order;
  volatile T *vref = reinterpret_cast<volatile T *>(ref);
  return *vref;
}

template <typename T1, typename T2>
TL_DEVICE void AtomicStore(T1 *ref, T2 value, int memory_order) {
  (void)memory_order;
  volatile T1 *vref = reinterpret_cast<volatile T1 *>(ref);
  *vref = static_cast<T1>(value);
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
template <typename T1, typename T2>
__forceinline__ __device__ void AtomicOr(T1 *address, T2 val,
                                         int memory_order = 0) {
  atomicOr(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
// Overload for when the first argument is a value instead of a pointer.
template <typename T1, typename T2>
__forceinline__ __device__ void AtomicOr(T1 &address, T2 val,
                                         int memory_order = 0) {
  atomicOr(reinterpret_cast<T1 *>(&address), static_cast<T1>(val));
}
