// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights
// reserved.
#pragma once
#include "barrier.h"
#include "common.h"
#include <mctlass/arch/maca_memory.h>
namespace tl {
TL_DEVICE void cp_async_commit() { asm volatile("" ::: "memory"); }

template <int N = 0> TL_DEVICE void cp_async_wait() {
  mctlass::arch::maca_cp_async_wait<N>();
  __syncthreadshared();
}

template <int N>
TL_DEVICE void cp_async_gs(void *lds_base_ptr, void const *global_base_ptr) {
  static_assert(N == 16 || N == 8 || N == 4);
  mctlass::arch::maca_cp_async_zfill<N>(lds_base_ptr, global_base_ptr);
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void *lds_base_ptr,
                                       void const *global_base_ptr, bool cond) {
  static_assert(N == 16 || N == 8 || N == 4);
  mctlass::arch::maca_cp_async_zfill<N>(lds_base_ptr, global_base_ptr, cond);
}

// Global memory load intrinsics with explicit vector widths
// MACA-compatible implementation using standard pointer casts
// load_global_32: Load 32 bits, return uint32_t
TL_DEVICE uint32_t load_global_32(const void *ptr) {
  return *reinterpret_cast<const uint32_t *>(ptr);
}
// load_global_64: Load 64 bits, return uint2
TL_DEVICE uint2 load_global_64(const void *ptr) {
  return *reinterpret_cast<const uint2 *>(ptr);
}
// load_global_128: Load 128 bits, return uint4
TL_DEVICE uint4 load_global_128(const void *ptr) {
  return *reinterpret_cast<const uint4 *>(ptr);
}
// load_global_256: Load 256 bits, return ulonglong4
TL_DEVICE ulonglong4 load_global_256(const void *ptr) {
  return *reinterpret_cast<const ulonglong4 *>(ptr);
}
// Predicated (conditional) versions
TL_DEVICE uint32_t load_global_32_conditional(const void *ptr, bool pred) {
  if (pred) {
    return *reinterpret_cast<const uint32_t *>(ptr);
  }
  return 0u;
}
TL_DEVICE uint2 load_global_64_conditional(const void *ptr, bool pred) {
  if (pred) {
    return *reinterpret_cast<const uint2 *>(ptr);
  }
  return make_uint2(0u, 0u);
}
TL_DEVICE uint4 load_global_128_conditional(const void *ptr, bool pred) {
  if (pred) {
    return *reinterpret_cast<const uint4 *>(ptr);
  }
  return make_uint4(0u, 0u, 0u, 0u);
}
TL_DEVICE ulonglong4 load_global_256_conditional(const void *ptr, bool pred) {
  if (pred) {
    return *reinterpret_cast<const ulonglong4 *>(ptr);
  }
  ulonglong4 zero;
  zero.x = 0;
  zero.y = 0;
  zero.z = 0;
  zero.w = 0;
  return zero;
}

TL_DEVICE uint32_t load_shared_32(const void *ptr) {
  return *reinterpret_cast<const uint32_t *>(ptr);
}

TL_DEVICE uint2 load_shared_64(const void *ptr) {
  return *reinterpret_cast<const uint2 *>(ptr);
}

TL_DEVICE uint4 load_shared_128(const void *ptr) {
  return *reinterpret_cast<const uint4 *>(ptr);
}

TL_DEVICE void store_shared_32(void *ptr, uint32_t value) {
  *reinterpret_cast<uint32_t *>(ptr) = value;
}

TL_DEVICE void store_shared_64(void *ptr, uint2 value) {
  *reinterpret_cast<uint2 *>(ptr) = value;
}

TL_DEVICE void store_shared_128(void *ptr, uint4 value) {
  *reinterpret_cast<uint4 *>(ptr) = value;
}

// Global memory store intrinsics with explicit vector widths
// store_global_32: Store 32 bits
TL_DEVICE void store_global_32(void *ptr, uint32_t value) {
  *reinterpret_cast<uint32_t *>(ptr) = value;
}
// store_global_64: Store 64 bits
TL_DEVICE void store_global_64(void *ptr, uint2 value) {
  *reinterpret_cast<uint2 *>(ptr) = value;
}
// store_global_128: Store 128 bits
TL_DEVICE void store_global_128(void *ptr, uint4 value) {
  *reinterpret_cast<uint4 *>(ptr) = value;
}
// store_global_256: Store 256 bits
TL_DEVICE void store_global_256(void *ptr, ulonglong4 value) {
  *reinterpret_cast<ulonglong4 *>(ptr) = value;
}
// Predicated (conditional) versions
TL_DEVICE void store_global_32_conditional(void *ptr, uint32_t value,
                                           bool pred) {
  if (pred) {
    *reinterpret_cast<uint32_t *>(ptr) = value;
  }
}
TL_DEVICE void store_global_64_conditional(void *ptr, uint2 value, bool pred) {
  if (pred) {
    *reinterpret_cast<uint2 *>(ptr) = value;
  }
}
TL_DEVICE void store_global_128_conditional(void *ptr, uint4 value, bool pred) {
  if (pred) {
    *reinterpret_cast<uint4 *>(ptr) = value;
  }
}
TL_DEVICE void store_global_256_conditional(void *ptr, ulonglong4 value,
                                            bool pred) {
  if (pred) {
    *reinterpret_cast<ulonglong4 *>(ptr) = value;
  }
}
} // namespace tl
