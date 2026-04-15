#pragma once

#include "common.h"

namespace tl {

TL_DEVICE void cp_async_commit() {}

template <int N>
TL_DEVICE auto cp_async_gs(void *dst_shared, void const *src_global) {
  static_assert(N == 4 || N == 8 || N == 16);
  return memcpy_async<N>(dst_shared, const_cast<void *>(src_global));
}

template <int N>
TL_DEVICE auto cp_async_gs_conditional(void *dst_shared, void const *src_global,
                                       bool cond) {
  static_assert(N == 4 || N == 8 || N == 16);
  return memcpy_async_pred<N, MACA_ICMP_NE, true>(
      dst_shared, const_cast<void *>(src_global), static_cast<int>(cond), 0);
}

template <typename Token> TL_DEVICE void cp_async_wait_token(Token token) {
  barrier_arrive_and_wait(token);
}

} // namespace tl
