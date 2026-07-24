// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights
// reserved.

#pragma once

#include "common.h"

namespace tl {
TL_DEVICE void mbarrier_init(uint64_t &smem_barrier, uint32_t arrive_count);
TL_DEVICE uint32_t mbarrier_try_wait(uint64_t &smem_barrier, int phase_bit);
TL_DEVICE void mbarrier_wait(uint64_t &smem_barrier, int phase_bit);
TL_DEVICE void mbarrier_arrive(uint64_t &smem_barrier);
TL_DEVICE void mbarrier_arrive(uint64_t &smem_barrier, int cta_id,
                               uint32_t pred);
TL_DEVICE void mbarrier_expect_tx(uint64_t &smem_barrier,
                                  uint32_t transaction_bytes);
TL_DEVICE void mbarrier_arrive_expect_tx(uint64_t &smem_barrier,
                                         uint32_t transaction_bytes);
template <typename BarrierType>
TL_DEVICE void mbarrier_cp_async_arrive(BarrierType &smem_mbar);
template <typename BarrierType>
TL_DEVICE void mbarrier_cp_async_arrive_noinc(BarrierType &smem_mbar);
TL_DEVICE void fence_proxy_async();
TL_DEVICE void fence_barrier_init();
} // namespace tl

struct Barrier {
  uint64_t value;

  TL_DEVICE void init(uint32_t arrive_count) {
    tl::mbarrier_init(value, arrive_count);
  }
  TL_DEVICE void arrive() { tl::mbarrier_arrive(value); }
  TL_DEVICE void arrive(int cta_id, uint32_t pred) {
    tl::mbarrier_arrive(value, cta_id, pred);
  }
  TL_DEVICE void wait(int phase_bit) { tl::mbarrier_wait(value, phase_bit); }
  TL_DEVICE void expect_transaction(uint32_t transaction_bytes) {
    tl::mbarrier_expect_tx(value, transaction_bytes);
  }
  TL_DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes) {
    tl::mbarrier_arrive_expect_tx(value, transaction_bytes);
  }
};

namespace tl {

TL_DEVICE void mbarrier_init(uint64_t &smem_barrier, uint32_t arrive_count) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.init.shared.b64 [%1], %0;"
               :
               : "r"(arrive_count), "r"(smem_int_ptr)
               : "memory");
}

TL_DEVICE uint32_t mbarrier_try_wait(uint64_t &smem_barrier, int phase_bit) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  uint32_t waitComplete;
  asm volatile("{\n\t"
               ".reg .pred P1; \n\t"
               "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2; \n\t"
               "selp.b32 %0, 1, 0, P1; \n\t"
               "}"
               : "=r"(waitComplete)
               : "r"(smem_int_ptr), "r"(phase_bit)
               : "memory");
  return waitComplete;
}

TL_DEVICE void mbarrier_wait(uint64_t &smem_barrier, int phase_bit) {
  if (mbarrier_try_wait(smem_barrier, phase_bit) == 0) {
    uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred       P1; \n\t"
                 "LAB_WAIT_%=: \n\t"
                 "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1, %2; \n\t"
                 "@P1 bra DONE_%=; \n\t"
                 "bra     LAB_WAIT_%=; \n\t"
                 "DONE_%=: \n\t"
                 "}"
                 :
                 : "r"(smem_int_ptr), "r"(phase_bit), "r"(ticks)
                 : "memory");
  }
}

TL_DEVICE void mbarrier_arrive(uint64_t &smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];"
               :
               : "r"(smem_int_ptr)
               : "memory");
}

TL_DEVICE void mbarrier_arrive(uint64_t &smem_barrier, int cta_id,
                               uint32_t pred) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  if (pred) {
    asm volatile("{\n\t"
                 ".reg .b32 remAddr32;\n\t"
                 "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
                 "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
                 "}"
                 :
                 : "r"(smem_int_ptr), "r"(cta_id)
                 : "memory");
  }
}

TL_DEVICE void mbarrier_expect_tx(uint64_t &smem_barrier,
                                  uint32_t transaction_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.expect_tx.shared.b64 [%1], %0;"
               :
               : "r"(transaction_bytes), "r"(smem_int_ptr)
               : "memory");
}

TL_DEVICE void mbarrier_arrive_expect_tx(uint64_t &smem_barrier,
                                         uint32_t transaction_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0;"
               :
               : "r"(transaction_bytes), "r"(smem_int_ptr)
               : "memory");
}

template <typename BarrierType>
TL_DEVICE void mbarrier_cp_async_arrive(BarrierType &smem_mbar) {
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];"
               :
               : "r"(smem_int_mbar)
               : "memory");
}

template <typename BarrierType>
TL_DEVICE void mbarrier_cp_async_arrive_noinc(BarrierType &smem_mbar) {
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  asm volatile("{\n\t"
               "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
               "}"
               :
               : "r"(smem_int_mbar)
               : "memory");
}

TL_DEVICE void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
}

TL_DEVICE void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
}

} // namespace tl
