#pragma once

#include "gemm.h"

namespace tl {

using WsmFloat4 = __NATIVE_VECTOR__(4, float);
using WsmUint2 = __NATIVE_VECTOR__(2, uint);
using WsmUint4 = __NATIVE_VECTOR__(4, uint);

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int kPack, int AStrideElements,
          typename A_type, typename B_type, typename C_type, typename WSM_type,
          typename A_source_type, typename B_source_type>
MCTLASS_DEVICE void gemm_ss_wsm(A_type *pA, B_type *pB, C_type *accum,
                                WSM_type *wsm, A_source_type *global_A,
                                B_source_type *global_B) {
  constexpr int Stage = 4;
  static_assert(!trans_A && !trans_B,
                "gemm_ss_wsm only supports non-transposed A/B operands");
  static_assert(num_warp_m == 1 && num_warp_n == 1,
                "gemm_ss_wsm currently uses a fixed single-warp layout");
  static_assert(kPack == 8,
                "gemm_ss_wsm expects kPack=8 for its K/8 BSM predicate");
  static_assert(AStrideElements % 8 == 0,
                "gemm_ss_wsm requires AStrideElements divisible by 8");
  static_assert(sizeof(A_type) == 2 && sizeof(B_type) == 2,
                "gemm_ss_wsm expects fp16 A/B operands");
  static_assert(sizeof(C_type) == 4,
                "gemm_ss_wsm expects fp32 accumulator fragments");
  static_assert(Stage == 4, "gemm_ss_wsm hardcodes a 4-stage schedule");
  uchar *WSM = reinterpret_cast<uchar *>(wsm);
  uchar *APtr = const_cast<uchar *>(reinterpret_cast<const uchar *>(global_A));
  uchar *BPtr = const_cast<uchar *>(reinterpret_cast<const uchar *>(global_B));
  const int tid = threadIdx.x;
  const int slot = __builtin_mxc_readfirstlane(tid / 64);
  const int lane = tid & 63;
  uchar *WSM_Ldg = WSM + slot * 0x400;
  uchar *WSM_lds = WSM;
  int ALdsOffset[4];
  int BLdsOffset[4];
  int ALdgOffset[2][Stage];
  int BLdgOffset[2][Stage];
  const int lda_vec = AStrideElements / 8;
  const int B_row_offset = lane + slot * 64 * (N / 16);
  const int lds_row = tid & 15;
  int lds_col[4];

#pragma unroll
  for (int stage_i = 0; stage_i < Stage; ++stage_i) {
    ALdgOffset[0][stage_i] = (tid + 16 * lda_vec * stage_i) * 16;
    ALdgOffset[1][stage_i] = (tid + 16 * lda_vec * (4 + stage_i)) * 16;
    BLdgOffset[0][stage_i] = (B_row_offset + 64 * stage_i) * 16;
    BLdgOffset[1][stage_i] = (B_row_offset + 64 * (4 + stage_i)) * 16;
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lds_col[i] = (4 * i + lane / 16) ^ lds_row;
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int tmp = lds_row * 16 + lds_col[i];
    ALdsOffset[i] = (tmp + (slot / 2) * 0x1000 / 16) * 16;
    BLdsOffset[i] = (tmp + 0x2000 / 16 + (slot & 1) * 0x1000 / 16) * 16;
  }

#pragma unroll
  for (int stage_i = 0; stage_i < Stage; ++stage_i) {
    __builtin_mxc_ldg_b128_bsm_predicator(
        WSM_Ldg + 0x4000 * stage_i + 0x0000, APtr + ALdgOffset[0][stage_i], 0,
        true, true, false, true, 0, K / 8, MACA_ICMP_SLT);
    __builtin_mxc_ldg_b128_bsm_predicator(
        WSM_Ldg + 0x4000 * stage_i + 0x1000, APtr + ALdgOffset[1][stage_i], 0,
        true, true, false, true, 0, K / 8, MACA_ICMP_SLT);
    __builtin_mxc_ldg_b128_bsm_predicator(
        WSM_Ldg + 0x4000 * stage_i + 0x2000, BPtr + BLdgOffset[0][stage_i], 0,
        true, true, false, true, stage_i * 16, N, MACA_ICMP_SLT);
    __builtin_mxc_ldg_b128_bsm_predicator(
        WSM_Ldg + 0x4000 * stage_i + 0x3000, BPtr + BLdgOffset[1][stage_i], 0,
        true, true, false, true, stage_i * 16 + 64, N, MACA_ICMP_SLT);
  }

  __builtin_mxc_arrive_gvmcnt(4 * (Stage - 1));
  __builtin_mxc_barrier_inst();

  WsmFloat4 C_f32[4][4];
#pragma unroll
  for (int row = 0; row < 4; ++row) {
#pragma unroll
    for (int col = 0; col < 4; ++col) {
      if constexpr (clear_accum) {
        C_f32[row][col] = WsmFloat4{0.0f, 0.0f, 0.0f, 0.0f};
      } else {
        C_f32[row][col] = reinterpret_cast<WsmFloat4 *>(accum)[row * 4 + col];
      }
    }
  }

#pragma unroll
  for (int stage_i = 0; stage_i < Stage; ++stage_i) {
    WsmUint4 a_frag0 = *reinterpret_cast<WsmUint4 *>(
        WSM_lds + 0x4000 * stage_i + ALdsOffset[0]);
    WsmUint4 a_frag1 = *reinterpret_cast<WsmUint4 *>(
        WSM_lds + 0x4000 * stage_i + ALdsOffset[1]);
    WsmUint4 a_frag2 = *reinterpret_cast<WsmUint4 *>(
        WSM_lds + 0x4000 * stage_i + ALdsOffset[2]);
    WsmUint4 a_frag3 = *reinterpret_cast<WsmUint4 *>(
        WSM_lds + 0x4000 * stage_i + ALdsOffset[3]);
    WsmUint4 b_frag0 = *reinterpret_cast<WsmUint4 *>(
        WSM_lds + 0x4000 * stage_i + BLdsOffset[0]);
    WsmUint4 b_frag1 = *reinterpret_cast<WsmUint4 *>(
        WSM_lds + 0x4000 * stage_i + BLdsOffset[1]);
    WsmUint4 b_frag2 = *reinterpret_cast<WsmUint4 *>(
        WSM_lds + 0x4000 * stage_i + BLdsOffset[2]);
    WsmUint4 b_frag3 = *reinterpret_cast<WsmUint4 *>(
        WSM_lds + 0x4000 * stage_i + BLdsOffset[3]);
    WsmUint2 mma_a[4] = {{a_frag0[0], a_frag0[1]},
                         {a_frag1[0], a_frag1[1]},
                         {a_frag2[0], a_frag2[1]},
                         {a_frag3[0], a_frag3[1]}};
    WsmUint2 mma_b[4] = {{b_frag0[0], b_frag0[1]},
                         {b_frag1[0], b_frag1[1]},
                         {b_frag2[0], b_frag2[1]},
                         {b_frag3[0], b_frag3[1]}};

#pragma unroll
    for (int row = 0; row < 4; ++row) {
#pragma unroll
      for (int col = 0; col < 4; ++col) {
        C_f32[row][col] = __builtin_mxc_mma_16x16x16f16(mma_a[row], mma_b[col],
                                                        C_f32[row][col]);
      }
    }
  }

  reinterpret_cast<WsmFloat4 *>(accum)[0] = C_f32[0][0];
  reinterpret_cast<WsmFloat4 *>(accum)[1] = C_f32[0][1];
  reinterpret_cast<WsmFloat4 *>(accum)[2] = C_f32[0][2];
  reinterpret_cast<WsmFloat4 *>(accum)[3] = C_f32[0][3];
  reinterpret_cast<WsmFloat4 *>(accum)[4] = C_f32[1][0];
  reinterpret_cast<WsmFloat4 *>(accum)[5] = C_f32[1][1];
  reinterpret_cast<WsmFloat4 *>(accum)[6] = C_f32[1][2];
  reinterpret_cast<WsmFloat4 *>(accum)[7] = C_f32[1][3];
  reinterpret_cast<WsmFloat4 *>(accum)[8] = C_f32[2][0];
  reinterpret_cast<WsmFloat4 *>(accum)[9] = C_f32[2][1];
  reinterpret_cast<WsmFloat4 *>(accum)[10] = C_f32[2][2];
  reinterpret_cast<WsmFloat4 *>(accum)[11] = C_f32[2][3];
  reinterpret_cast<WsmFloat4 *>(accum)[12] = C_f32[3][0];
  reinterpret_cast<WsmFloat4 *>(accum)[13] = C_f32[3][1];
  reinterpret_cast<WsmFloat4 *>(accum)[14] = C_f32[3][2];
  reinterpret_cast<WsmFloat4 *>(accum)[15] = C_f32[3][3];
  (void)pA;
  (void)pB;
}

} // namespace tl
