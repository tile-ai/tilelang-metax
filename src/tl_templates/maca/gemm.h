// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights
// reserved.

#pragma once

#include "common.h"
#include <cute/tensor.hpp>
#include <type_traits>

namespace cute {

template <typename A_type, typename B_type, typename C_type>
struct DispatchInstruction;

template <> struct DispatchInstruction<half_t, half_t, float> {
  using MMA = MMA_Atom<MMA_Traits<MACA_16x16x16_F32F16F16F32>>;
};

template <int Bits, int N, int K, bool K_inner, int num_warp_n,
          typename Enable = void>
struct OperandTraits;

template <int N, int K, int num_warp_n>
struct OperandTraits<16, N, K, true, num_warp_n,
                     typename std::enable_if<K % 64 == 32>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = Copy_Traits<UniversalCopy<cute::uint64_t>>;
};

template <int N, int K, int num_warp_n>
struct OperandTraits<16, N, K, true, num_warp_n,
                     typename std::enable_if<K % 64 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<4, 2, 4>{}, Layout<Shape<_16, _64>, Stride<_64, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = Copy_Traits<UniversalCopy<cute::uint64_t>>;
};

template <int N, int K, int num_warp_n>
struct OperandTraits<16, N, K, false, num_warp_n,
                     typename std::enable_if<K % 64 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<4, 2, 4>{}, Layout<Shape<_64, _16>, Stride<_1, _64>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = Copy_Traits<UniversalCopy<cute::uint16_t>>;
};

template <int N, int K, int num_warp_n>
struct OperandTraits<16, N, K, false, num_warp_n,
                     typename std::enable_if<K % 64 == 32>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                        Step<_2, _1>{}));
  using Copy = Copy_Traits<UniversalCopy<cute::uint16_t>>;
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
public:
  using A_type = A_type_raw;
  using B_type = B_type_raw;
  using C_type = C_type_raw;

  using Instruction = DispatchInstruction<A_type, B_type, C_type>;

  using OperandATraits =
      OperandTraits<sizeof_bits<A_type>::value, M, K, !trans_A, num_warp_m>;
  using OperandBTraits =
      OperandTraits<sizeof_bits<B_type>::value, N, K, trans_B, num_warp_n>;

  using SmemLayoutA = typename OperandATraits::Layout;
  using SmemLayoutB = typename OperandBTraits::Layout;
  using SmemCopyA = Copy_Atom<typename OperandATraits::Copy, A_type>;
  using SmemCopyB = Copy_Atom<typename OperandBTraits::Copy, B_type>;

  using TileMma = TiledMMA<typename Instruction::MMA,
                           Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>,
                           Layout<Shape<_1, _1, _1>>>;

  CUTE_DEVICE static void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                            SmemLayoutB{});

    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsA = thr_copy_A.partition_S(sA);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    if constexpr (clear_accum) {
      clear(acc);
    }

    for (int k = 0; k < size<2>(tCrA); ++k) {
      copy(tiled_copy_A, tCsA(_, _, k), tCrA_copy_view(_, _, k));
      copy(tiled_copy_B, tCsB(_, _, k), tCrB_copy_view(_, _, k));
      gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), acc);
    }
  }

  CUTE_DEVICE static void body_rs(A_type_raw *pA, B_type_raw *pB,
                                  C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                            SmemLayoutB{});

    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrA =
        make_tensor(make_rmem_ptr(reinterpret_cast<A_type *>(pA)),
                    partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));

    if constexpr (clear_accum) {
      clear(acc);
    }

    for (int k = 0; k < size<2>(tCrA); ++k) {
      copy(tiled_copy_B, tCsB(_, _, k), tCrB_copy_view(_, _, k));
      gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), acc);
    }
  }
};

} // namespace cute

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int kPack, typename A_type,
          typename B_type, typename C_type>
MCTLASS_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {
  using MMA = cute::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                 trans_B, clear_accum, A_type, B_type, C_type>;
  MMA::body(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int kPack, typename A_type,
          typename B_type, typename C_type>
TL_DEVICE void gemm_rs(A_type *pA, B_type *pB, C_type *accum) {
  using MMA = cute::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                 trans_B, clear_accum, A_type, B_type, C_type>;
  MMA::body_rs(pA, pB, accum);
}
} // namespace tl
