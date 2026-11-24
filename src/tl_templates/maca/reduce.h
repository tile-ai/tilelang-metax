// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights
// reserved.

#pragma once

#include "common.h"

namespace tl {

template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op) {
  constexpr uint64_t MASK = uint64_t(-1);

  value = op(value, tl::shfl_xor_sync(MASK, value, 32));
  value = op(value, tl::shfl_xor_sync(MASK, value, 16));
  value = op(value, tl::shfl_xor_sync(MASK, value, 8));
  value = op(value, tl::shfl_xor_sync(MASK, value, 4));
  value = op(value, tl::shfl_xor_sync(MASK, value, 2));
  value = op(value, tl::shfl_xor_sync(MASK, value, 1));

  return value;
}
struct SumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }
};

struct MaxOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::fast_max(x, y);
  }
  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x, bfloat16_t const &y) {
    return __hmax(x, y);
  }
  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmax(x, y));
  }
};

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::fast_min(x, y);
  }
  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x, bfloat16_t const &y) {
    return __hmin(x, y);
  }
  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmin(x, y));
  }
};

struct MaxOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::fast_max(x, y);
  }
  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x, bfloat16_t const &y) {
    return __hmax_nan(x, y);
  }
  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmax_nan(x, y));
  }
};

struct MinOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::fast_min(x, y);
  }
  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x, bfloat16_t const &y) {
    return __hmin_nan(x, y);
  }
  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmin_nan(x, y));
  }
};

// Packed x2 reduce operators for bf16x2 and fp16x2
// These operate on uint1 (packed 32-bit) values

struct SumOp_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(tl::add2(tl::from_uint1<maca_bfloat162>(x),
                                 tl::from_uint1<maca_bfloat162>(y)));
  }
};

struct MaxOp_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(tl::max2(tl::from_uint1<maca_bfloat162>(x),
                                 tl::from_uint1<maca_bfloat162>(y)));
  }
};

struct MinOp_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(tl::min2(tl::from_uint1<maca_bfloat162>(x),
                                 tl::from_uint1<maca_bfloat162>(y)));
  }
};

struct SumOp_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(
        tl::add2(tl::from_uint1<half2>(x), tl::from_uint1<half2>(y)));
  }
};

struct MaxOp_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(
        tl::max2(tl::from_uint1<half2>(x), tl::from_uint1<half2>(y)));
  }
};

struct MinOp_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(
        tl::min2(tl::from_uint1<half2>(x), tl::from_uint1<half2>(y)));
  }
};

struct MaxOpNan_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(tl::max2_nan(tl::from_uint1<bfloat16x2>(x),
                                     tl::from_uint1<bfloat16x2>(y)));
  }
};

struct MinOpNan_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(tl::min2_nan(tl::from_uint1<bfloat16x2>(x),
                                     tl::from_uint1<bfloat16x2>(y)));
  }
};

struct MaxOpNan_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(tl::max2_nan(tl::from_uint1<float16x2>(x),
                                     tl::from_uint1<float16x2>(y)));
  }
};

struct MinOpNan_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(tl::min2_nan(tl::from_uint1<float16x2>(x),
                                     tl::from_uint1<float16x2>(y)));
  }
};

struct SumOp_f32x2 {
  TL_DEVICE float2 operator()(float2 const &x, float2 const &y) {
    return tl::add2(x, y);
  }
};

struct MaxOp_f32x2 {
  TL_DEVICE float2 operator()(float2 const &x, float2 const &y) {
    return tl::max2(x, y);
  }
};

struct MinOp_f32x2 {
  TL_DEVICE float2 operator()(float2 const &x, float2 const &y) {
    return tl::min2(x, y);
  }
};

struct BitAndOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return x & y;
  }
};

struct BitOrOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return x | y;
  }
};

struct BitXorOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return x ^ y;
  }
};

// Barrier policy: warps __syncthreads().
// The phase template parameter is ignored (all phases use the same barrier).
struct SyncThreadsBarrier {
  template <int phase = 0> static TL_DEVICE void sync() { __syncthreads(); }
};

// AllReduce performs a cross-thread reduction over a group of `threads`
// threads.
//
// Template parameters:
//   Reducer         - binary reduction functor (e.g. SumOp, MaxOp).
//   threads         - number of threads that span the reduce dimension,
//                     equal to extent * scale.
//   scale           - stride of participating threads in the thread index
//                     space. When the thread-to-data mapping is normalized as
//                       threadIdx = source * scale + ...
//                     `scale` is the stride between consecutive logical
//                     participants in the reduce dimension.
//                     The recursion terminates when threads == scale, meaning
//                     each reduce group has been collapsed to a single thread.
//                     Uses a recursive XOR-butterfly pattern: at each level,
//                     offset >= 64 goes through shared memory + barrier,
//                     offset < 64 uses warp shuffle (shfl_xor_sync).
//   thread_offset   - base thread index offset within the block.
//   Barrier         - barrier policy type (SyncThreadsBarrier).
//   batch_size      - number of independent values to reduce in parallel,
//                     sharing synchronization barriers across all values.
//                     Default 1 preserves the original scalar behaviour.
//   workspace_stride - stride between per-channel slices in the shared-memory
//                     workspace (typically total threads in the block).
//                     Only used when batch_size > 1.
template <class Reducer, int threads, int scale, int thread_offset = 0,
          class Barrier = SyncThreadsBarrier, int batch_size = 1,
          int workspace_stride = 0>
struct AllReduce {
  static_assert(threads > 0, "tl::AllReduce threads must be positive");
  static_assert(scale > 0, "tl::AllReduce scale must be positive");
  static_assert(threads % scale == 0,
                "tl::AllReduce threads must be divisible by scale");
  static_assert(((threads / scale) & (threads / scale - 1)) == 0,
                "AllReduce reduce width (threads / scale) must be a power of "
                "two");
  // t ^ (scale * k) only flips the reduce coordinate when the non-reduced
  // part of the thread index sits below `scale`, which needs scale to be a
  // power of two; otherwise the pairs straddle reduce groups.
  static_assert((scale & (scale - 1)) == 0,
                "AllReduce scale (thread stride between reduce participants) "
                "must be a power of two");

  // Scalar interface (backward-compatible).
  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return x;
    } else if (full_warp) {
      return warp_reduce(x, Reducer());
    } else {
      return butterfly_reduce_scalar(x, red_buf);
    }
  }

  // Batch interface (named run_batch to avoid overload-resolution ambiguity
  // with the scalar run(T x, T*) when a pointer is passed as the first arg).
  template <typename T>
  static TL_DEVICE void run_batch(T *x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return;
    } else if (full_warp) {
#pragma unroll
      for (int index = 0; index < batch_size; ++index) {
        x[index] = warp_reduce(x[index], Reducer());
      }
    } else {
      butterfly_reduce_batch(x, red_buf);
    }
  }

private:
  static constexpr bool full_warp =
      threads == 64 && scale == 1 && thread_offset % 64 == 0;
  static constexpr bool is_idempotent =
      std::is_same_v<Reducer, MaxOp> || std::is_same_v<Reducer, MinOp> ||
      std::is_same_v<Reducer, BitAndOp> || std::is_same_v<Reducer, BitOrOp>;
  static constexpr bool has_zero_identity =
      std::is_same_v<Reducer, SumOp> || std::is_same_v<Reducer, BitXorOp>;
  static constexpr int in_warp_scale = scale < 64 ? scale : 64;
  static constexpr int cross_scale = scale < 64 ? 64 : scale;
  static constexpr int cross_radix =
      threads > cross_scale ? (threads / cross_scale) : 1;
  static constexpr int in_warp_lanes = 64 / in_warp_scale;
  static constexpr int merge_lanes =
      cross_radix < in_warp_lanes ? cross_radix : in_warp_lanes;
  static constexpr int items_per_lane =
      cross_radix > merge_lanes ? (cross_radix / merge_lanes) : 1;

  using InWarp = AllReduce<Reducer, 64, in_warp_scale, thread_offset, Barrier,
                           batch_size, workspace_stride>;
  using CrossWarpInLane =
      AllReduce<Reducer, merge_lanes * in_warp_scale, in_warp_scale,
                thread_offset, Barrier, batch_size, workspace_stride>;
  using Next = AllReduce<Reducer, threads / 2, scale, thread_offset, Barrier,
                         batch_size, workspace_stride>;

  template <typename T>
  static TL_DEVICE T butterfly_reduce_scalar(T x, T *red_buf) {
    if constexpr (threads > 64) {
      const int rel_tid = threadIdx.x - thread_offset;
      Barrier::template sync<1>();
      if constexpr (scale < 64) {
        x = InWarp::run(x);
        red_buf[rel_tid] = x;
        Barrier::template sync<2>();
        const int lane_in_group = (rel_tid & 63) / in_warp_scale;
        const int warp_base =
            (lane_in_group & (merge_lanes - 1)) * items_per_lane;
        const int group_base = rel_tid & ~(threads - cross_scale);
        const int smem_base = group_base + warp_base * cross_scale;
        T acc = red_buf[smem_base];
#pragma unroll 4
        for (int k = 1; k < items_per_lane; ++k) {
          acc = Reducer()(acc, red_buf[smem_base + k * cross_scale]);
        }
        if constexpr (scale == 1 && thread_offset % 64 == 0 &&
                      sizeof(T) <= sizeof(float) &&
                      (is_idempotent ||
                       (has_zero_identity && std::is_integral_v<T>))) {
          if constexpr (!is_idempotent && merge_lanes < 64) {
            if (lane_in_group >= merge_lanes) {
              acc = static_cast<T>(0);
            }
          }
          return InWarp::run(acc);
        } else {
          return CrossWarpInLane::run(acc);
        }
      } else {
        red_buf[rel_tid] = x;
        Barrier::template sync<2>();
        const int base = rel_tid & ~(threads - cross_scale);
        T acc = red_buf[base];
#pragma unroll 4
        for (int k = 1; k < cross_radix; ++k) {
          acc = Reducer()(acc, red_buf[base + k * cross_scale]);
        }
        return acc;
      }
    } else {
      constexpr int offset = threads / 2;
      x = Reducer()(x, tl::shfl_xor_sync(uint64_t(-1), x, offset));
      if constexpr (offset == scale) {
        return x;
      } else {
        return Next::run(x, red_buf);
      }
    }
  }

  template <typename T>
  static TL_DEVICE void butterfly_reduce_batch(T *x, T *red_buf) {
    if constexpr (threads > 64) {
      const int rel_tid = threadIdx.x - thread_offset;
      Barrier::template sync<1>();
      if constexpr (scale < 64) {
        InWarp::run_batch(x);
#pragma unroll
        for (int i = 0; i < batch_size; i++) {
          red_buf[rel_tid + i * workspace_stride] = x[i];
        }
        Barrier::template sync<2>();
        const int lane_in_group = (rel_tid & 63) / in_warp_scale;
        const int warp_base =
            (lane_in_group & (merge_lanes - 1)) * items_per_lane;
        const int group_base = rel_tid & ~(threads - cross_scale);
        const int smem_base = group_base + warp_base * cross_scale;
#pragma unroll
        for (int i = 0; i < batch_size; i++) {
          T acc = red_buf[smem_base + i * workspace_stride];
#pragma unroll 4
          for (int k = 1; k < items_per_lane; ++k) {
            acc = Reducer()(
                acc,
                red_buf[smem_base + k * cross_scale + i * workspace_stride]);
          }
          if constexpr (scale == 1 && thread_offset % 64 == 0 &&
                        sizeof(T) <= sizeof(float) && !is_idempotent &&
                        has_zero_identity && std::is_integral_v<T> &&
                        merge_lanes < 64) {
            if (lane_in_group >= merge_lanes) {
              acc = static_cast<T>(0);
            }
          }
          x[i] = acc;
        }
        if constexpr (scale == 1 && thread_offset % 64 == 0 &&
                      sizeof(T) <= sizeof(float) &&
                      (is_idempotent ||
                       (has_zero_identity && std::is_integral_v<T>))) {
          InWarp::run_batch(x);
        } else {
          CrossWarpInLane::run_batch(x);
        }
      } else {
#pragma unroll
        for (int i = 0; i < batch_size; i++) {
          red_buf[rel_tid + i * workspace_stride] = x[i];
        }
        Barrier::template sync<2>();
        const int base = rel_tid & ~(threads - cross_scale);
#pragma unroll
        for (int i = 0; i < batch_size; i++) {
          T acc = red_buf[base + i * workspace_stride];
#pragma unroll 4
          for (int k = 1; k < cross_radix; ++k) {
            acc = Reducer()(
                acc, red_buf[base + k * cross_scale + i * workspace_stride]);
          }
          x[i] = acc;
        }
      }
    } else {
      constexpr int offset = threads / 2;
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] = Reducer()(x[i], tl::shfl_xor_sync(uint64_t(-1), x[i], offset));
      }
      if constexpr (offset == scale) {
        return;
      } else {
        Next::run_batch(x, red_buf);
      }
    }
  }
};

template <typename T> TL_DEVICE T warp_reduce_sum(T value) {
  return warp_reduce<T>(value, SumOp());
}

template <typename T> TL_DEVICE T warp_reduce_max(T value) {
  return warp_reduce<T>(value, MaxOp());
}

template <typename T> TL_DEVICE T warp_reduce_min(T value) {
  return warp_reduce<T>(value, MinOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitand(T value) {
  return warp_reduce<T>(value, BitAndOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitor(T value) {
  return warp_reduce<T>(value, BitOrOp());
}

} // namespace tl
