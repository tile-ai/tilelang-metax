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
    return max(x, y);
  }
};

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return min(x, y);
  }
};

struct MaxOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return max(x, y);
  }
  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x,
                                  bfloat16_t const &y) const {
    return __hmax_nan(x, y);
  }
  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) const {
    return half_t(__hmax_nan(x, y));
  }
};

struct MinOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return min(x, y);
  }
  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x,
                                  bfloat16_t const &y) const {
    return __hmin_nan(x, y);
  }
  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) const {
    return half_t(__hmin_nan(x, y));
  }
};

// Packed x2 reduce operators for bf16x2 and fp16x2
// These operate on uint1 (packed 32-bit) values

struct SumOp_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(
        tl::add2(tl::from_uint1<bfloat16x2>(x), tl::from_uint1<bfloat16x2>(y)));
  }
};

struct MaxOp_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(
        tl::max2(tl::from_uint1<bfloat16x2>(x), tl::from_uint1<bfloat16x2>(y)));
  }
};

struct MinOp_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(
        tl::min2(tl::from_uint1<bfloat16x2>(x), tl::from_uint1<bfloat16x2>(y)));
  }
};

struct SumOp_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(
        tl::add2(tl::from_uint1<float16x2>(x), tl::from_uint1<float16x2>(y)));
  }
};

struct MaxOp_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(
        tl::max2(tl::from_uint1<float16x2>(x), tl::from_uint1<float16x2>(y)));
  }
};

struct MinOp_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) const {
    return tl::to_uint1(
        tl::min2(tl::from_uint1<float16x2>(x), tl::from_uint1<float16x2>(y)));
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
  static_assert(threads % scale == 0);

  // Scalar interface (backward-compatible).
  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return x;
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
    } else {
      butterfly_reduce_batch(x, red_buf);
    }
  }

private:
  using Next = AllReduce<Reducer, threads / 2, scale, thread_offset, Barrier,
                         batch_size, workspace_stride>;

  template <typename T>
  static TL_DEVICE T butterfly_reduce_scalar(T x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 64) {
      Barrier::template sync<1>();
      red_buf[threadIdx.x - thread_offset] = x;
      Barrier::template sync<2>();
      x = Reducer()(x, red_buf[(threadIdx.x - thread_offset) ^ offset]);
    } else {
      x = Reducer()(x, tl::shfl_xor_sync(uint64_t(-1), x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return Next::run(x, red_buf);
    }
  }

  template <typename T>
  static TL_DEVICE void butterfly_reduce_batch(T *x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 64) {
      Barrier::template sync<1>();
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        red_buf[(threadIdx.x - thread_offset) + i * workspace_stride] = x[i];
      }
      Barrier::template sync<2>();
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] =
            Reducer()(x[i], red_buf[((threadIdx.x - thread_offset) ^ offset) +
                                    i * workspace_stride]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] = Reducer()(x[i], tl::shfl_xor_sync(uint64_t(-1), x[i], offset));
      }
    }
    if constexpr (offset == scale) {
      return;
    } else {
      Next::run_batch(x, red_buf);
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

template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64);
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W) {

    constexpr int TILE_H = threads / SEG;
    constexpr unsigned long MASK = 0xFFFFFFFFFFFFFFFF;
    const int num_blocks = (H + TILE_H - 1) / TILE_H;
    const int tid = threadIdx.x;
    const int lane = tid % SEG;
    const int row = tid / SEG;

    for (int b = 0; b < num_blocks; ++b) {
      const int gRow = b * TILE_H + row;
      if (gRow >= H)
        continue;

      T carry = (T)0;

      if (reverse) {
        // Start from the last segment for reverse mode
        for (int seg = (W + SEG - 1) / SEG - 1; seg >= 0; --seg) {
          const int col = seg * SEG + lane;

          const int real_row = (Axis == 1) ? gRow : col;
          const int real_col = (Axis == 1) ? col : gRow;

          T val = (col < W && real_row < H && real_col < W)
                      ? src[real_row * W + real_col]
                      : (T)0;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = __shfl_down_sync(MASK, val, off);
            if (lane < SEG - off)
              val += n;
          }

          val += carry;

          if (real_col < W && real_row < H)
            dst[real_row * W + real_col] = val;

          T segSum = __shfl_sync(MASK, val, 0);
          if (lane == 0)
            carry = segSum;
          carry = __shfl_sync(MASK, carry, 0);
        }
      } else {
        for (int seg = 0; seg * SEG < W; ++seg) {
          const int col = seg * SEG + lane;

          const int real_row = (Axis == 1) ? gRow : col;
          const int real_col = (Axis == 1) ? col : gRow;

          T val = (col < W && real_row < H && real_col < W)
                      ? src[real_row * W + real_col]
                      : (T)0;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = __shfl_up_sync(MASK, val, off);
            if (lane >= off)
              val += n;
          }

          val += carry;

          if (real_col < W && real_row < H)
            dst[real_row * W + real_col] = val;

          T segSum = __shfl_sync(MASK, val, SEG - 1);
          if (lane == SEG - 1)
            carry = segSum;
          carry = __shfl_sync(MASK, carry, SEG - 1);
        }
      }
    }
  }
};

template <int threads, bool reverse = false> struct CumSum1D {
  template <typename T>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    CumSum2D<threads, 1, reverse>::run(src, dst, 1, N);
  }
};
} // namespace tl
