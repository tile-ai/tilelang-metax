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
    return mctlass::fast_max(x, y);
  }
};

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return mctlass::fast_min(x, y);
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

template <class Reducer, int threads, int scale, int thread_offset = 0,
          int all_threads = threads>
struct AllReduce {
  static_assert(threads == 1024 || threads == 512 || threads == 256 ||
                threads == 128 || threads == 64 || threads == 32 ||
                threads == 16 || threads == 8 || threads == 4 || threads == 2);
  static_assert(threads % scale == 0);
  template <typename T> static __device__ T run(T x, T *red_buf = nullptr) {
    constexpr int offset = threads / 2;
    constexpr int warpSize = 64;
    if constexpr (offset >= warpSize) {
      __syncthreads();
      red_buf[threadIdx.x - thread_offset] = x;
      __syncthreads();
      x = Reducer()(x, red_buf[(threadIdx.x - thread_offset) ^ offset]);
    } else {
      x = Reducer()(x, tl::shfl_xor_sync(uint64_t(-1), x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale, thread_offset, all_threads>::run(
          x, red_buf);
    }
  }
};

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
