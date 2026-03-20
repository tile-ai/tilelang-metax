#pragma once

#include "common.h"

namespace mctlass {

struct Barrier {
  uint64_t data;
  __device__ inline void init(int count) {}
};

} // namespace mctlass

namespace tl {

__device__ inline void fence_barrier_init() { __syncthreads(); }

} // namespace tl

using Barrier = mctlass::Barrier;
