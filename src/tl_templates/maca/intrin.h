#pragma once

#include "common.h"
#include "mctlass/mctlass.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"

namespace tl {

constexpr int NumThreadsPerWarp = 64;

namespace detail {

TL_DEVICE constexpr int default_warp_size() { return 64; }

TL_DEVICE constexpr int default_warps_per_group() { return 16; }

TL_DEVICE int linear_thread_idx_in_block() {
#if defined(__MACA_ARCH__)
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
  return 0;
#endif
}

} // namespace detail

TL_DEVICE int get_lane_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() % warp_size;
}

TL_DEVICE int get_warp_idx_sync(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int get_warp_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int
get_warp_group_idx(int warp_size = detail::default_warp_size(),
                   int warps_per_group = detail::default_warps_per_group()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  warps_per_group =
      warps_per_group > 0 ? warps_per_group : detail::default_warps_per_group();
  int threads_per_group = warp_size * warps_per_group;
  threads_per_group = threads_per_group > 0 ? threads_per_group : warp_size;
  return detail::linear_thread_idx_in_block() / threads_per_group;
}

TL_DEVICE void warpgroup_arrive() { cute::warpgroup_arrive(); }
TL_DEVICE void warpgroup_commit_batch() { cute::warpgroup_commit_batch(); }

template <int NumMma> TL_DEVICE void warpgroup_wait() {
  cute::warpgroup_wait<NumMma>();
}

TL_DEVICE void warpgroup_fence_operand(uint32_t *regs, int count) {
#pragma unroll
  for (int i = 0; i < count; ++i) {
    cute::warpgroup_fence_operand(regs[i]);
  }
}

TL_DEVICE void warpgroup_fence_operand(float *regs, int count) {
#pragma unroll
  for (int i = 0; i < count; ++i) {
    cute::warpgroup_fence_operand(regs[i]);
  }
}

MCTLASS_DEVICE
int canonical_warp_idx_sync() {
#if defined(__MACA_ARCH__)
  return __shfl_sync(UINT64_MAX, threadIdx.x / NumThreadsPerWarp, 0);
#else
  return 0;
#endif
}

// Template parameter:
//   thread_extent: the logical size (in number of threads) of each "group"
//                  within which we want to elect exactly ONE representative
//                  thread.
template <int thread_extent> TL_DEVICE bool tl_shuffle_elect() {

  // Special case: thread_extent == 0 means "elect exactly one thread
  // in the entire thread block", i.e., the leader of the first warp of the
  // block.
  if constexpr (thread_extent == 0) {
    // mctlass::canonical_warp_idx_sync():
    //   Returns the warp ID within the thread block in a "canonical" way
    //   (0 for the first warp, 1 for the second, ...).
    // cute::elect_one_sync():
    //   Elect exactly one lane in the warp to return true (typically lane 0),
    //   other lanes return false.
    // The condition ensures that:
    //   (1) We are in warp 0 of the block.
    //   (2) We are the elected lane in this warp.
    return canonical_warp_idx_sync() == 0 && cute::elect_one_sync();
  } else if constexpr (thread_extent == 64) {
    return cute::elect_one_sync();
  }
  // General case: thread_extent != 0
  // (threadIdx.x / 64) is the warp index in the block.
  // (thread_extent / 64) is the number of warps in one group of size
  // thread_extent. We take warp_id % num_warps_in_group to get the warp's index
  // within the group.
  // __shfl_sync(mask, value, srcLane): broadcast 'value' from srcLane to all
  // lanes in the warp. Here it broadcasts the group-local warp index from lane
  // 0. Comparing to 0 selects only the group's warp 0.
  return __shfl_sync(UINT64_MAX, // full warp mask
                     (threadIdx.x / 64) %
                         (thread_extent / 64), // warp index within group
                     0                         // take the value from lane 0
                     ) == 0 &&
         // Within that group leader warp, elect exactly one lane (typically
         // lane 0) to be the single representative for the group.
         cute::elect_one_sync();
}

} // namespace tl
