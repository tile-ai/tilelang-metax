#pragma once

#include "../common.h"

namespace tl {

TL_DEVICE float32x4 mxc_mma_16x16x4f32(float a, float b, float32x4 c) {
  return __builtin_mxc_mma_16x16x8tf32({a, 0.0f}, {b, 0.0f}, c);
}

TL_DEVICE int32x4 mxc_mma_16x16x16i8(int a, int b, int32x4 c) {
  return __builtin_mxc_mma_16x16x16i8({a, 0}, {b, 0}, c);
}

} // namespace tl
