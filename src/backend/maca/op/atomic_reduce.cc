/*!
 * \file tl/backend/maca/op/atomic_reduce.cc
 * \brief MACA implementation for tl.atomicmax/tl.atomicmin lowering.
 */

#include "backend/common/op/atomic_reduce.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMacaAtomicReduceTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "maca.AtomicReduce",
      MatchMacaAtomicReduceTarget,
      backend::AtomicReduce::InferLayout,
      backend::AtomicReduce::Lower,
  });
  return true;
}

const bool maca_atomic_reduce_registered = RegisterMacaAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm
