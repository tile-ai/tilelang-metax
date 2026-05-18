/*!
 * \file tl/backend/maca/op/cumsum.cc
 * \brief MACA implementation for tl.cumsum lowering.
 */

#include "backend/common/op/cumsum.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMacaCumSumTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaCumSum() {
  RegisterCumSumImpl(CumSumImpl{
      "maca.CumSum",
      MatchMacaCumSumTarget,
      backend::CumSum::Lower,
  });
  return true;
}

const bool maca_cumsum_registered = RegisterMacaCumSum();

} // namespace

} // namespace tl
} // namespace tvm
