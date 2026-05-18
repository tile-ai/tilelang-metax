/*!
 * \file tl/backend/maca/op/fill.cc
 * \brief MACA implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMacaFillTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaFill() {
  RegisterFillImpl(FillImpl{
      "maca.Fill",
      MatchMacaFillTarget,
      backend::Fill::Lower,
  });
  return true;
}

const bool maca_fill_registered = RegisterMacaFill();

} // namespace

} // namespace tl
} // namespace tvm
