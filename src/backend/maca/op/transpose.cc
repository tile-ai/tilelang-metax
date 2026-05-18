/*!
 * \file tl/backend/maca/op/transpose.cc
 * \brief MACA implementation for tl.transpose lowering.
 */

#include "backend/common/op/transpose.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMacaTransposeTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "maca.Transpose",
      MatchMacaTransposeTarget,
      backend::Transpose::Lower,
  });
  return true;
}

const bool maca_transpose_registered = RegisterMacaTranspose();

} // namespace

} // namespace tl
} // namespace tvm
