/*!
 * \file tl/maca/target_utils.h
 * \brief MACA target attribute helpers.
 */

#ifndef TVM_TL_MACA_TARGET_UTILS_H_
#define TVM_TL_MACA_TARGET_UTILS_H_

#include <tvm/runtime/data_type.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsMaca(Target target);
bool TargetIsCuTeDSL(Target target);

bool TargetMacaHasAsyncCopy(Target target);
int TargetMacaGetWarpSize(Target target);

bool IsMacaVectorizableFP8(DataType dtype);
bool IsMacaVectorizableCast(DataType from_ty, DataType target_ty);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_MACA_TARGET_UTILS_H_
