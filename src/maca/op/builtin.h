/*!
 * \file tl/maca/op/builtin.h
 * \brief MACA-specific TileLang intrinsic Ops and compiler attributes.
 */

#ifndef TVM_TL_MACA_OP_BUILTIN_H_
#define TVM_TL_MACA_OP_BUILTIN_H_

#include "cuda/op/builtin.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

/*!
 * \brief tvm intrinsic for amd matrix core mfma instructions.
 *
 *  void maca_mma(StringImm shape, StringImm A_layout, StringImm B_layout,
 *               StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *               Var multiplicand_a, Expr a_index,
 *               Var multiplicand_b, Expr b_index,
 *               Var accumulator, Expr c_index);
 */
TVM_DLL const Op &maca_mma();

/*!
 * \brief tilelang intrinsic for MACA barrier arrive and wait.
 */
TVM_DLL const Op &maca_barrier_arrive_and_wait();

/*!
 * \brief tilelang intrinsic for MACA synchronization barrier.
 */
TVM_DLL const Op &mxc_barrier_inst();

/*!
 * \brief tilelang intrinsic for MACA wait until the number of outstanding
 * global memory instructions drops below the threshold n.
 */
TVM_DLL const Op &mxc_arrive_gvmcnt();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_MACA_OP_BUILTIN_H_
