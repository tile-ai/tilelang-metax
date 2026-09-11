/*!
 * \file tl/maca/op/builtin.cc
 * \brief Registration of MACA-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

// maca_barrier_arrive_and_wait(barrier)
// MACA barrier arrive and wait operation
TIR_DEFINE_TL_BUILTIN(maca_barrier_arrive_and_wait)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(maca_mma).set_num_inputs(12).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// mxc_barrier_inst()
// synchronization barrier for MXC operations.
TIR_DEFINE_TL_BUILTIN(mxc_barrier_inst)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// mxc_arrive_gvmcnt(n)
// Wait until the number of outstanding global memory instructions
// drops below the threshold n. Used for flow control and dependency
// synchronization of asynchronous memory copies.
TIR_DEFINE_TL_BUILTIN(mxc_arrive_gvmcnt)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm

#undef TIR_DEFINE_TL_BUILTIN
