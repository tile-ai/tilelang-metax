/*!
 * \file tl/backend/maca/op/finalize_reducer.cc
 * \brief MACA implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"

#include "maca/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace maca {

struct FinalizeReducer : backend::FinalizeReducerLowerer<FinalizeReducer> {
  static int WarpSize(Target target) { return TargetMacaGetWarpSize(target); }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ", tl::SyncThreadsBarrier"
       << ", " << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ">::run";
    return ss.str();
  }
};

} // namespace maca

namespace {

bool MatchMacaFinalizeReducerTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "maca.FinalizeReducer",
      MatchMacaFinalizeReducerTarget,
      maca::FinalizeReducer::Lower,
  });
  return true;
}

const bool maca_finalize_reducer_registered = RegisterMacaFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
