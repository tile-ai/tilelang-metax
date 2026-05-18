/*!
 * \file tl/backend/maca/op/reduce.cc
 * \brief MACA implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace maca {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool SupportsFp16Bf16NanReduce(Target target) {
    return TargetIsMaca(target);
  }

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

bool MatchMacaReduceTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaReduce() {
  RegisterReduceImpl(ReduceImpl{
      "maca.Reduce",
      MatchMacaReduceTarget,
      maca::Reduce::Lower,
  });
  return true;
}

const bool maca_reduce_registered = RegisterMacaReduce();

} // namespace

} // namespace tl
} // namespace tvm
