/*!
 * \file tl/backend/maca/op/copy_analysis.cc
 * \brief MACA copy instruction classification helpers.
 */

#include "backend/maca/op/copy.h"

#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"

#include <tvm/tir/transform.h>

#include <sstream>
#include <utility>

namespace tvm {
namespace tl {
namespace maca {

using namespace tir;

namespace {

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool CheckMemcpyAsyncCopyPreconditions(const CopyNode &op) {
  return IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst) &&
         op.src->dtype == op.dst->dtype;
}

bool CheckMemcpyAsyncCopy(const CopyNode &op, Target target,
                          const LayoutMap &layout_map,
                          arith::Analyzer *analyzer) {
  if (!TargetHasAsyncCopy(target)) {
    return false;
  }
  if (!CheckMemcpyAsyncCopyPreconditions(op)) {
    return false;
  }
  // Skip vectorize size checks here because the layout is not stable during
  // layout inference and transform classification.
  return true;
}

} // namespace

const char *CopyInstToString(CopyInst inst) {
  switch (inst) {
  case CopyInst::kNormal:
    return "Normal";
  case CopyInst::kMemcpyAsync:
    return "MemcpyAsync";
  case CopyInst::kInvalid:
    return "Invalid";
  default:
    return "Unknown";
  }
}

bool CopyInstIsMemcpyAsync(CopyInst inst) {
  return inst == CopyInst::kMemcpyAsync;
}

namespace {

struct CopyFacts {
  bool maca_like_target = false;
  bool has_layout_map = false;
  bool explicit_memcpy_async = false;
  bool can_memcpy_async = false;
  std::string async_unavailable_reason;
};

bool IsMacaLikeTarget(Target target) {
  return target.defined() && (TargetIsMaca(target) || TargetIsCuTeDSL(target));
}

CopyInstSelection Supported(CopyInst inst) {
  return CopyInstSelection{inst, true, ""};
}

CopyInstSelection Unsupported(std::string reason) {
  return CopyInstSelection{CopyInst::kInvalid, false, std::move(reason)};
}

std::string MakeAsyncUnavailableReason(const CopyNode &op, Target target) {
  std::ostringstream oss;
  if (!target.defined()) {
    oss << "T.maca_async_copy requires a defined target.";
  } else if (!TargetHasAsyncCopy(target)) {
    oss << "T.maca_async_copy is only supported on targets with memcpy_async. "
           "Got target="
        << target;
  } else if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
    oss << "T.maca_async_copy only supports global->shared/shared.dyn copies. "
           "Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << "), dst=" << op.dst->name << " (scope=" << op.dst.scope() << ").";
  } else if (op.src->dtype != op.dst->dtype) {
    oss << "T.maca_async_copy requires equal byte-addressable dtypes. Got src "
           "dtype="
        << op.src->dtype << ", dst dtype=" << op.dst->dtype << ".";
  } else {
    oss << "Explicit async copy semantics require memcpy_async lowering, but "
           "constraints were not satisfied. Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
        << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype << ").";
  }
  return oss.str();
}

bool IsAutoAsyncCopyEnabled(bool default_enabled) {
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  return pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(default_enabled))
      .value();
}

CopyFacts AnalyzeCopyFacts(const CopyNode &op, const CopyAnalysisContext &ctx) {
  CopyFacts facts;
  facts.maca_like_target = IsMacaLikeTarget(ctx.target);
  facts.has_layout_map = ctx.layout_map != nullptr;
  facts.explicit_memcpy_async = GetIsAsyncCopy(op);
  facts.async_unavailable_reason = MakeAsyncUnavailableReason(op, ctx.target);

  if (!facts.maca_like_target) {
    return facts;
  }

  arith::Analyzer local_analyzer;
  arith::Analyzer *analyzer =
      ctx.analyzer != nullptr ? ctx.analyzer : &local_analyzer;
  static const LayoutMap empty_layout_map;
  const LayoutMap &layout_map =
      ctx.layout_map != nullptr ? *ctx.layout_map : empty_layout_map;

  facts.can_memcpy_async =
      CheckMemcpyAsyncCopy(op, ctx.target, layout_map, analyzer);
  return facts;
}

} // namespace

CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx) {
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);

  if (facts.explicit_memcpy_async) {
    return facts.can_memcpy_async ? Supported(CopyInst::kMemcpyAsync)
                                  : Unsupported(facts.async_unavailable_reason);
  }

  return Supported(CopyInst::kNormal);
}

std::string ClassifyCopyForInstructionAnnotation(const CopyNode &op,
                                                 Target target,
                                                 bool in_pipeline) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.maca_like_target) {
    return "sync";
  }

  if (facts.explicit_memcpy_async) {
    return facts.can_memcpy_async ? "memcpy_async" : "sync";
  }

  if (in_pipeline && IsAutoAsyncCopyEnabled(/*default_enabled=*/false) &&
      facts.can_memcpy_async) {
    return "memcpy_async";
  }

  return "sync";
}

bool IsPipelineManagedMemcpyAsyncCopy(const CopyNode &op, Target target) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.maca_like_target || facts.explicit_memcpy_async) {
    return false;
  }
  return facts.can_memcpy_async;
}

} // namespace maca
} // namespace tl
} // namespace tvm
