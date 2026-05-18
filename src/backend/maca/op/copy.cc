/*!
 * \file tl/backend/maca/op/copy.cc
 * \brief MACA implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "backend/maca/op/copy.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"
#include "transform/maca_memcpy_async_injector.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cstdint>
#include <sstream>
#include <vector>

namespace tvm {
namespace tl {

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

int GetEvictionPolicy(const CopyNode &op) {
  if (auto val = op.annotations.Get("eviction_policy")) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value;
    }
  }
  return 0; // default: evict_normal
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  // Backward-compatibility with historical annotation key.
  return GetBoolAnnotation(op, "force_cp_async");
}

} // namespace

namespace maca {

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level);

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer);

private:
  static Layout ComputeLinearLayout(const Buffer &shared_tensor);

  static void CollectFragmentLayouts(const PrimExpr &expr,
                                     const Map<Var, PrimExpr> &let_var_to_expr,
                                     const LayoutMap &existing_layouts,
                                     PrimExpr thread_extent,
                                     Range thread_bounds,
                                     Map<Buffer, Layout> &result_map);

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer, bool buffer_oob);

  static void CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst);

  static Stmt LowerNormal(const CopyNode &op, const LowerArgs &T,
                          arith::Analyzer *analyzer);

  static Stmt LowerMemcpyAsync(const CopyNode &op, const LowerArgs &T,
                               arith::Analyzer *analyzer);
};

Layout Copy::ComputeLinearLayout(const Buffer &shared_tensor) {
  Array<PrimExpr> input_size = shared_tensor->shape;
  Array<PrimExpr> forward_vars;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_vars.push_back(InputPlaceholder(i));
  }

  Array<PrimExpr> forward_index;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorDiv(forward_vars[i], 256));
  }
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorMod(forward_vars[i], 256));
  }
  return Layout(input_size, forward_index);
}

void Copy::CollectFragmentLayouts(const PrimExpr &expr,
                                  const Map<Var, PrimExpr> &let_var_to_expr,
                                  const LayoutMap &existing_layouts,
                                  PrimExpr thread_extent, Range thread_bounds,
                                  Map<Buffer, Layout> &result_map) {
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (auto bl = node.as<BufferLoadNode>()) {
      if (IsFragmentBuffer(bl->buffer) && !existing_layouts.count(bl->buffer) &&
          !result_map.count(bl->buffer)) {
        auto f = Fragment::FullyReplicated(bl->buffer->shape, thread_extent);
        result_map.Set(bl->buffer, f->BindThreadRange(thread_bounds));
      }
    } else if (auto var_node = node.as<VarNode>()) {
      auto var = tvm::ffi::GetRef<Var>(var_node);
      if (let_var_to_expr.count(var)) {
        CollectFragmentLayouts(let_var_to_expr[var], let_var_to_expr,
                               existing_layouts, thread_extent, thread_bounds,
                               result_map);
      }
    }
  });
}

LayoutMap Copy::InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                            InferLevel level) {
  CopyInst copy_inst =
      SelectInst(op, T.target, T.layout_map, T.analyzer, T.buffer_oob);
  CheckParallelLoopLayout(op, copy_inst);

  // For normal/cp.async, layout inference follows the generated
  // SIMT loop. MACA-specific explicit layout cases are handled above.
  return op.InferSIMTLayout(T, level);
}

void Copy::CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst) {
  if (!op.annotations.count(attr::kParallelLoopLayout)) {
    return;
  }
  if (copy_inst == CopyInst::kNormal || copy_inst == CopyInst::kMemcpyAsync) {
    return;
  }

  std::ostringstream oss;
  oss << "T.copy loop layout annotation requires SIMT copy; got "
      << CopyInstToString(copy_inst) << " for src=" << op.src->name
      << ", dst=" << op.dst->name
      << ". Remove loop_layout or change copy pattern.";
  LOG(FATAL) << oss.str();
}

CopyInst Copy::SelectInst(const CopyNode &op, Target target,
                          const LayoutMap &layout_map,
                          arith::Analyzer *analyzer, bool buffer_oob) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  ctx.layout_map = &layout_map;
  ctx.analyzer = analyzer;
  ctx.buffer_oob = buffer_oob;
  ctx.emit_diagnostics = true;
  auto result = SelectCopyInstForLowering(op, ctx);
  ICHECK(result.supported) << result.reason;
  return result.inst;
}

Stmt Copy::Lower(const CopyNode &op, const LowerArgs &T,
                 arith::Analyzer *analyzer) {
  auto copy_inst =
      SelectInst(op, T.target, T.layout_map, analyzer, /*buffer_oob=*/false);
  if (copy_inst == CopyInst::kMemcpyAsync) {
    auto memcpy_async = LowerMemcpyAsync(op, T, analyzer);
    ICHECK(memcpy_async.defined()) << "Failed to lower memcpy_async copy";
    return memcpy_async;
  } else if (copy_inst == CopyInst::kNormal) {
    return LowerNormal(op, T, analyzer);
  } else {
    LOG(FATAL) << "Unsupported copy inst " << static_cast<int>(copy_inst);
  }
}

Stmt Copy::LowerMemcpyAsync(const CopyNode &op, const LowerArgs &T,
                            arith::Analyzer *analyzer) {
  using namespace tvm::transform;

  PassContext pass_ctx = PassContext::Current();
  bool enable_async_copy =
      pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
  if (!enable_async_copy) {
    return LowerNormal(op, T, analyzer);
  }

  PrimExpr mbar_handle;
  if (auto user_barrier = op.annotations.Get("barrier")) {
    mbar_handle = Downcast<PrimExpr>(user_barrier.value());
  } else {
    LOG(FATAL) << "T.maca_async_copy() requires a barrier argument. "
               << "Use T.maca_async_copy(src, dst, barrier=bar).";
  }

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  Stmt lowered_loop =
      LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var, analyzer,
                        T.layout_map, par_op->GetPredicate(T.thread_var));

  auto inject_result = InjectMACAMemcpyAsync(lowered_loop, mbar_handle);
  Stmt memcpy_async_loop = inject_result.stmt;
  if (!inject_result.injected_maca_memcpy_async) {
    DLOG(WARNING) << "maca_async_copy rewrite miss for copy src ="
                  << op.src->name << " (scope=" << op.src.scope()
                  << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
                  << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype
                  << ")";
    DLOG(WARNING) << "Fallback to normal copy because cp.async rewrite found "
                     "no eligible global->shared store.";
    return LowerNormal(op, T, analyzer);
  }
  return memcpy_async_loop;
}

Stmt Copy::LowerNormal(const CopyNode &op, const LowerArgs &T,
                       arith::Analyzer *analyzer) {
  return tl::LowerNormalCopy(op, T, analyzer);
}

} // namespace maca

namespace {

bool MatchMacaCopyTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaCopy() {
  RegisterCopyImpl(CopyImpl{
      "maca.Copy",
      MatchMacaCopyTarget,
      100,
      maca::Copy::InferLayout,
      maca::Copy::Lower,
  });
  return true;
}

const bool maca_copy_registered = RegisterMacaCopy();

} // namespace

} // namespace tl
} // namespace tvm
