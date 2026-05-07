/*!
 * \brief Lower eligible global->shared copies into MACA memcpy_async
 * \file lower_maca_memcpy_async.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "../op/builtin.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "maca_memcpy_async_injector.h"
#include "tir/ir/buffer_common.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {

using namespace tir;

class MACAMemcpyAsyncInjector : public StmtMutator {
public:
  explicit MACAMemcpyAsyncInjector(const PrimExpr &mbar) : mbar_(mbar) {}

  bool InjectedMACAMemcpyAsync() const { return injected_maca_memcpy_async_; }

  Stmt VisitStmt_(const ForNode *op) final {
    // Track nested vectorized loop extents so we can decide whether an
    // element-wise copy has a legal final memcpy_async width after later loop
    // vectorization:
    //   for v in T.vectorized(k): tl.maca_memcpy_async(dst, src, elem_count)
    // => maca_memcpy_async(dst_base, src_base, elem_count * k)
    //
    // TileLang records logical element counts in tl.maca_memcpy_async. The
    // final byte width is derived later from the access_ptr dtype, so subbyte
    // dtypes such as int4/fp4/int2/int1 remain representable here.
    int previous_vectorized_lanes = current_vectorized_lanes_;
    bool pushed_vectorized_loop = false;
    if (op->kind == ForKind::kVectorized) {
      const auto *extent_imm = op->extent.as<IntImmNode>();
      ICHECK(extent_imm)
          << "Vectorized loops must have constant extent, but got "
          << op->extent;
      int lanes = static_cast<int>(extent_imm->value);
      if (lanes > 1 && current_vectorized_lanes_ <=
                           std::numeric_limits<int>::max() / lanes) {
        current_vectorized_lanes_ *= lanes;
        active_vectorized_loops_.push_back({op->loop_var, lanes});
        pushed_vectorized_loop = true;
      }
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    if (pushed_vectorized_loop) {
      active_vectorized_loops_.pop_back();
    }
    current_vectorized_lanes_ = previous_vectorized_lanes;
    return stmt;
  }

  Optional<Stmt> TryInject(const BufferLoadNode *load,
                           const BufferStoreNode *store,
                           bool predicated = false,
                           const PrimExpr &predicate_value = PrimExpr()) {
    // Pipeline:
    // 1) Analyze source/destination indices and transfer width eligibility.
    // 2) Build tl.maca_memcpy_async with scalar/vectorized base offsets when
    // the eventual byte width is representable.
    std::optional<CopyIndexInfo> index_info = PrepareCopyIndexInfo(load, store);
    if (!index_info.has_value()) {
      return Optional<Stmt>();
    }

    if (index_info->index_lanes == 1) {
      if (current_vectorized_lanes_ > 1 &&
          !HasContiguousVectorizedOffsets(index_info->src_index,
                                          index_info->dst_index)) {
        return Optional<Stmt>();
      }
      return MakeMemcpyAsyncStmtFromLoads(
          store,
          /*dst_base_load=*/BufferLoad(store->buffer, store->indices),
          /*src_base_load=*/BufferLoad(load->buffer, load->indices),
          /*num_elems=*/index_info->per_access_num_elems,
          /*total_bytes=*/index_info->total_bytes, /*mbar=*/mbar_, predicated,
          predicate_value);
    }

    Optional<Array<PrimExpr>> src_base_indices =
        ExtractVectorBaseIndices(load->indices);
    Optional<Array<PrimExpr>> dst_base_indices =
        ExtractVectorBaseIndices(store->indices);
    if (!src_base_indices.defined() || !dst_base_indices.defined()) {
      // If we can't extract base indices from vectorized accesses, fall back.
      if (predicated) {
        LOG(WARNING)
            << "Cannot extract base indices from vectorized accesses for "
               "predicated cp.async; falling back to regular buffer store/load";
      }
      return Optional<Stmt>();
    }
    return MakeMemcpyAsyncStmtFromLoads(
        store,
        /*dst_base_load=*/BufferLoad(store->buffer, dst_base_indices.value()),
        /*src_base_load=*/BufferLoad(load->buffer, src_base_indices.value()),
        /*num_elems=*/index_info->per_access_num_elems,
        /*total_bytes=*/index_info->total_bytes, /*mbar=*/mbar_, predicated,
        predicate_value);
  }

  Stmt VisitStmt_(const BufferStoreNode *store) final {
    if (!IsSharedBuffer(store->buffer)) {
      return StmtMutator::VisitStmt_(store);
    }

    Optional<PrimExpr> predicate = std::nullopt;
    const BufferLoadNode *load =
        MatchZeroFillBufferLoad(store->value, &predicate);
    if (load) {
      Optional<Stmt> injected =
          TryInject(load, store, predicate.defined(),
                    predicate.defined() ? predicate.value() : PrimExpr());
      if (injected.defined()) {
        injected_maca_memcpy_async_ = true;
        return injected.value();
      }
    }

    return StmtMutator::VisitStmt_(store);
  }

private:
  // A copy candidate represented after flattening source/destination indexing.
  struct CopyIndexInfo {
    PrimExpr src_index;
    PrimExpr dst_index;
    int index_lanes{1};
    int total_bytes{0};
    int per_access_num_elems{0};
  };

  struct ActiveVectorizedLoop {
    Var loop_var;
    int extent;
  };

  // ---- Copy candidate analysis helpers ----
  static bool IsZeroValue(const PrimExpr &expr) {
    if (const auto *broadcast = expr.as<BroadcastNode>()) {
      return IsZeroValue(broadcast->value);
    }
    if (const auto *float_imm = expr.as<FloatImmNode>()) {
      return float_imm->value == 0.0f;
    }
    if (const auto *int_imm = expr.as<IntImmNode>()) {
      return int_imm->value == 0;
    }
    return false;
  }

  static const BufferLoadNode *
  MatchZeroFillBufferLoad(const PrimExpr &value,
                          Optional<PrimExpr> *predicate) {
    if (const auto *load = value.as<BufferLoadNode>()) {
      return load;
    }

    const auto *call = value.as<CallNode>();
    if (!call || !call->op.same_as(builtin::if_then_else()) ||
        !IsZeroValue(call->args[2])) {
      return nullptr;
    }

    const BufferLoadNode *load =
        MatchZeroFillBufferLoad(call->args[1], predicate);
    if (load == nullptr) {
      return nullptr;
    }

    *predicate =
        predicate->defined()
            ? Optional<PrimExpr>(And(call->args[0], predicate->value()))
            : Optional<PrimExpr>(call->args[0]);
    return load;
  }

  static Optional<PrimExpr>
  FlattenToLinearOffset(const Buffer &buf,
                        const ffi::Array<PrimExpr> &indices) {
    // Convert N-D indices (potentially with axis_separators) into a single
    // row-major linear element offset.
    ffi::Array<PrimExpr> physical = buf.OffsetOf(indices);
    Buffer flattened_buf = buf.GetFlattenedBuffer();
    if (physical.size() != flattened_buf->shape.size() || physical.empty()) {
      return Optional<PrimExpr>();
    }

    PrimExpr linear = physical[0];
    for (size_t i = 1; i < physical.size(); ++i) {
      linear = linear * flattened_buf->shape[i] + physical[i];
    }
    return linear;
  }

  std::optional<CopyIndexInfo>
  PrepareCopyIndexInfo(const BufferLoadNode *load,
                       const BufferStoreNode *store) {
    if (!IsGlobalBuffer(load->buffer)) {
      return std::nullopt;
    }

    Optional<PrimExpr> src_index_opt =
        FlattenToLinearOffset(load->buffer, load->indices);
    Optional<PrimExpr> dst_index_opt =
        FlattenToLinearOffset(store->buffer, store->indices);
    if (!src_index_opt.defined() || !dst_index_opt.defined()) {
      return std::nullopt;
    }

    PrimExpr src_index = src_index_opt.value();
    PrimExpr dst_index = dst_index_opt.value();
    if (src_index->dtype.lanes() != dst_index->dtype.lanes()) {
      // Not a straightforward vectorized copy; skip.
      return std::nullopt;
    }

    const int index_lanes = src_index->dtype.lanes();
    const int value_lanes = load->dtype.lanes();
    if (value_lanes > 1 && index_lanes > 1 && value_lanes != index_lanes) {
      // Mismatched vector lane representations; be conservative.
      return std::nullopt;
    }

    const int effective_lanes = std::max(value_lanes, index_lanes);
    const int per_access_bits = effective_lanes * load->dtype.bits();
    const int total_bits = static_cast<int>(per_access_bits) *
                           static_cast<int>(current_vectorized_lanes_);
    // memcpy_async is byte-granular. `tl.maca_memcpy_async` stores logical
    // element counts, but we still need to know that the eventual vectorized
    // transfer can map to a legal byte width without over-copying packed
    // subbyte data.
    if (total_bits % 8 != 0) {
      return std::nullopt;
    }
    const int total_bytes = total_bits / 8;
    if (!IsValidCPAsyncTransferBytes(total_bytes)) {
      return std::nullopt;
    }

    CopyIndexInfo info;
    info.src_index = src_index;
    info.dst_index = dst_index;
    info.index_lanes = index_lanes;
    info.total_bytes = total_bytes;
    info.per_access_num_elems = effective_lanes;
    return info;
  }

  static PrimExpr ExtractVectorBase(const PrimExpr &index) {
    if (index.dtype().lanes() == 1) {
      return index;
    }
    if (const auto *broadcast = index.as<BroadcastNode>()) {
      return broadcast->value;
    }
    if (const auto *ramp = index.as<RampNode>()) {
      if (!is_one(ramp->stride)) {
        return PrimExpr();
      }
      return ramp->base;
    }

    const auto *add = index.as<AddNode>();
    if (!add) {
      return PrimExpr();
    }

    // Common pattern after flattening a vectorized N-D buffer access:
    //   (broadcast(base_offset) + ramp(vec_base, 1, lanes))
    // or its commuted form:
    //   (ramp(vec_base, 1, lanes) + broadcast(base_offset))
    const PrimExpr &lhs = add->a;
    const PrimExpr &rhs = add->b;
    if (const auto *lhs_ramp = lhs.as<RampNode>()) {
      if (!is_one(lhs_ramp->stride)) {
        return PrimExpr();
      }
      if (const auto *rhs_broadcast = rhs.as<BroadcastNode>()) {
        return tir::Add(lhs_ramp->base, rhs_broadcast->value);
      }
    }
    if (const auto *rhs_ramp = rhs.as<RampNode>()) {
      if (!is_one(rhs_ramp->stride)) {
        return PrimExpr();
      }
      if (const auto *lhs_broadcast = lhs.as<BroadcastNode>()) {
        return tir::Add(rhs_ramp->base, lhs_broadcast->value);
      }
    }
    return PrimExpr();
  }

  static Optional<Array<PrimExpr>>
  ExtractVectorBaseIndices(const Array<PrimExpr> &indices) {
    Array<PrimExpr> base_indices;
    base_indices.reserve(indices.size());
    for (const PrimExpr &index : indices) {
      PrimExpr base = ExtractVectorBase(index);
      if (!base.defined()) {
        return Optional<Array<PrimExpr>>();
      }
      base_indices.push_back(base);
    }
    return base_indices;
  }

  static PrimExpr MakeAccessPtrFromLoad(const BufferLoad &base_load, int extent,
                                        int rw_mask) {
    return Call(DataType::Handle(), tvm::tl::access_ptr(),
                {base_load, IntImm(DataType::Int(32), extent),
                 IntImm(DataType::Int(32), rw_mask)});
  }

  static Optional<Stmt> MakeMemcpyAsyncStmtFromLoads(
      const BufferStoreNode *store, const BufferLoad &dst_base_load,
      const BufferLoad &src_base_load, int num_elems, int total_bytes,
      const PrimExpr &mbar, bool predicated, const PrimExpr &predicate_value) {
    PrimExpr dst_access_ptr =
        MakeAccessPtrFromLoad(dst_base_load, num_elems, /*rw_mask=*/2);
    PrimExpr src_access_ptr =
        MakeAccessPtrFromLoad(src_base_load, num_elems, /*rw_mask=*/1);

    ffi::Array<PrimExpr> memcpy_async_args;
    if (predicated) {
      memcpy_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(num_elems),
                           mbar, predicate_value};
    } else {
      memcpy_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(num_elems),
                           mbar};
    }
    std::string barrier_type;
    if (4 == total_bytes) {
      barrier_type = "b32vectype";
    } else if (8 == total_bytes) {
      barrier_type = "b64vectype";
    } else {
      barrier_type = "b128vectype";
    }
    return Evaluate(Call(store->buffer->dtype, tvm::tl::maca_memcpy_async(),
                         memcpy_async_args,
                         {{"barrier_type", StringImm(barrier_type)}}));
  }

  // ---- Vectorized-offset contiguity helpers ----
  static bool TryGetConstInt64(const PrimExpr &expr, int64_t *value) {
    if (const auto *imm = expr.as<IntImmNode>()) {
      *value = imm->value;
      return true;
    }
    return false;
  }

  bool HasUnitStrideForVectorizedLoop(const PrimExpr &expr,
                                      const ActiveVectorizedLoop &loop) {
    PrimExpr prev = analyzer_.Simplify(
        Substitute(expr, {{loop.loop_var, IntImm(loop.loop_var->dtype, 0)}}));

    int64_t stride = 0;
    for (int value = 1; value < loop.extent; ++value) {
      PrimExpr curr = analyzer_.Simplify(Substitute(
          expr, {{loop.loop_var, IntImm(loop.loop_var->dtype, value)}}));
      PrimExpr delta = analyzer_.Simplify(curr - prev);
      int64_t delta_value = 0;
      if (!TryGetConstInt64(delta, &delta_value)) {
        return false;
      }
      if (value == 1) {
        stride = delta_value;
      } else if (delta_value != stride) {
        return false;
      }
      prev = curr;
    }

    return stride == 1;
  }

  bool HasContiguousVectorizedOffsets(const PrimExpr &src_index,
                                      const PrimExpr &dst_index) {
    for (const auto &loop : active_vectorized_loops_) {
      if (!HasUnitStrideForVectorizedLoop(src_index, loop) ||
          !HasUnitStrideForVectorizedLoop(dst_index, loop)) {
        return false;
      }
    }
    return true;
  }

  int current_vectorized_lanes_{1};
  std::vector<ActiveVectorizedLoop> active_vectorized_loops_;
  arith::Analyzer analyzer_;
  PrimExpr mbar_;
  bool injected_maca_memcpy_async_{false};
};

using namespace tir::transform;

MACAMemcpyAsyncInjectResult InjectMACAMemcpyAsync(const Stmt &body,
                                                  const PrimExpr &mbar) {
  MACAMemcpyAsyncInjector injector(mbar);
  Stmt injected = injector(body);
  return {injected, injector.InjectedMACAMemcpyAsync()};
}

} // namespace tl
} // namespace tvm
