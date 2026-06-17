/*!
 * \file tl/maca/transform/lower_maca_intrin.cc
 * \brief Lower MACA target-specific intrinsics.
 */

#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/transform.h>

#include "backend/common/target_utils.h"
#include "maca/runtime/maca_runtime.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

namespace attr {
constexpr const char *kL2PersistentMap = "l2_persistent_map";
} // namespace attr

using namespace tirx;
using namespace ffi;

class LowerMACAIntrin {
public:
  static PrimFunc Substitute(PrimFunc f) {
    Optional<Target> target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsMaca(target.value())) {
      return f;
    }

    Optional<Map<String, Array<PrimExpr>>> l2_map =
        f->GetAttr<Map<String, Array<PrimExpr>>>(attr::kL2PersistentMap);
    if (!l2_map.defined()) {
      return f;
    }

    Array<Stmt> prologue_stmts;
    for (const auto &kv : l2_map.value()) {
      Buffer buf = FindBufferByName(f, kv.first);
      if (!buf.defined()) {
        continue;
      }

      const Array<PrimExpr> &args = kv.second;
      ICHECK_GE(args.size(), 2);

      Array<PrimExpr> packed_args;
      packed_args.push_back(
          StringImm(tvm_maca_stream_set_access_policy_window));
      packed_args.push_back(MakeBasePtr(buf));
      packed_args.push_back(args[1]);
      packed_args.push_back(args[0]);
      prologue_stmts.push_back(Evaluate(
          Call(DataType::Int(32), builtin::tvm_call_packed(), packed_args)));
    }

    if (prologue_stmts.empty()) {
      return f;
    }

    Array<PrimExpr> reset_args;
    reset_args.push_back(StringImm(tvm_maca_stream_reset_access_policy_window));
    Stmt epilogue_stmt = Evaluate(
        Call(DataType::Int(32), builtin::tvm_call_packed(), reset_args));

    PrimFuncNode *fptr = f.CopyOnWrite();
    Stmt prologue = prologue_stmts.size() == 1 ? prologue_stmts[0]
                                               : SeqStmt(prologue_stmts);
    fptr->body = SeqStmt({prologue, fptr->body, epilogue_stmt});
    return f;
  }

private:
  static Buffer FindBufferByName(const PrimFunc &f, const String &name) {
    for (const auto &kv : f->buffer_map) {
      if (kv.second->name == name) {
        return kv.second;
      }
    }
    return Buffer();
  }

  static PrimExpr MakeBasePtr(const Buffer &buf) {
    PrimExpr base_ptr = buf->data;
    if (buf->elem_offset.defined() && !is_zero(buf->elem_offset)) {
      PrimExpr byte_offset = buf->elem_offset * IntImm(buf->elem_offset.dtype(),
                                                       buf->dtype.bytes());
      base_ptr = Call(DataType::Handle(), builtin::handle_add_byte_offset(),
                      {base_ptr, byte_offset});
    }
    return base_ptr;
  }
};

using namespace tirx::transform;

tvm::transform::Pass LowerMACAIntrin() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LowerMACAIntrin::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerMACAIntrin", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.maca.transform.LowerMACAIntrin", LowerMACAIntrin);
}

} // namespace tl
} // namespace tvm
