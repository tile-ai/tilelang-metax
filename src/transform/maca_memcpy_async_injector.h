#pragma once

#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

struct MACAMemcpyAsyncInjectResult {
  tvm::tir::Stmt stmt;
  bool injected_maca_memcpy_async{false};
};

/*! \brief Inject MACA memcpy_async lowering patterns into a statement.
 */
MACAMemcpyAsyncInjectResult InjectMACAMemcpyAsync(const tvm::tir::Stmt &body,
                                                  const tvm::PrimExpr &mbar);

} // namespace tl
} // namespace tvm
