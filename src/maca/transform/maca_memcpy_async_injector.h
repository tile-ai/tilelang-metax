#pragma once

#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

struct MACAMemcpyAsyncInjectResult {
  tvm::tirx::Stmt stmt;
  bool injected_maca_memcpy_async{false};
};

/*! \brief Inject MACA memcpy_async lowering patterns into a statement.
 */
MACAMemcpyAsyncInjectResult InjectMACAMemcpyAsync(const tvm::tirx::Stmt &body);

} // namespace tl
} // namespace tvm
