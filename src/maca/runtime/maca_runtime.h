/*!
 * \file maca_runtime.h
 * \brief MACA runtime helpers for TileLang.
 */
#ifndef TVM_TL_BACKEND_MACA_RUNTIME_H_
#define TVM_TL_BACKEND_MACA_RUNTIME_H_

namespace tvm {
namespace tl {

constexpr const char *tvm_maca_stream_set_access_policy_window =
    "__tvm_maca_stream_set_access_policy_window";
constexpr const char *tvm_maca_stream_reset_access_policy_window =
    "__tvm_maca_stream_reset_access_policy_window";

} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_MACA_RUNTIME_H_
