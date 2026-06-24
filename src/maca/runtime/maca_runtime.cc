/*!
 * \file maca_runtime.cc
 * \brief MACA L2 persisting cache access policy window helpers.
 */

#include "maca_runtime.h"

#include "maca_common.h"

#include <cstring>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

namespace {

thread_local size_t __tl_prev_persisting_l2_cache_size = 0;
thread_local bool __tl_prev_persisting_l2_cache_saved = false;

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed(
          tvm_maca_stream_set_access_policy_window,
          [](ffi::PackedArgs args, ffi::Any *ret) {
            TVM_FFI_ICHECK(args.size() >= 2)
                << "Expected at least base_ptr and num_bytes";

            void *base_ptr = args[0].cast<void *>();
            size_t num_bytes = static_cast<size_t>(args[1].cast<int64_t>());
            float hit_ratio = 0.8f;
            if (args.size() >= 3) {
              hit_ratio = static_cast<float>(args[2].cast<double>());
            }
            mcStream_t stream = nullptr;
            if (args.size() >= 4) {
              stream = reinterpret_cast<mcStream_t>(args[3].cast<void *>());
            }
            size_t l2_limit_bytes = num_bytes;
            if (args.size() >= 5) {
              l2_limit_bytes = static_cast<size_t>(args[4].cast<int64_t>());
            }

            int device_id = 0;
            MACA_CALL(mcGetDevice(&device_id));
            int max_persisting = 0;
            mcError_t attrib_err = mcDeviceGetAttribute(
                &max_persisting, mcDeviceAttributeMaxPersistingL2CacheSize,
                device_id);
            if (attrib_err == mcSuccess && max_persisting > 0 &&
                l2_limit_bytes > static_cast<size_t>(max_persisting)) {
              l2_limit_bytes = static_cast<size_t>(max_persisting);
            }

            size_t init_persisting_l2_cache_size = 0;
            mcError_t limit_get_err = mcDeviceGetLimit(
                &init_persisting_l2_cache_size, mcLimitPersistingL2CacheSize);
            if (limit_get_err == mcSuccess) {
              __tl_prev_persisting_l2_cache_size =
                  init_persisting_l2_cache_size;
              __tl_prev_persisting_l2_cache_saved = true;
              mcDeviceSetLimit(mcLimitPersistingL2CacheSize, l2_limit_bytes);
            }

            mcStreamAttrValue stream_attribute;
            std::memset(&stream_attribute, 0, sizeof(stream_attribute));
            stream_attribute.accessPolicyWindow.base_ptr = base_ptr;
            stream_attribute.accessPolicyWindow.num_bytes = l2_limit_bytes;
            stream_attribute.accessPolicyWindow.hitRatio = hit_ratio;
            stream_attribute.accessPolicyWindow.hitProp =
                mcAccessPropertyPersisting;
            stream_attribute.accessPolicyWindow.missProp =
                mcAccessPropertyStreaming;

            mcStreamSetAttribute(stream, mcStreamAttributeAccessPolicyWindow,
                                 &stream_attribute);
            *ret = 0;
          })
      .def_packed(
          tvm_maca_stream_reset_access_policy_window,
          [](ffi::PackedArgs args, ffi::Any *ret) {
            mcStream_t stream = nullptr;
            if (args.size() >= 1) {
              stream = reinterpret_cast<mcStream_t>(args[0].cast<void *>());
            }

            mcStreamAttrValue stream_attribute;
            std::memset(&stream_attribute, 0, sizeof(stream_attribute));
            stream_attribute.accessPolicyWindow.num_bytes = 0;
            mcStreamSetAttribute(stream, mcStreamAttributeAccessPolicyWindow,
                                 &stream_attribute);
            // On some MACA devices, mcCtxResetPersistingL2Cache is not
            // supported and returns mcErrorNotSupported or mcErrorInvalidValue.
            // We can safely ignore it.
            mcCtxResetPersistingL2Cache();

            if (__tl_prev_persisting_l2_cache_saved) {
              mcDeviceSetLimit(mcLimitPersistingL2CacheSize,
                               __tl_prev_persisting_l2_cache_size);
              __tl_prev_persisting_l2_cache_saved = false;
            }
            *ret = 0;
          });
}

} // namespace tl
} // namespace tvm
