# todo: support prebuilt tvm

set(TVM_BUILD_FROM_SOURCE TRUE)
set(TVM_SOURCE ${CMAKE_SOURCE_DIR}/3rdparty/tvm)

if(DEFINED ENV{TVM_ROOT})
  if(EXISTS $ENV{TVM_ROOT}/cmake/config.cmake)
    set(TVM_SOURCE $ENV{TVM_ROOT})
    message(STATUS "Using TVM_ROOT from environment variable: ${TVM_SOURCE}")
  endif()
endif()

message(STATUS "Using TVM source: ${TVM_SOURCE}")

set(TVM_INCLUDES
  ${TVM_SOURCE}/include
  ${TVM_SOURCE}/src
  ${TVM_SOURCE}/3rdparty/dlpack/include
  ${TVM_SOURCE}/3rdparty/dmlc-core/include
)

if(EXISTS ${TVM_SOURCE}/ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/ffi/include)
elseif(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
endif()

if(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
endif()

# update 3rdparty/tvm/3rdparty/tvm-ffi for adding kDLMACA/kDLMACAHost
set(dlpack_header "${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack/dlpack.h")
file(READ "${dlpack_header}" FILE_CONTENTS)
if(NOT FILE_CONTENTS MATCHES ".*kDLMACA.*")
  string(REPLACE "} DLDeviceType;" "  kDLMACA = 19,\n  kDLMACAHost = 20,\n} DLDeviceType;" NEW_CONTENTS "${FILE_CONTENTS}")
  file(WRITE "${dlpack_header}" "${NEW_CONTENTS}")
endif()
set(ffi_core_pyi "${TVM_SOURCE}/3rdparty/tvm-ffi/python/tvm_ffi/core.pyi")
file(READ "${ffi_core_pyi}" FILE_CONTENTS)
if(NOT FILE_CONTENTS MATCHES ".*kDLMACA.*")
  string(REPLACE "kDLTrn = 17" "kDLTrn = 17\n    kDLMACA = 19\n    kDLMACAHost = 20" NEW_CONTENTS "${FILE_CONTENTS}")
  file(WRITE "${ffi_core_pyi}" "${NEW_CONTENTS}")
endif()
set(ffi_container_tensor "${TVM_SOURCE}/3rdparty/tvm-ffi/include/tvm/ffi/container/tensor.h")
file(READ "${ffi_container_tensor}" FILE_CONTENTS)
if(NOT FILE_CONTENTS MATCHES ".*kDLMACA.*")
  string(REPLACE "device.device_type == kDLROCMHost;" "device.device_type == kDLROCMHost ||\n         device.device_type == kDLMACA || device.device_type == kDLMACAHost;" NEW_CONTENTS "${FILE_CONTENTS}")
  file(WRITE "${ffi_container_tensor}" "${NEW_CONTENTS}")
endif()
set(ffi_cython_base_pxi "${TVM_SOURCE}/3rdparty/tvm-ffi/python/tvm_ffi/cython/base.pxi")
file(READ "${ffi_cython_base_pxi}" FILE_CONTENTS)
if(NOT FILE_CONTENTS MATCHES ".*kDLMACA.*")
  string(REPLACE "kDLTrn = 18" "kDLTrn = 18\n        kDLMACA = 19\n        kDLMACAHost = 20" NEW_CONTENTS "${FILE_CONTENTS}")
  file(WRITE "${ffi_cython_base_pxi}" "${NEW_CONTENTS}")
endif()
set(ffi_cython_device_pxi "${TVM_SOURCE}/3rdparty/tvm-ffi/python/tvm_ffi/cython/device.pxi")
file(READ "${ffi_cython_device_pxi}" FILE_CONTENTS)
if(NOT FILE_CONTENTS MATCHES ".*kDLMACA.*")
  string(REPLACE "kDLTrn = 17" "kDLTrn = 17\n    kDLMACA = 19\n    kDLMACAHost = 20" NEW_CONTENTS "${FILE_CONTENTS}")
  string(REPLACE "DLDeviceType.kDLTrn: \"trn\"," "DLDeviceType.kDLTrn: \"trn\",\n      DLDeviceType.kDLMACA: \"maca\",\n      DLDeviceType.kDLMACAHost: \"maca_host\"," NEW_CONTENTS "${NEW_CONTENTS}")
  string(REPLACE "\"trn\": DLDeviceType.kDLTrn," "\"trn\": DLDeviceType.kDLTrn,\n        \"maca\": DLDeviceType.kDLMACA," NEW_CONTENTS "${NEW_CONTENTS}")
  file(WRITE "${ffi_cython_device_pxi}" "${NEW_CONTENTS}")
endif()
# update 3rdparty/tvm for adding kDLMACA/kDLMACAHost
set(tvm_device_api_header "${TVM_SOURCE}/include/tvm/runtime/device_api.h")
file(READ "${tvm_device_api_header}" FILE_CONTENTS)
if (NOT FILE_CONTENTS MATCHES ".*kDLMACA.*")
  string(REPLACE "default:" "case kDLMACA:\n      return \"maca\";\n    case kDLMACAHost:\n      return \"maca_host\";\n    default:" NEW_CONTENTS "${FILE_CONTENTS}")
  file(WRITE "${tvm_device_api_header}" "${NEW_CONTENTS}")
endif()

#Automate bugfix for tvm_ffi/error.py (Resolve inner & outer cyclic references)
set(ffi_error_py "${TVM_SOURCE}/3rdparty/tvm-ffi/python/tvm_ffi/error.py")

if(EXISTS "${ffi_error_py}")
  file(READ "${ffi_error_py}" FILE_CONTENTS)
  set(MODIFIED FALSE)

  # --- Fix 1: Inner frame cycle in append_traceback ---
  if(NOT FILE_CONTENTS MATCHES "def create")
    message(STATUS "Auto-patching tvm_ffi/error.py: Fix inner frame cycle in append_traceback...")
    set(OLD_CODE_1
"        frame = self._create_frame(filename, lineno, func)
        return types.TracebackType(tb, frame, frame.f_lasti, lineno)"
    )
    set(NEW_CODE_1
"        def create(
            tb: types.TracebackType | None, frame: types.FrameType, lineno: int
        ) -> types.TracebackType:
            return types.TracebackType(tb, frame, frame.f_lasti, lineno)

        return create(tb, self._create_frame(filename, lineno, func), lineno)"
    )
    string(REPLACE "${OLD_CODE_1}" "${NEW_CODE_1}" FILE_CONTENTS "${FILE_CONTENTS}")
    set(MODIFIED TRUE)
  endif()

  # --- Fix 2: Outer cycle in _with_append_backtrace ---
  if(NOT FILE_CONTENTS MATCHES "del py_error, tb")
    message(STATUS "Auto-patching tvm_ffi/error.py: Fix outer cycle in _with_append_backtrace...")
    set(OLD_CODE_2
"    tb = py_error.__traceback__
    for filename, lineno, func in _parse_backtrace(backtrace):
        tb = _TRACEBACK_MANAGER.append_traceback(tb, filename, lineno, func)
    return py_error.with_traceback(tb)"
    )
    set(NEW_CODE_2
"    tb = py_error.__traceback__
    try:
        for filename, lineno, func in _parse_backtrace(backtrace):
            tb = _TRACEBACK_MANAGER.append_traceback(tb, filename, lineno, func)
        return py_error.with_traceback(tb)
    finally:
        del py_error, tb"
    )
    string(REPLACE "${OLD_CODE_2}" "${NEW_CODE_2}" FILE_CONTENTS "${FILE_CONTENTS}")
    set(MODIFIED TRUE)
  endif()

  if(MODIFIED)
    file(WRITE "${ffi_error_py}" "${FILE_CONTENTS}")
    message(STATUS "tvm_ffi/error.py memory leak patches applied successfully.")
  endif()
endif()
