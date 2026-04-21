"""MACA-specific TVM FFI adapter."""

from __future__ import annotations

import torch
from tvm import runtime

from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter


class MACATVMFFIKernelAdapter(TVMFFIKernelAdapter):
    _FLOAT8_DTYPE_MAP = {
        getattr(torch, "float8_e4m3fn", None): "float8_e4m3fn",
        getattr(torch, "float8_e4m3fnuz", None): "float8_e4m3fnuz",
        getattr(torch, "float8_e5m2", None): "float8_e5m2",
        getattr(torch, "float8_e5m2fnuz", None): "float8_e5m2fnuz",
    }
    _FLOAT8_DTYPE_MAP = {k: v for k, v in _FLOAT8_DTYPE_MAP.items() if k is not None}

    def _adapt_tensor_for_runtime(self, arg):
        if not isinstance(arg, torch.Tensor):
            return arg

        float8_dtype = self._FLOAT8_DTYPE_MAP.get(arg.dtype)
        if float8_dtype is None:
            return arg

        return runtime.from_dlpack(torch.utils.dlpack.to_dlpack(arg.view(torch.int8)))._create_view(
            arg.shape,
            dtype=float8_dtype,
        )
