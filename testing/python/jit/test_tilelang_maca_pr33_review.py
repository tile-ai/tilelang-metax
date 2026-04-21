from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import tilelang.language as T
from tilelang import tvm
from tilelang.contrib import dlpack as tl_dlpack
from tilelang.engine.param import KernelParam
from tilelang.intrinsics.maca_mma_macro_generator import TensorCoreIntrinEmitter
from tilelang.jit import adapter as jit_adapter
from tilelang.jit import kernel as jit_kernel
from tilelang.jit.adapter import maca_tvm_ffi as maca_tvm_ffi_mod
from tilelang.jit.adapter import tvm_ffi as tvm_ffi_mod


def _make_fake_prim_func(dtype_str: str):
    param_token = object()
    return SimpleNamespace(
        params=[param_token],
        buffer_map={
            param_token: SimpleNamespace(
                dtype=tvm.DataType(dtype_str),
                shape=[],
                strides=[],
            )
        },
    )


def _make_fake_adapter(adapter_cls, monkeypatch, dtype_str: str):
    fake_prim_func = _make_fake_prim_func(dtype_str)
    monkeypatch.setattr(adapter_cls, "prim_func", property(lambda self: fake_prim_func))
    monkeypatch.setattr(adapter_cls, "_process_dynamic_symbolic", lambda self: {})

    adapter = adapter_cls.__new__(adapter_cls)
    adapter.params = [KernelParam(dtype=tvm.DataType(dtype_str), shape=[4])]
    adapter.result_idx = []
    adapter.executable = lambda *args: args
    adapter.rt_mod = None
    adapter.ir_module = None
    return adapter


@pytest.mark.skipif(not hasattr(torch, "float8_e5m2fnuz"), reason="Torch build lacks float8 fnuz support")
def test_dlpack_keeps_original_float8_logical_dtype_mapping(monkeypatch):
    class FakeArray:
        def __init__(self, token):
            self.token = token

        def _create_view(self, shape, dtype):
            return {"shape": tuple(shape), "dtype": dtype, "token": self.token}

    monkeypatch.setattr(tl_dlpack.runtime, "from_dlpack", lambda token: FakeArray(token))

    wrapped = tl_dlpack.convert_func(
        lambda arg: arg,
        torch.Tensor,
        lambda tensor: ("dlpack", tensor.dtype),
    )

    result = wrapped(torch.zeros((4,), dtype=torch.float8_e5m2fnuz))

    assert result["dtype"] == "float8_e5m2"


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fnuz"), reason="Torch build lacks float8 fnuz support")
def test_generic_tvm_ffi_keeps_float8_passthrough(monkeypatch):
    adapter = _make_fake_adapter(tvm_ffi_mod.TVMFFIKernelAdapter, monkeypatch, "float8_e4m3fnuz")
    tensor = torch.zeros((4,), dtype=torch.float8_e4m3fnuz)
    seen = []

    adapter.executable = lambda *args: seen.extend(args)
    monkeypatch.setattr(
        tvm_ffi_mod.runtime,
        "from_dlpack",
        lambda *_args, **_kwargs: pytest.fail("generic TVMFFIKernelAdapter should not adapt float8 tensors"),
    )

    func = adapter._convert_torch_func()
    func(tensor)

    assert seen == [tensor]


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Torch build lacks float8 fn support")
def test_maca_tvm_ffi_wraps_float8_with_maca_logical_dtype_mapping(monkeypatch):
    maca_adapter_cls = getattr(jit_adapter, "MACATVMFFIKernelAdapter", None)
    assert maca_adapter_cls is not None

    adapter = _make_fake_adapter(maca_adapter_cls, monkeypatch, "float8_e4m3fn")
    tensor = torch.zeros((4,), dtype=torch.float8_e4m3fn)

    class FakeArray:
        def __init__(self, token):
            self.token = token

        def _create_view(self, shape, dtype):
            return {"shape": tuple(shape), "dtype": dtype, "token": self.token}

    seen = []
    adapter.executable = lambda *args: seen.extend(args)
    monkeypatch.setattr(maca_tvm_ffi_mod.runtime, "from_dlpack", lambda token: FakeArray(token))

    func = adapter._convert_torch_func()
    func(tensor)

    assert seen[0]["dtype"] == "float8_e4m3fn"


def test_jit_kernel_routes_maca_tvm_ffi_through_target_specific_adapter():
    adapter_cls = jit_kernel._get_tvm_ffi_adapter_cls(tvm.target.Target("maca"))

    assert adapter_cls is getattr(jit_adapter, "MACATVMFFIKernelAdapter", None)
    assert jit_kernel._get_tvm_ffi_adapter_cls(tvm.target.Target("cuda")) is tvm_ffi_mod.TVMFFIKernelAdapter


def test_maca_mma_dtype_table_carries_storage_and_mma_metadata():
    fp8_info = TensorCoreIntrinEmitter.dtype_abbrv["float8_e4m3fnuz"]

    assert fp8_info["storage"] == "e4m3fnuz"
    assert fp8_info["mma"] == "fp8"
    assert fp8_info["mma_input_dtype"] == T.float16

    emitter = TensorCoreIntrinEmitter(
        a_dtype="float8_e4m3fnuz",
        b_dtype="float8_e4m3fnuz",
        accum_dtype="float32",
    )
    assert emitter.mma_input_dtype == T.float16
    assert emitter.mma_suffix == "16x16x16fp8"
