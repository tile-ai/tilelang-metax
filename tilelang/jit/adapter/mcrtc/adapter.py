from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from tvm import tirx
from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.backend.target import determine_target
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter, CachedTextSource
from tilelang.jit.adapter.mcrtc import check_mcrtc_available
from tilelang.utils.language import retrieve_func_from_module

from .driver import get_function
from .libgen import MCRTCLibraryGenerator
from .wrapper import TLMCRTCSourceWrapper

logger = logging.getLogger(__name__)


class MCRTCKernelAdapter(BaseKernelAdapter):
    pymodule = None

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        check_mcrtc_available()

        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.device_kernel_source = device_kernel_source
        self.kernels = {}

        if isinstance(func_or_mod, tirx.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.param_dtypes = [param.torch_dtype() for param in params]
        self.param_shapes = []
        for param in params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tirx.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tirx.Var):
                    native_shape.append(dim)
                else:
                    native_shape.append(dim)
            self.param_shapes.append(native_shape)

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.target = Target(determine_target(target))
        self.verbose = verbose

        self.wrapper = TLMCRTCSourceWrapper(
            self.ir_module,
            device_kernel_source,
            self.target,
            device_mod=device_mod,
            host_mod=host_mod,
            pass_configs=pass_configs,
        )
        wrapper_result = self.wrapper.wrap()

        self.host_func = wrapper_result["host_func"]
        self.function_names = wrapper_result["function_names"]

        self.lib_generator = MCRTCLibraryGenerator(self.target, self.verbose)
        self.lib_generator.update_lib_code(self.device_kernel_source)
        self.lib_generator.update_host_func(self.host_func)
        self.lib_generator.assign_compile_flags(compile_flags)
        self.lib_generator.compile_lib()
        self.lib_generator.load_lib()

        self.libpath = self.lib_generator.libpath
        self.pymodule = self.lib_generator.pymodule

        module = self.lib_generator.module
        for name in self.function_names:
            self.kernels[name] = get_function(module, name)

        self._post_init()

    @classmethod
    def from_database(
        cls,
        params: list[KernelParam],
        result_idx: list[int],
        target: str,
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        host_kernel_source: CachedTextSource,
        device_kernel_source: CachedTextSource,
        kernel_lib_path: str,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        check_mcrtc_available()

        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.kernels = {}

        adapter._set_cached_text_source(
            "host_func",
            "_host_kernel_source_path",
            host_kernel_source,
        )
        adapter.host_kernel_source = host_kernel_source.text
        adapter._set_cached_text_source(
            "device_kernel_source",
            "_device_kernel_source_path",
            device_kernel_source,
        )

        if isinstance(func_or_mod, tirx.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        adapter.param_dtypes = [param.torch_dtype() for param in params]
        adapter.param_shapes = []
        for param in params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tirx.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tirx.Var):
                    native_shape.append(dim)
                else:
                    native_shape.append(dim)
            adapter.param_shapes.append(native_shape)

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()
        adapter.target = Target(determine_target(target))
        adapter.verbose = verbose

        adapter.lib_generator = MCRTCLibraryGenerator(adapter.target, adapter.verbose)
        adapter.lib_generator.assign_compile_flags(compile_flags)
        adapter.lib_generator.load_lib(lib_path=kernel_lib_path)

        adapter.libpath = kernel_lib_path
        adapter.pymodule = adapter.lib_generator.pymodule
        adapter.function_names = adapter.pymodule._function_names

        module = adapter.lib_generator.module
        for name in adapter.function_names:
            adapter.kernels[name] = get_function(module, name)

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self) -> dict[tirx.Var, tuple[int, int, int, int]]:
        """Extract runtime sources for scalar, shape, and stride symbols.

        Each entry contains ``(kind, parameter_index, dimension, scale)``.
        ``kind`` is 0 for a shape, 1 for a stride, and 2 for an explicit
        scalar parameter.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        self._dynamic_symbolic_candidates_map: dict[tirx.Var, list[tuple[int, int, int, int]]] = {}
        self._dynamic_symbolic_name_candidates_map: dict[str, list[tuple[int, int, int, int]]] = {}
        self._dynamic_symbolic_name_map: dict[str, tuple[int, int, int, int]] = {}

        def unique_push_back(v: tirx.Var, entry: tuple[int, int, int, int]):
            self._dynamic_symbolic_candidates_map.setdefault(v, []).append(entry)
            self._dynamic_symbolic_name_candidates_map.setdefault(v.name, []).append(entry)
            if v in dynamic_symbolic_map or v.name in self._dynamic_symbolic_name_map:
                return
            dynamic_symbolic_map[v] = entry
            self._dynamic_symbolic_name_map[v.name] = entry

        for i, param in enumerate(params):
            if param not in buffer_map:
                unique_push_back(param, (2, i, -1, 1))

        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tirx.Var):
                    unique_push_back(shape, (0, i, j, 1))

        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            element_bits = buffer.dtype.bits * buffer.dtype.lanes
            stride_scale = 8 // element_bits if element_bits < 8 else 1
            for j, stride in enumerate(buffer.strides):
                if isinstance(stride, tirx.Var):
                    unique_push_back(stride, (1, i, j, stride_scale))

        return dynamic_symbolic_map

    def _lookup_dynamic_symbolic_candidates(self, variable: tirx.Var) -> list[tuple[int, int, int, int]]:
        """Return all scalar, shape, and stride sources for a dynamic symbol."""
        if variable in self._dynamic_symbolic_candidates_map:
            return self._dynamic_symbolic_candidates_map[variable]
        if variable.name in self._dynamic_symbolic_name_candidates_map:
            return self._dynamic_symbolic_name_candidates_map[variable.name]
        raise KeyError(f"Dynamic symbolic variable '{variable.name}' was not found in symbolic map.")

    def _resolve_dynamic_symbolic_value(self, variable: tirx.Var, param_values: list[Any]):
        """Resolve one symbolic value from the first available runtime source."""
        unavailable_sources = []
        for ref_id, param_idx, dim_idx, stride_scale in self._lookup_dynamic_symbolic_candidates(variable):
            ref_value = param_values[param_idx]
            if ref_id == 2 and ref_value is not None:
                return ref_value
            if isinstance(ref_value, torch.Tensor):
                if ref_id == 0:
                    return ref_value.shape[dim_idx]
                if ref_id == 1:
                    return ref_value.stride()[dim_idx] * stride_scale
                raise ValueError(f"Unknown dynamic symbolic reference kind: {ref_id}")
            unavailable_sources.append(f"parameter {param_idx}: {type(ref_value).__name__}")
        details = ", ".join(unavailable_sources)
        raise TypeError(f"Dynamic symbolic variable '{variable.name}' has no available runtime source ({details})")

    def get_kernel_source(self, kernel_only: bool = True) -> str | None:
        """Return cached MACA kernel source or generated host source."""
        if kernel_only:
            return self._load_cached_text_source(
                "device_kernel_source",
                "_device_kernel_source_path",
            )
        return self._load_cached_text_source(
            "host_func",
            "_host_kernel_source_path",
        )

    def get_host_source(self) -> str | None:
        """Return generated Python launch source."""
        return self._load_cached_text_source(
            "host_func",
            "_host_kernel_source_path",
        )

    def _forward_from_prebuild_lib(self, *args, stream: int | None = None):
        """Invoke the generated Python launcher."""
        return self.pymodule.call(self.kernels, *args, stream=stream)

    def _wrap_forward_from_prebuild_lib(self, *ins: Any, stream: int | None = None):
        """Validate inputs, allocate outputs and launch the MACA kernel."""
        if len(ins) + len(self.result_idx) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} "
                f"with {len(ins)} inputs and {len(self.result_idx)} outputs."
            )

        input_index = 0
        param_values: list[Any] = [None] * len(self.params)
        for index in range(len(self.params)):
            if index in self.result_idx:
                continue
            param_values[index] = ins[input_index]
            input_index += 1

        first_tensor = next((value for value in param_values if isinstance(value, torch.Tensor)), None)

        # Allocate outputs in their PrimFunc parameter positions so symbolic
        # dimensions can resolve against both tensor and scalar inputs.
        for index in self.result_idx:
            dtype = self.param_dtypes[index]
            shape = []
            for dim in self.param_shapes[index]:
                if isinstance(dim, tirx.Var):
                    shape.append(self._resolve_dynamic_symbolic_value(dim, param_values))
                else:
                    shape.append(dim)

            device = first_tensor.device if first_tensor is not None else torch.cuda.current_device()
            param_values[index] = torch.empty(*shape, dtype=dtype, device=device)

        args = list(param_values)
        for symbol, (ref_id, _, _, _) in self.dynamic_symbolic_map.items():
            if ref_id != 2:
                args.append(self._resolve_dynamic_symbolic_value(symbol, param_values))

        if stream is None:
            if torch.cuda.is_available():
                stream = int(torch.cuda.current_stream().cuda_stream)
            else:
                stream = 0

        self._forward_from_prebuild_lib(*args, stream=stream)

        if len(self.result_idx) == 1:
            return args[self.result_idx[0]]
        return [args[index] for index in self.result_idx]

    def _convert_torch_func(
        self,
    ) -> Callable[..., torch.Tensor | list[torch.Tensor]]:
        """Return a PyTorch-compatible callable."""
        return self._wrap_forward_from_prebuild_lib

    @property
    def prim_func(self) -> tirx.PrimFunc:
        """Return the primary TIR function from the IRModule."""
        return retrieve_func_from_module(self.ir_module)
