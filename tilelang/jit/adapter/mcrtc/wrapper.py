"""MCRTC Source Wrapper for TileLang.

Generates Python runtime code for launching MACA kernels compiled via MCRTC.
"""

from __future__ import annotations

import textwrap
from typing import Any, ClassVar

from tvm import IRModule
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit

from tilelang import tvm as tvm
from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper
from tilelang.jit.adapter.utils import match_declare_kernel, parse_function_call_args, pythonic_expr


PREDEFINED_HOST_FUNC_PY = """from tilelang.jit.adapter.mcrtc.driver import launch_kernel
import ctypes

_function_names = {0}

def call({1}):
{2}
"""


KERNEL_LAUNCH_FUNC_PY = """launch_kernel(
    kernels['{0}'],
    ({1}, {2}, {3}),
    ({4}, {5}, {6}),
    {7},
    stream,
    [{8}],
)
"""


class TLMCRTCSourceWrapper(TLCUDASourceWrapper):
    """MCRTC backend wrapper that generates Python kernel launch code."""

    _TYPE_MAP: ClassVar[dict[str, str]] = {
        "float32": "ctypes.c_float",
        "float16": "ctypes.c_uint16",
        "bfloat16": "ctypes.c_uint16",
        "float8_e4m3": "ctypes.c_uint8",
        "float8_e4m3fn": "ctypes.c_uint8",
        "float8_e5m2": "ctypes.c_uint8",
        "float64": "ctypes.c_double",
        "int64": "ctypes.c_int64",
        "int32": "ctypes.c_int32",
        "uint32": "ctypes.c_uint32",
        "bool": "ctypes.c_bool",
        "int8": "ctypes.c_int8",
        "uint8": "ctypes.c_uint8",
        "int16": "ctypes.c_int16",
        "uint16": "ctypes.c_uint16",
        "uchar": "ctypes.c_uint8",
    }

    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        super().__init__(scheduled_ir_module, source, target, device_mod, host_mod, pass_configs)

    def _pythonic_expr(self, expr: tvm.tir.PrimExpr) -> str:
        return pythonic_expr(expr, self._TYPE_MAP, ignore_cast=True, floor_div_op="//")

    def create_dispatch_func(self, code: str, function_informations: dict[str, dict]) -> str:
        """Generate a Python function that packs arguments and launches MACA kernels."""
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)
        function_args = [{"name": "kernels", "type": "dict"}]

        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append({"name": buffer.data.name, "type": "ctypes.c_void_p"})
            elif isinstance(param, tvm.tir.Var):
                function_args.append({"name": param.name, "type": self._lookup_type(param.dtype)})
            else:
                raise ValueError(f"Parameter {param} is not in the buffer map of the primary function.")

        for dyn_sym, dyn_sym_dtype in dynamic_symbolic_set:
            if dyn_sym not in [arg["name"] for arg in function_args]:
                function_args.append({"name": dyn_sym, "type": self._lookup_type(dyn_sym_dtype)})

        function_args.append(self.get_stream_type())
        def_args = ", ".join(arg["name"] for arg in function_args)
        kernel_launch_code = ""

        for function_name, function_info in function_informations.items():
            block_info = function_info["block_info"]
            grid_info = function_info["grid_info"]
            dynamic_smem_buf = function_info["dynamic_smem_buf"]
            function_params = function_info["function_params"]

            index = match_declare_kernel(code, function_name + "(")
            if index < 0:
                raise ValueError(f"Cannot find kernel declaration for {function_name}.")

            kernel_code = code[index:]
            semicolon_pos = kernel_code.find(";")
            brace_pos = kernel_code.find("{")

            if semicolon_pos >= 0 and (brace_pos < 0 or semicolon_pos < brace_pos):
                declaration = kernel_code[:semicolon_pos]
            else:
                declaration = kernel_code[:brace_pos]

            def transform_mcrtc_arg(name: str, arg_type: str):
                if arg_type == "ctypes.c_void_p":
                    return f"{name}.data_ptr()", arg_type
                return name, arg_type

            call_args = parse_function_call_args(
                declaration,
                function_args,
                function_params,
                {},
                {},
                transform_mcrtc_arg,
            )

            if len(call_args) != len(function_params):
                raise ValueError(
                    f"Kernel {function_name} expects {len(function_params)} arguments, "
                    f"but {len(call_args)} launch arguments were generated."
                )
            arg_values = ", ".join(f"{arg_type}({arg_name})" for arg_name, arg_type in call_args)
            shared_mem = self._pythonic_expr(dynamic_smem_buf) if dynamic_smem_buf is not None else "0"

            kernel_launch_code += KERNEL_LAUNCH_FUNC_PY.format(
                function_name,
                self._pythonic_expr(grid_info[0]),
                self._pythonic_expr(grid_info[1]),
                self._pythonic_expr(grid_info[2]),
                self._pythonic_expr(block_info[0]),
                self._pythonic_expr(block_info[1]),
                self._pythonic_expr(block_info[2]),
                shared_mem,
                arg_values,
            )

        body = kernel_launch_code.strip() or "pass"
        return PREDEFINED_HOST_FUNC_PY.format(
            repr(list(function_informations.keys())),
            def_args,
            textwrap.indent(body, "    "),
        )

    def update_lib_code(self, code: str):
        """Update MACA source code and generate the Python host dispatcher."""
        self.lib_code = code
        function_informations = {}

        for function_name in self.function_names:
            if function_name not in self.block_info or function_name not in self.grid_info:
                continue
            if self.device_mod is None:
                raise RuntimeError("Device IRModule is required to generate MCRTC launch code.")
            if function_name not in self.device_mod:
                raise ValueError(f"Function {function_name} was not found in the device module.")

            device_func = self.device_mod[function_name]
            kernel_params_count = len(device_func.params)
            function_params = None

            def visitor(node, fn=function_name, param_count=kernel_params_count):
                nonlocal function_params
                if not isinstance(node, tvm.tirx.Call):
                    return
                if not (hasattr(node, "op") and node.op == tvm.ir.Op.get("tirx.tvm_call_packed")):
                    return

                args = node.args
                if not args or args[0] != fn:
                    return
                if len(args) < 1 + param_count:
                    raise AssertionError(f"Kernel call for {fn} requires {param_count} arguments, but only {len(args) - 1} were found.")
                function_params = args[1 : 1 + param_count]

            post_order_visit(self.host_func.body, visitor)
            if function_params is None:
                raise ValueError(f"Cannot find packed call arguments for kernel {function_name}.")

            function_informations[function_name] = {
                "function_name": function_name,
                "block_info": self.block_info[function_name],
                "grid_info": self.grid_info[function_name],
                "dynamic_smem_buf": self.dynamic_smem_buf[function_name],
                "function_params": function_params,
            }

        self._host_func = self.create_dispatch_func(code, function_informations)
        return self.lib_code

    def wrap(self):
        return {
            "host_func": self._host_func,
            "function_names": self.function_names,
        }

    def get_stream_type(self) -> dict[str, str]:
        return {"name": "stream", "type": "int"}
