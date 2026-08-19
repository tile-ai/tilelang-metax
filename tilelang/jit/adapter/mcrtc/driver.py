from __future__ import annotations

import ctypes

from tilelang.contrib.mcrtc import mcrtc


MC_SUCCESS = 0

mcrtc.mcGetErrorName.argtypes = [ctypes.c_int]
mcrtc.mcGetErrorName.restype = ctypes.c_char_p

mcrtc.mcGetErrorString.argtypes = [ctypes.c_int]
mcrtc.mcGetErrorString.restype = ctypes.c_char_p

mcrtc.mcModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
mcrtc.mcModuleLoadData.restype = ctypes.c_int

mcrtc.mcModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
mcrtc.mcModuleGetFunction.restype = ctypes.c_int

mcrtc.mcModuleUnload.argtypes = [ctypes.c_void_p]
mcrtc.mcModuleUnload.restype = ctypes.c_int

mcrtc.mcModuleLaunchKernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
]
mcrtc.mcModuleLaunchKernel.restype = ctypes.c_int


def get_error_string(result: int) -> str:
    name = mcrtc.mcGetErrorName(result)
    message = mcrtc.mcGetErrorString(result)
    name = name.decode("utf-8") if name else f"mcError({result})"
    message = message.decode("utf-8") if message else "Unknown MACA runtime error"
    return f"{name}: {message}"


def check_error(result: int, operation: str):
    if result != MC_SUCCESS:
        raise RuntimeError(f"{operation} failed: {get_error_string(result)}")


def load_module(bitcode: bytes) -> ctypes.c_void_p:
    module = ctypes.c_void_p()
    bitcode_buffer = (ctypes.c_char * len(bitcode)).from_buffer_copy(bitcode)
    result = mcrtc.mcModuleLoadData(ctypes.byref(module), ctypes.cast(bitcode_buffer, ctypes.c_void_p))
    check_error(result, "mcModuleLoadData")
    return module


def get_function(module: ctypes.c_void_p, function_name: str) -> ctypes.c_void_p:
    function = ctypes.c_void_p()
    result = mcrtc.mcModuleGetFunction(ctypes.byref(function), module, function_name.encode("utf-8"))
    check_error(result, "mcModuleGetFunction")
    return function


def unload_module(module: ctypes.c_void_p):
    result = mcrtc.mcModuleUnload(module)
    check_error(result, "mcModuleUnload")


def launch_kernel(
    function: ctypes.c_void_p,
    grid_dim: tuple[int, int, int],
    block_dim: tuple[int, int, int],
    shared_mem_bytes: int,
    stream: int | ctypes.c_void_p | None,
    args: list[ctypes._SimpleCData],
):
    """Launch a MACA kernel with ctypes-packed arguments."""
    grid_x, grid_y, grid_z = (int(value) for value in grid_dim)
    block_x, block_y, block_z = (int(value) for value in block_dim)
    stream_handle = stream if isinstance(stream, ctypes.c_void_p) else ctypes.c_void_p(stream or 0)

    arg_ptrs = (ctypes.c_void_p * len(args))()
    for index, arg in enumerate(args):
        arg_ptrs[index] = ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p)

    result = mcrtc.mcModuleLaunchKernel(
        function,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        int(shared_mem_bytes),
        stream_handle,
        arg_ptrs,
        None,
    )
    check_error(result, "mcModuleLaunchKernel")
