from __future__ import annotations
from pathlib import Path

import ctypes
import os


MCRTC_SUCCESS = 0


def _load_mcrtc_library() -> ctypes.CDLL:
    maca_path = os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME")
    lib_path = os.path.join(maca_path, "lib", "libmcruntime.so") if maca_path else "libmcruntime.so"

    try:
        return ctypes.CDLL(lib_path)
    except OSError as error:
        raise ImportError(f"Failed to load MCRTC library: {lib_path}") from error


mcrtc = _load_mcrtc_library()

mcrtc.mcrtcGetErrorString.argtypes = [ctypes.c_int]
mcrtc.mcrtcGetErrorString.restype = ctypes.c_char_p

mcrtc.mcrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
mcrtc.mcrtcVersion.restype = ctypes.c_int

mcrtc.mcrtcCreateProgram.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]
mcrtc.mcrtcCreateProgram.restype = ctypes.c_int

mcrtc.mcrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
mcrtc.mcrtcCompileProgram.restype = ctypes.c_int

mcrtc.mcrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
mcrtc.mcrtcGetProgramLogSize.restype = ctypes.c_int

mcrtc.mcrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
mcrtc.mcrtcGetProgramLog.restype = ctypes.c_int

mcrtc.mcrtcGetBitcodeSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
mcrtc.mcrtcGetBitcodeSize.restype = ctypes.c_int

mcrtc.mcrtcGetBitcode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
mcrtc.mcrtcGetBitcode.restype = ctypes.c_int

mcrtc.mcrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
mcrtc.mcrtcDestroyProgram.restype = ctypes.c_int


def get_mcrtc_error_string(result: int) -> str:
    message = mcrtc.mcrtcGetErrorString(result)
    return message.decode("utf-8") if message else f"Unknown MCRTC error: {result}"


def get_mcrtc_version() -> tuple[int, int]:
    major = ctypes.c_int()
    minor = ctypes.c_int()
    result = mcrtc.mcrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    assert result == MCRTC_SUCCESS, f"Failed to get MCRTC version: {get_mcrtc_error_string(result)}"
    return major.value, minor.value


def compile_maca(
    code: str,
    options: str | list[str] | None = None,
    verbose: bool = False,
) -> bytes:
    """Compile MACA code with MCRTC."""

    file_name = "tvm_kernels"
    tilelang_include = Path(__file__).resolve().parents[2] / "src"
    maca_include = Path(os.environ.get("MACA_PATH", "/opt/maca")) / "include"
    final_options = ["-std=c++17", f"-I{tilelang_include}", f"-I{maca_include}"]

    if options:
        if isinstance(options, str):
            final_options.append(options)
        elif isinstance(options, list):
            final_options.extend(options)
        else:
            raise ValueError("options must be str or list of str")

    code_bytes = bytes(code, "utf-8")
    program = ctypes.c_void_p()

    result = mcrtc.mcrtcCreateProgram(ctypes.byref(program), code_bytes, bytes(file_name, "utf-8"), 0, None, None)
    assert result == MCRTC_SUCCESS, f"Failed to create program: {get_mcrtc_error_string(result)}"

    try:
        options_bytes = [bytes(option, "utf-8") for option in final_options]
        options_array = (ctypes.c_char_p * len(options_bytes))(*options_bytes) if options_bytes else None
        compile_result = mcrtc.mcrtcCompileProgram(program, len(options_bytes), options_array)

        if compile_result != MCRTC_SUCCESS:
            msg = f"{code}\nCompilation error:\n"
            if verbose:
                log_size = ctypes.c_size_t()
                result = mcrtc.mcrtcGetProgramLogSize(program, ctypes.byref(log_size))
                assert result == MCRTC_SUCCESS, f"Failed to get program log size: {get_mcrtc_error_string(result)}"
                log_buffer = ctypes.create_string_buffer(log_size.value)
                result = mcrtc.mcrtcGetProgramLog(program, log_buffer)
                assert result == MCRTC_SUCCESS, f"Failed to get program log: {get_mcrtc_error_string(result)}"
                msg += f"{log_buffer.value.decode('utf-8')}\n"
            else:
                msg += "Turn on verbose to see the full compilation log."
            msg += f"\nOptions: {' '.join(final_options)}\n"
            raise RuntimeError(msg)

        bitcode_size = ctypes.c_size_t()
        result = mcrtc.mcrtcGetBitcodeSize(program, ctypes.byref(bitcode_size))
        assert result == MCRTC_SUCCESS, f"Failed to get bitcode size: {get_mcrtc_error_string(result)}"

        result_buffer = ctypes.create_string_buffer(bitcode_size.value)
        result = mcrtc.mcrtcGetBitcode(program, result_buffer)
        assert result == MCRTC_SUCCESS, f"Failed to get bitcode: {get_mcrtc_error_string(result)}"

        return result_buffer.raw

    finally:
        result = mcrtc.mcrtcDestroyProgram(ctypes.byref(program))
        assert result == MCRTC_SUCCESS, f"Failed to destroy program: {get_mcrtc_error_string(result)}"
