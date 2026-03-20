# pylint: disable=invalid-name
# modified from MetaX mcTVM python/tvm/contrib/mxcc.py
"""Utility to invoke mxcc compiler in the system"""

from __future__ import annotations

import re
import os
import subprocess
from tilelang.env import MACA_HOME, TILELANG_TEMPLATE_PATH
import tvm_ffi
from tilelang import tvm as tvm
from tvm.target import Target

from tvm.base import py_str
from tvm.contrib import utils


def compile_maca(code, target_format="mcbin", arch=None, options=None, path_target=None, verbose=False):
    """Compile maca code with MXCC from env.

    Parameters
    ----------
    code : str
        The maca code.

    target_format : str
        The target format of mxcc compiler.

    arch : str
        The maca architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    Return
    ------
    cubin : bytearray
        The bytearray of the fatbin
    """
    if arch is None:
        # If None, then it will use `tvm.target.Target.current().arch`.
        compute_version = get_target_compute_version(Target.current(allow_none=True))
        target_arch = get_target_arch(compute_version)
        arch = [f"--offload-arch={target_arch}"]

    temp = utils.tempdir()
    file_name = "tvm_kernels"
    if target_format not in ["mcbin", "mcir", "fatbin"]:
        raise ValueError("target_format must be in cubin, mcir, fatbin")
    temp_code = temp.relpath(f"{file_name}.maca")
    temp_target = temp.relpath(f"{file_name}.{target_format}")

    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    kernels_output_dir = pass_context.config.get("maca.kernels_output_dir", None)
    if kernels_output_dir is not None:
        if not os.path.isdir(kernels_output_dir):
            os.makedirs(kernels_output_dir)
        temp_code = os.path.join(kernels_output_dir, f"{file_name}.maca")
        temp_target = os.path.join(kernels_output_dir, f"{file_name}.{target_format}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd = [get_mxcc_compiler(), "-x", "maca"]
    if target_format == "mcbin":
        cmd.append("-device-obj")
    elif target_format == "mcir":
        cmd.extend(["-emit-llvm", "-maca-device-only"])
    else:
        cmd.append("-fatbin")
    # Always include line info for better profiling and mapping
    cmd += ["-O3", "-lineinfo"]
    if isinstance(arch, list):
        cmd += arch
    elif isinstance(arch, str):
        cmd += [f"--offload-arch={arch}"]

    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-D__FAST_HALF_CVT__"]
    cmd += ["-o", file_target]
    cmd += [temp_code]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if verbose:
        print(py_str(out))

    if proc.returncode != 0:
        msg = f"{code}\nCompilation error:\n{py_str(out)}\nCommand: {' '.join(cmd)}\n"
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


def default_compile_options(compile_flags: list[str] | None = None) -> list[str]:
    """
    Build a set of default MXCC compile options for TileLang generated sources.

    Includes C++ standard and common include paths (TileLang templates, MACA include).
    Merges user-provided compile flags if given.

    Parameters
    ----------
    compile_flags : Optional[List[str]]
        Additional flags to include. Items are split on whitespace.

    Returns
    -------
    List[str]
        A list of flags suitable for MXCC's command line.
    """
    options: list[str] = ["-std=c++17"]
    try:
        if TILELANG_TEMPLATE_PATH:
            options.append(f"-I{TILELANG_TEMPLATE_PATH}")
    except Exception:
        pass
    try:
        if MACA_HOME:
            options.append(f"-I{os.path.join(MACA_HOME, 'include')}")
    except Exception:
        pass

    # Preserve user flags exactly, including repeated tokens required by MXCC
    # (e.g., repeated "-Xcompiler" entries).
    if compile_flags:
        import shlex

        for flag in compile_flags:
            # Split each string like a shell would, preserving quoted args
            tokens = shlex.split(flag) if isinstance(flag, str) else [str(flag)]
            options.extend(tokens)
    return options


@tvm_ffi.register_global_func("tvm_callback_maca_get_arch")
def get_maca_arch(maca_path="/opt/maca"):
    """Utility function to get the MetaX GPU architecture

    Parameters
    ----------
    maca_path : str
        The path to maca installation directory

    Returns
    -------
    gpu_arch : str
        The MetaX GPU architecture
    """
    gpu_arch = "xcore1000"
    # check if maca is installed
    if not os.path.exists(maca_path):
        print("MACA not detected, using default xcore1000")
        return gpu_arch
    try:
        # Execute macainfo command
        macainfo_output = subprocess.check_output([f"{maca_path}/bin/macainfo"]).decode("utf-8")

        # Use regex to match the "Name" field
        match = re.search(r"Name:\s+(XCORE\d+[a-zA-Z]*)", macainfo_output)
        if match:
            gpu_arch = match.group(1)
        return gpu_arch.lower()
    except subprocess.CalledProcessError:
        print(
            f"Unable to execute macainfo command, \
                please ensure MACA is installed and you have an MetaX GPU on your system.\
                    using default {gpu_arch}."
        )
        return gpu_arch


def find_maca_path():
    """Utility function to find maca path

    Returns
    -------
    path : str
        Path to maca root.
    """
    if MACA_HOME:
        return MACA_HOME
    raise RuntimeError(
        "Failed to automatically detect MACA installation. Please set the MACA_PATH environment variable manually (e.g., export MACA_PATH=/opt/maca)."
    )


@tvm_ffi.register_global_func("tvm.contrib.mxcc.get_compute_version", override=True)
def get_target_compute_version(target=None):
    """Utility function to get compute capability of compilation target.

    Looks for the target arch in three different places, first in the target input, then the
    Target.current() scope, and finally the GPU device (if it exists).

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target

    Returns
    -------
    compute_version : str
        compute capability of a GPU
    """
    # 1. input target object
    # 2. Target.current()
    target = target or tvm.target.Target.current()
    if target and target.mcpu:
        arch = target.mcpu[5:]
        major = arch[:2]
        minor = arch[2:]
        if minor == "00":
            minor = "0"
        return major + "." + minor

    # 3. GPU compute version
    if tvm.device(tvm.ffi.DLDeviceType.kDLMACA, 0).exist:
        return tvm.device(tvm.ffi.DLDeviceType.kDLMACA, 0).compute_version

    raise ValueError(
        "No MACA architecture was specified or GPU detected.Try specifying it by adding '--offload-arch=xcorexxxx' to your target."
    )


def parse_compute_version(compute_version) -> tuple[int, int]:
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        # pylint: disable=raise-missing-from
        raise RuntimeError("Compute version parsing error") from err


def get_target_arch(compute_version: str | tuple[int, int]) -> str:
    if isinstance(compute_version, str):
        major, minor = parse_compute_version(compute_version)
    else:
        major, minor = compute_version
    target_arch = "xcore" + str(major * 100 + minor)
    return target_arch


def have_fp16(compute_version):
    """Either fp16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version: str
        compute capability of a GPU
    """
    return True


def have_int8(compute_version):
    """Either int8 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU
    """
    return True


def have_tensorcore(compute_version=None, target=None):
    """Either TensorCore support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str, optional
        compute capability of a GPU

    target : tvm.target.Target, optional
        The compilation target, will be used to determine arch if compute_version
        isn't specified.
    """
    return True


def have_macagraph():
    """Either MACA Graph support is provided"""
    return True


@tvm_ffi.register_global_func("tvm.contrib.mxcc.supports_bf16", override=True)
def have_bf16(compute_version):
    """Either bf16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU
    """
    return True


@tvm_ffi.register_global_func("tvm.contrib.mxcc.supports_fp8", override=True)
def have_fp8(compute_version):
    """Whether fp8 support is provided in the specified compute capability or not

    Parameters
    ----------
    compute_version : str
        GPU capability
    """
    return True


@tvm_ffi.register_global_func("tvm.contrib.mxcc.supports_tma", override=True)
def have_tma(target):
    """Whether TMA support is provided in the specified compute capability or not

    Parameters
    ----------
    target : tvm.target.Target
        The compilation target
    """
    return False


def have_pdl(target):
    return True


def get_mxcc_compiler() -> str:
    """Get the path to the mxcc compiler"""
    return os.path.join(find_maca_path(), "mxgpu_llvm", "bin", "mxcc")
