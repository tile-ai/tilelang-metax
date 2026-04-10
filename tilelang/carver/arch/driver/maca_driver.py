from __future__ import annotations
import ctypes
import sys

try:
    import torch.cuda._CudaDeviceProperties as _MacaDeviceProperties
except ImportError:
    _MacaDeviceProperties = type("DummyMacaDeviceProperties", (), {})


class macaDeviceAttrNames:
    r"""
    refer to MetaX MACA / MC Runtime API definitions.
    """

    macaDevAttrMaxThreadsPerBlock: int = 1
    macaDevAttrMaxSharedMemoryPerBlock: int = 8
    macaDevAttrMaxRegistersPerBlock: int = 12
    macaDevAttrMultiProcessorCount: int = 16
    macaDevAttrMaxSharedMemoryPerMultiprocessor: int = 81
    macaDevAttrMaxPersistingL2CacheSize: int = 108


def get_maca_device_properties(device_id: int = 0) -> _MacaDeviceProperties | None:
    try:
        import torch.cuda

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(torch.device(device_id))
    except ImportError:
        return None


def get_device_name(device_id: int = 0) -> str | None:
    prop = get_maca_device_properties(device_id)
    if prop:
        return prop.name


def get_shared_memory_per_block(device_id: int = 0, format: str = "bytes") -> int | None:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    shared_mem = get_device_attribute(macaDeviceAttrNames.macaDevAttrMaxSharedMemoryPerBlock, device_id)
    if shared_mem is None:
        raise RuntimeError("Failed to get device properties via libmcruntime.so.")

    shared_mem = int(shared_mem)
    if format == "bytes":
        return shared_mem
    elif format == "kb":
        return shared_mem // 1024
    elif format == "mb":
        return shared_mem // (1024 * 1024)
    else:
        raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def get_device_attribute(attr: int, device_id: int = 0) -> int | None:
    try:
        if sys.platform == "win32":
            libmacart = ctypes.windll.LoadLibrary("mcruntime64.dll")
        else:
            libmacart = ctypes.cdll.LoadLibrary("libmcruntime.so")

        value = ctypes.c_int()

        mcDeviceGetAttribute = libmacart.mcDeviceGetAttribute
        mcDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        mcDeviceGetAttribute.restype = ctypes.c_int

        ret = mcDeviceGetAttribute(ctypes.byref(value), attr, device_id)
        if ret != 0:
            raise RuntimeError(f"mcDeviceGetAttribute failed with error {ret}")

        return value.value
    except Exception as e:
        print(f"Error getting device attribute: {str(e)}")
        return None


def get_max_dynamic_shared_size_bytes(device_id: int = 0, format: str = "bytes") -> int | None:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    shared_mem = get_device_attribute(macaDeviceAttrNames.macaDevAttrMaxSharedMemoryPerMultiprocessor, device_id)
    if shared_mem is None:
        return None
    if format == "bytes":
        return shared_mem
    elif format == "kb":
        return shared_mem // 1024
    elif format == "mb":
        return shared_mem // (1024 * 1024)
    else:
        raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def get_persisting_l2_cache_max_size(device_id: int = 0) -> int | None:
    prop = get_device_attribute(macaDeviceAttrNames.macaDevAttrMaxPersistingL2CacheSize, device_id)
    return prop


def get_num_sms(device_id: int = 0) -> int:
    num_sms = get_device_attribute(macaDeviceAttrNames.macaDevAttrMultiProcessorCount, device_id)
    if num_sms is None:
        raise RuntimeError("Failed to get device properties via libmcruntime.so.")
    return num_sms


def get_registers_per_block(device_id: int = 0) -> int | None:
    prop = get_device_attribute(
        macaDeviceAttrNames.macaDevAttrMaxRegistersPerBlock,
        device_id,
    )
    return prop
