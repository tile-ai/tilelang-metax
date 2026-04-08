import tilelang.testing
from tilelang.carver.arch.driver.maca_driver import (
    get_maca_device_properties,
    get_device_name,
    get_shared_memory_per_block,
    get_device_attribute,
    get_max_dynamic_shared_size_bytes,
    get_persisting_l2_cache_max_size,
    get_num_sms,
    get_registers_per_block,
    macaDeviceAttrNames as _macaDeviceAttrNames,
)
import torch


def test_driver_get_device_properties():
    prop = get_maca_device_properties()
    assert prop is not None, "Failed to get MACA device properties"
    assert isinstance(prop, torch.cuda._CudaDeviceProperties), "Returned object is not of type _macaDeviceProperties"


def test_device_get_device_name():
    tl_device_name = get_device_name()
    th_device_name = torch.cuda.get_device_name()
    assert tl_device_name == th_device_name, "Device names do not match"


def test_device_get_shared_memory_per_block():
    tl_smem = get_shared_memory_per_block()
    driver_smem = get_device_attribute(_macaDeviceAttrNames.macaDevAttrMaxSharedMemoryPerBlock)
    assert tl_smem == driver_smem, "Shared memory per block values do not match"


def test_device_get_persisting_l2_cache_size():
    tl_cache_size = get_persisting_l2_cache_max_size()
    driver_cache_size = get_device_attribute(_macaDeviceAttrNames.macaDevAttrMaxPersistingL2CacheSize)
    assert tl_cache_size == driver_cache_size, "Persisting L2 cache size values do not match"


def test_device_get_num_sms():
    tl_num_sms = get_num_sms()
    driver_num_sms = get_device_attribute(_macaDeviceAttrNames.macaDevAttrMultiProcessorCount)
    assert tl_num_sms == driver_num_sms, "Number of SMs do not match"


def test_device_get_registers_per_block():
    tl_regs_per_block = get_registers_per_block()
    driver_regs_per_block = get_device_attribute(_macaDeviceAttrNames.macaDevAttrMaxRegistersPerBlock)
    assert tl_regs_per_block == driver_regs_per_block, "Registers per block values do not match"


def test_device_get_max_dynamic_shared_size_bytes():
    tl_dynamic_smem = get_max_dynamic_shared_size_bytes()
    driver_dynamic_smem = get_device_attribute(_macaDeviceAttrNames.macaDevAttrMaxSharedMemoryPerMultiprocessor)
    assert tl_dynamic_smem == driver_dynamic_smem, "Max dynamic shared size bytes values do not match"


if __name__ == "__main__":
    tilelang.testing.main()
