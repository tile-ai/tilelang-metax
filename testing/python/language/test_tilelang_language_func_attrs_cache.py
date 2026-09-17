"""Regression tests for compiling PrimFunc attrs from the disk cache."""

import torch

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.cache import _dispatch_map
from tilelang.env import env


@tilelang.testing.requires_cuda
def test_out_idx_via_attr_lazy_from_disk_cache(tmp_path, monkeypatch):
    """A cached executable should match the cold-compiled adapter contract."""

    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path / "tilelang_cache"))
    cache = _dispatch_map["tvm_ffi"]
    cache._memory_cache.clear()

    @T.prim_func
    def kernel(A):
        A: T.Tensor[[128, 128], T.float32]
        B = T.empty([128, 128], T.float32)
        with T.Kernel(1):
            for i in T.serial(128):
                for j in T.serial(128):
                    B[i, j] = A[i, j] + 1.0
        return B

    compiled = tilelang.compile(kernel)
    a = torch.randn(128, 128, device="cuda")
    torch.testing.assert_close(compiled(a), a + 1.0)
    compiled.adapter.get_exportable_executable().jit()

    cache._memory_cache.clear()
    cached = tilelang.compile(kernel)

    assert cached.artifact is None
    torch.testing.assert_close(cached(a), a + 1.0)
    cached.adapter.get_exportable_executable().jit()

    cache._memory_cache.clear()
