import tilelang
import tilelang.language as T
import tilelang.testing
import pytest
from tilelang import tvm


@tilelang.testing.requires_cuda
def test_access_ptr_cp_async_codegen():
    """Smoke-test codegen for T.access_ptr -> tl.access_ptr -> tvm_access_ptr -> cp.async."""

    @T.prim_func
    def main(
        A: T.Tensor((64,), T.uint8),
        B: T.Tensor((64,), T.uint8),
    ):
        with T.Kernel(1, threads=4):
            S = T.alloc_shared((64,), T.uint8)
            bar = T.alloc_maca_barrier()
            T.maca_async_copy(
                A[16:32],
                S[8:24],
                barrier=bar,
            )
            T.maca_barrier_arrive_and_wait(bar)
            # Keep the shared buffer live so the pointers remain in generated code.
            B[0] = S[8]

    kernel = tilelang.compile(main, out_idx=[1], target="auto")
    src = kernel.get_kernel_source()
    print("=== access_ptr memcpy_async codegen ===")
    print(src)
    assert "memcpy_async<4>" in src, "Expected memcpy_async<4> in generated MACA source"


@tilelang.testing.requires_cuda
def test_vectorized_cp_async_num_elems_codegen():
    """check vectorized tl.ptx_cp_async widens logical element counts."""

    @T.prim_func
    def main(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            S = T.alloc_shared((64,), T.float16)
            bar = T.alloc_maca_barrier()
            for i in T.vectorized(4):
                T.maca_async_copy(A[i : i + 1], S[i : i + 1], barrier=bar)
            T.maca_barrier_arrive_and_wait(bar)
            B[0] = S[0]

    kernel = tilelang.compile(main, out_idx=[1], target="auto")
    src = kernel.get_kernel_source()
    print("=== vectorized memcpy_async codegen ===")
    print(src)
    assert "memcpy_async<8>" in src, "Expected vectorized cp.async bytes to fold into memcpy_async<8>"
    assert "memcpy_async<2>" not in src, "Did not expect scalar 2-byte memcpy_async bytes in generated MACA source"


@tilelang.testing.requires_cuda
def test_vectorized_int4_cp_async_num_elems_codegen():
    """Check subbyte tl.ptx_cp_async derives PTX bytes from logical element counts."""

    @T.prim_func
    def main(
        A: T.Tensor((128,), T.int4),
        B: T.Tensor((128,), T.int4),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((128,), T.int4)
            bar = T.alloc_maca_barrier()
            for i in T.vectorized(32):
                T.maca_async_copy(A[i : i + 1], S[i : i + 1], barrier=bar)
            T.maca_barrier_arrive_and_wait(bar)
            B[0] = S[0]

    kernel = tilelang.compile(main, out_idx=[1], target="auto")
    src = kernel.get_kernel_source()
    print("=== vectorized int4 memcpy_async codegen ===")
    print(src)
    assert "memcpy_async<16>" in src, "Expected 32 x int4 elems to fold into memcpy_async<16>"


@tilelang.testing.requires_cuda
def test_async_copy_tileop_lowers_to_cp_async():
    """Check T.async_copy always uses CPAsync path and does not auto-wait."""

    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((4,), T.float16)
            bar = T.alloc_maca_barrier()
            T.maca_async_copy(A[0:4], S, bar)
            T.maca_barrier_arrive_and_wait(bar)
            T.copy(S, B[0:4])

    kernel = tilelang.compile(main, out_idx=[1], target="auto")
    src = kernel.get_kernel_source()
    print("=== maca_async_copy -> memcpy_async codegen ===")
    print(src)
    assert "memcpy_async<8>" in src, "Expected T.maca_async_copy to lower to memcpy_async<8>"
    assert "tl::maca_barrier_arrive_and_wait<0>" not in src, "Did not expect maca_async_copy lowering to auto-emit wait"


@tilelang.testing.requires_cuda
def test_async_copy_tileop_rejects_invalid_cp_async_scope():
    """Check T.async_copy rejects non global->shared patterns."""

    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S0 = T.alloc_shared((4,), T.float16)
            S1 = T.alloc_shared((4,), T.float16)
            T.copy(A[0:4], S0)
            # shared->shared cannot use cp.async and should fail for async_copy.
            bar = T.alloc_maca_barrier()
            T.maca_async_copy(
                S0,
                S1,
                barrier=bar,
            )
            T.maca_barrier_arrive_and_wait(bar)
            T.copy(S1, B[0:4])

    with pytest.raises(
        tvm.error.InternalError,
        match=r"T\.maca_async_copy only supports global->shared/shared\.dyn copies",
    ):
        tilelang.compile(main, out_idx=[1], target="auto")


@tilelang.testing.requires_cuda
def test_parallel_simt_copy_respects_enable_async_copy_config():
    """Check `tl.enable_async_copy=False` disables auto cp.async rewriting."""

    @T.prim_func
    def main(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128,), T.float32)
            for i in T.Parallel(128):
                S[i] = A[i]
            B[0] = S[0]

    kernel = tilelang.compile(
        main,
        out_idx=[1],
        target="auto",
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: False},
    )
    src = kernel.get_kernel_source()
    print("=== Parallel SIMT copy (maca async disabled) codegen ===")
    print(src)
    assert "memcpy_async<" not in src, "Did not expect memcpy_async when async copy is disabled"


@tilelang.testing.requires_cuda
def test_async_copy_oob_lowers_to_predicated_cp_async_without_wait():
    """Check T.async_copy supports OOB via predicated cp.async and does not auto-wait."""
    M = 130
    K = 32
    block_m = 128
    block_k = 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, block_m)) as pid_m:
            S = T.alloc_shared((block_m, block_k), T.float16)
            bar = T.alloc_maca_barrier()
            T.maca_async_copy(A[pid_m * block_m : (pid_m + 1) * block_m, 0:block_k], S, barrier=bar)
            T.maca_barrier_arrive_and_wait(bar)
            # Don't read S here (no wait). Keep B live so kernel has an output.
            B[0, 0] = A[0, 0]

    kernel = tilelang.compile(main, out_idx=[1], target="auto")
    src = kernel.get_kernel_source()
    print("=== OOB maca_async_copy -> conditional codegen ===")
    print(src)
    assert "memcpy_async" in src, "Expected memcpy_async in generated MACA source"


if __name__ == "__main__":
    tilelang.testing.main()
