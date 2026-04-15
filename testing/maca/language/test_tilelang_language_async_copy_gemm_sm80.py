import tilelang
import tilelang.language as T
import tilelang.testing
import torch


def test_copy_and_async_copy_gemm_codegen_adapted_to_maca():
    """Keep the original GEMM codegen comparison shape, compiled for MACA.

    On CUDA SM80, the original test expects normal T.copy(global->shared) to
    lower to cp.async and match explicit T.async_copy. On MACA, automatic
    T.copy upgrade is intentionally disabled for these GEMM layouts because
    memcpy_async requires a uniform shared-memory base address. Explicit
    T.async_copy still lowers through the MACA memcpy_async wrapper.
    """

    M = 256
    N = 256
    K = 128
    block_M = 128
    block_N = 128
    block_K = 32

    @T.prim_func
    def matmul_relu_kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.async_copy(A[by * block_M, ko * block_K], A_shared)
                T.ptx_wait_group(0)

                T.async_copy(B[ko * block_K, bx * block_N], B_shared)
                T.ptx_wait_group(0)

                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    async_matmul_relu = matmul_relu_kernel

    @T.prim_func
    def matmul_relu_kernel(  # noqa: F811
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    sync_matmul_relu = matmul_relu_kernel

    async_kernel = tilelang.compile(async_matmul_relu, target="maca")
    sync_kernel = tilelang.compile(sync_matmul_relu, target="maca")

    async_src = async_kernel.get_kernel_source()
    sync_src = sync_kernel.get_kernel_source()

    assert "tl::cp_async_gs<" in async_src
    assert "tl::cp_async_commit" in async_src
    assert "tl::cp_async_wait_token" in async_src
    assert "tl::cp_async_gs<" not in sync_src


def test_maca_memcpy_async_parallel_correctness():
    block = 128
    n = 1024

    @T.prim_func
    def copy_kernel(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            S = T.alloc_shared((block,), T.float32)
            T.async_copy(A[bx * block], S)
            T.ptx_wait_group(0)
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    B[idx] = S[i] + T.float32(1.0)

    kernel = tilelang.compile(copy_kernel, target="maca", out_idx=[1])
    source = kernel.get_kernel_source()
    assert "tl::cp_async_gs<" in source
    assert "tl::cp_async_wait_token" in source

    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = kernel(a)

    torch.testing.assert_close(b, a + 1, rtol=0, atol=0)


def test_maca_memcpy_async_pipeline_correctness():
    block = 128
    tiles = 4
    n = block * tiles

    @T.prim_func
    def copy_kernel(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(1, threads=block):
            S = T.alloc_shared((block,), T.float32)
            for ko in T.Pipelined(tiles, num_stages=2):
                T.async_copy(A[ko * block], S)
                T.ptx_wait_group(0)
                for i in T.Parallel(block):
                    B[ko * block + i] = S[i] + T.float32(2.0)

    kernel = tilelang.compile(copy_kernel, target="maca", out_idx=[1])
    source = kernel.get_kernel_source()
    assert "tl::cp_async_gs<" in source
    assert "tl::cp_async_wait_token" in source

    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = kernel(a)

    torch.testing.assert_close(b, a + 2, rtol=0, atol=0)


def test_maca_memcpy_async_predicated_zero_fill_correctness():
    @T.prim_func
    def copy_kernel(
        A: T.Tensor((8,), T.float16),
        B: T.Tensor((16,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((16,), T.float16)
            T.fill(S, 0)
            T.ptx_cp_async(
                T.access_ptr(S[0], "w", 16),
                T.access_ptr(A[0], "r", 16),
                16,
                True,
            )
            T.ptx_cp_async(
                T.access_ptr(S[8], "w", 16),
                T.access_ptr(A[0], "r", 16),
                16,
                False,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.copy(S, B[0:16])

    kernel = tilelang.compile(copy_kernel, target="maca", out_idx=[1])
    source = kernel.get_kernel_source()
    assert "tl::cp_async_gs_conditional<16>" in source
    assert "tl::cp_async_wait_token" in source

    a = torch.randn((8,), dtype=torch.float16, device="cuda")
    b = kernel(a)
    expected = torch.zeros((16,), dtype=torch.float16, device="cuda")
    expected[:8] = a

    torch.testing.assert_close(b, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
