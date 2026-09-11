import tilelang
import tilelang.language as T

import pytest


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_MACA_ASYNC_COPY_SWIZZLE_STRATEGY: True,
    },
)
def matmul(M, N, K, block_M, block_N, block_K, num_threads=128, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Serial(T.ceildiv(K, block_K)):
                T.mxc_barrier_inst()
                T.async_copy(A[by * block_M, k * block_K], A_shared)
                T.async_copy(B[k * block_K, bx * block_N], B_shared)
                T.mxc_arrive_gvmcnt(0)
                T.mxc_barrier_inst()
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 64, 256)

    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    kernel = matmul(4096, 4096, 4096, 128, 128, 64, 256)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,num_threads",
    [
        (1024, 1024, 1024, 64, 64, 64, 128),
        (1024, 1024, 1024, 128, 128, 64, 128),
        (1024, 1024, 1024, 64, 64, 64, 256),
        (1024, 1024, 1024, 128, 128, 64, 256),
        (1024, 1024, 1024, 64, 64, 64, 512),
        (1024, 1024, 1024, 128, 128, 64, 512),
    ],
)
def test_gemm_with_async_copy_pipeline(M, N, K, block_M, block_N, block_K, num_threads):
    kernel = matmul(M, N, K, block_M, block_N, block_K, num_threads)

    import torch

    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()
    c = kernel(a, b)
    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


import itertools
from tilelang.autotuner import AutoTuner
import tilelang as tl


def get_configs():
    block_M = [64, 128, 256]
    block_N = [64, 120, 256]
    block_K = [64]
    num_threads = [256, 512]
    _configs = list(
        itertools.product(
            block_M,
            block_N,
            block_K,
            num_threads,
        )
    )

    configs = [
        {
            "M": 4096,
            "N": 4096,
            "K": 4096,
            "block_M": c[0],
            "block_N": c[1],
            "block_K": c[2],
            "num_threads": c[3],
        }
        for c in _configs
    ]
    return configs


def run_autotune():
    def matmul(M, N, K, block_M, block_N, block_K, num_threads=128, dtype=T.float16, accum_dtype=T.float32):
        @T.prim_func
        def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((M, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.use_swizzle(panel_size=0, enable=True)
                T.clear(C_local)
                for k in T.Serial(T.ceildiv(K, block_K)):
                    T.mxc_barrier_inst()
                    T.async_copy(A[by * block_M, k * block_K], A_shared)
                    T.async_copy(B[bx * block_N, k * block_K], B_shared)
                    T.mxc_arrive_gvmcnt(0)
                    T.mxc_barrier_inst()
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    def ref_program(A, B):
        return A @ B.T

    autotuner = (
        AutoTuner.from_kernel(kernel=matmul, configs=get_configs())
        .set_compile_args(
            out_idx=[-1],
            target="auto",
            pass_configs={
                tilelang.PassConfigKey.TL_ENABLE_MACA_ASYNC_COPY_SWIZZLE_STRATEGY: True,
            },
        )
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Auto,
            ref_prog=ref_program,
            skip_check=False,
            backend="event",
        )
    )
    result = autotuner.run(warmup=20, rep=200)
    print(result.config)
    print(result.latency)
    M, N, K = result.config["M"], result.config["N"], result.config["K"]
    print(f"TileLang Latency: {result.latency}")
    print(f"TileLang TFLOPS: {2 * M * N * K / result.latency * 1e-9}")


if __name__ == "__main__":
    main()
