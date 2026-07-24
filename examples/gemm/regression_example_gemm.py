import argparse
import os
from contextlib import contextmanager

import tilelang
import tilelang.language as T
import tilelang.testing
import example_gemm
import example_gemm_intrinsics

_BENCH_GEMM_CONFIG = {
    "block_M": 128,
    "block_N": 128,
    "block_K": 128,
    "threads": 256,
    "num_stages": 0,
}

_MACA_TEMPLATE_ENV = {
    "TILELANG_DEFAULT_TARGET": "maca",
    "TILELANG_MACA_GEMM_USE_TEMPLATE": "1",
    "TILELANG_MACA_GEMM_K_PACK": "1",
    "TILELANG_MACA_GEMM_CONSUMER_SURFACE": None,
}

_MACA_BASELINE_ENV = {
    "TILELANG_DEFAULT_TARGET": "maca",
    "TILELANG_MACA_GEMM_USE_TEMPLATE": None,
    "TILELANG_MACA_GEMM_K_PACK": None,
    "TILELANG_MACA_GEMM_CONSUMER_SURFACE": None,
}

_MACA_KPACK2_ENV = {
    "TILELANG_DEFAULT_TARGET": "maca",
    "TILELANG_MACA_GEMM_USE_TEMPLATE": None,
    "TILELANG_MACA_GEMM_K_PACK": "2",
    "TILELANG_MACA_GEMM_CONSUMER_SURFACE": None,
}

_BENCH_GEMM_CASES = (
    {"name": "bench_gemm_m1664_n1024_k262144", "M": 1664, "N": 1024, "K": 262144},
    {"name": "bench_gemm_m4096_n1024_k8192", "M": 4096, "N": 1024, "K": 8192},
    {"name": "bench_gemm_m4096_n8192_k8192", "M": 4096, "N": 8192, "K": 8192},
    {"name": "bench_gemm_m4096_n28672_k8192", "M": 4096, "N": 28672, "K": 8192},
    {"name": "bench_gemm_m4096_n8192_k28672", "M": 4096, "N": 8192, "K": 28672},
    {"name": "bench_gemm_m8192_n1024_k8192", "M": 8192, "N": 1024, "K": 8192},
    {"name": "bench_gemm_m8192_n8192_k8192", "M": 8192, "N": 8192, "K": 8192},
    {"name": "bench_gemm_m8192_n28672_k8192", "M": 8192, "N": 28672, "K": 8192},
    {"name": "bench_gemm_m8192_n8192_k28672", "M": 8192, "N": 8192, "K": 28672},
)

_BENCH_GEMM_MACA_TEMPLATE_CASE = {
    "name": "bench_gemm_maca_template_m1664_n1024_k262144",
    "M": 1664,
    "N": 1024,
    "K": 262144,
}

_BENCH_GEMM_MACA_BASELINE_CASE = {
    "name": "bench_gemm_maca_baseline_m1664_n1024_k262144",
    "M": 1664,
    "N": 1024,
    "K": 262144,
}

_BENCH_GEMM_MACA_KPACK2_CASE = {
    "name": "bench_gemm_maca_kpack2_m1664_n1024_k262144",
    "M": 1664,
    "N": 1024,
    "K": 262144,
}

_MACA_GEMM_CHECK_SHAPE = (128, 128, 128)


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    old_values = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@tilelang.jit(out_idx=[-1])
def _bench_gemm_matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    num_stages,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


@tilelang.jit(out_idx=[-1])
def _bench_gemm_maca_template_matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    num_stages,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


@tilelang.jit(out_idx=[-1])
def _bench_gemm_maca_kpack2_matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    num_stages,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def _run_bench_gemm(M, N, K, block_M, block_N, block_K, threads, num_stages):
    with _temporary_env(_MACA_BASELINE_ENV):
        kernel = _bench_gemm_matmul(M, N, K, block_M, block_N, block_K, threads, num_stages)
        profiler = kernel.get_profiler()
        return profiler.do_bench(backend="cupti")


def _run_bench_gemm_maca_baseline(M, N, K, block_M, block_N, block_K, threads, num_stages):
    with _temporary_env(_MACA_BASELINE_ENV):
        kernel = _bench_gemm_matmul(M, N, K, block_M, block_N, block_K, threads, num_stages)
        profiler = kernel.get_profiler()
        return profiler.do_bench(backend="cupti")


def _run_bench_gemm_maca_template(M, N, K, block_M, block_N, block_K, threads, num_stages):
    with _temporary_env(_MACA_TEMPLATE_ENV):
        kernel = _bench_gemm_maca_template_matmul(M, N, K, block_M, block_N, block_K, threads, num_stages)
        profiler = kernel.get_profiler()
        return profiler.do_bench(backend="cupti")


def _run_bench_gemm_maca_kpack2(M, N, K, block_M, block_N, block_K, threads, num_stages):
    with _temporary_env(_MACA_KPACK2_ENV):
        kernel = _bench_gemm_maca_kpack2_matmul(M, N, K, block_M, block_N, block_K, threads, num_stages)
        profiler = kernel.get_profiler()
        return profiler.do_bench(backend="cupti")


def _check_bench_gemm_maca_pair() -> None:
    M, N, K = _MACA_GEMM_CHECK_SHAPE
    cases = (
        (_MACA_BASELINE_ENV, _bench_gemm_matmul),
        (_MACA_TEMPLATE_ENV, _bench_gemm_maca_template_matmul),
    )
    for env, factory in cases:
        with _temporary_env(env):
            kernel = factory(M, N, K, **_BENCH_GEMM_CONFIG)
            kernel.get_profiler().assert_allclose(lambda a, b: a @ b, atol=1e-2, rtol=1e-2)


def _check_bench_gemm_maca_kpack2_pair() -> None:
    M, N, K = _MACA_GEMM_CHECK_SHAPE
    cases = (
        (_MACA_BASELINE_ENV, _bench_gemm_matmul),
        (_MACA_KPACK2_ENV, _bench_gemm_maca_kpack2_matmul),
    )
    for env, factory in cases:
        with _temporary_env(env):
            kernel = factory(M, N, K, **_BENCH_GEMM_CONFIG)
            kernel.get_profiler().assert_allclose(lambda a, b: a @ b, atol=1e-2, rtol=1e-2)


def _process_bench_gemm_case(case):
    tilelang.testing.process_func(
        _run_bench_gemm,
        case["name"],
        M=case["M"],
        N=case["N"],
        K=case["K"],
        **_BENCH_GEMM_CONFIG,
    )


def _process_bench_gemm_maca_template_case(case):
    tilelang.testing.process_func(
        _run_bench_gemm_maca_template,
        case["name"],
        M=case["M"],
        N=case["N"],
        K=case["K"],
        **_BENCH_GEMM_CONFIG,
    )


def _process_bench_gemm_maca_baseline_case(case):
    tilelang.testing.process_func(
        _run_bench_gemm_maca_baseline,
        case["name"],
        M=case["M"],
        N=case["N"],
        K=case["K"],
        **_BENCH_GEMM_CONFIG,
    )


def _process_bench_gemm_maca_kpack2_case(case):
    tilelang.testing.process_func(
        _run_bench_gemm_maca_kpack2,
        case["name"],
        M=case["M"],
        N=case["N"],
        K=case["K"],
        **_BENCH_GEMM_CONFIG,
    )


def _get_bench_gemm_case(name):
    for case in _BENCH_GEMM_CASES:
        if case["name"] == name:
            return case
    raise KeyError(f"unknown GEMM benchmark case: {name}")


def regression_bench_gemm_m1664_n1024_k262144():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m1664_n1024_k262144"))


def regression_bench_gemm_maca_template_m1664_n1024_k262144():
    _process_bench_gemm_maca_template_case(_BENCH_GEMM_MACA_TEMPLATE_CASE)


def regression_bench_gemm_maca_baseline_m1664_n1024_k262144():
    _process_bench_gemm_maca_baseline_case(_BENCH_GEMM_MACA_BASELINE_CASE)


def regression_bench_gemm_maca_kpack2_m1664_n1024_k262144():
    _process_bench_gemm_maca_kpack2_case(_BENCH_GEMM_MACA_KPACK2_CASE)


def regression_bench_gemm_m4096_n1024_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m4096_n1024_k8192"))


def regression_bench_gemm_m4096_n8192_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m4096_n8192_k8192"))


def regression_bench_gemm_m4096_n28672_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m4096_n28672_k8192"))


def regression_bench_gemm_m4096_n8192_k28672():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m4096_n8192_k28672"))


def regression_bench_gemm_m8192_n1024_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m8192_n1024_k8192"))


def regression_bench_gemm_m8192_n8192_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m8192_n8192_k8192"))


def regression_bench_gemm_m8192_n28672_k8192():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m8192_n28672_k8192"))


def regression_bench_gemm_m8192_n8192_k28672():
    _process_bench_gemm_case(_get_bench_gemm_case("bench_gemm_m8192_n8192_k28672"))


def regression_example_gemm_intrinsics():
    tilelang.testing.process_func(example_gemm_intrinsics.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_gemm():
    tilelang.testing.process_func(example_gemm.run_regression_perf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "generic-long-k",
            "maca-baseline-long-k",
            "maca-template-long-k",
            "maca-template-pair-long-k",
            "maca-kpack2-long-k",
            "maca-kpack2-pair-long-k",
        ),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        if args.case in {"maca-kpack2-long-k", "maca-kpack2-pair-long-k"}:
            _check_bench_gemm_maca_kpack2_pair()
        else:
            _check_bench_gemm_maca_pair()
    if args.case == "generic-long-k":
        tilelang.testing.regression(prefixes=("regression_bench_gemm_m1664_n1024_k262144",))
    elif args.case == "maca-baseline-long-k":
        tilelang.testing.regression(prefixes=("regression_bench_gemm_maca_baseline_m1664_n1024_k262144",))
    elif args.case == "maca-template-long-k":
        tilelang.testing.regression(prefixes=("regression_bench_gemm_maca_template_m1664_n1024_k262144",))
    elif args.case == "maca-template-pair-long-k":
        tilelang.testing.regression(
            prefixes=(
                "regression_bench_gemm_maca_baseline_m1664_n1024_k262144",
                "regression_bench_gemm_maca_template_m1664_n1024_k262144",
            )
        )
    elif args.case == "maca-kpack2-long-k":
        tilelang.testing.regression(prefixes=("regression_bench_gemm_maca_kpack2_m1664_n1024_k262144",))
    elif args.case == "maca-kpack2-pair-long-k":
        tilelang.testing.regression(
            prefixes=(
                "regression_bench_gemm_maca_baseline_m1664_n1024_k262144",
                "regression_bench_gemm_maca_kpack2_m1664_n1024_k262144",
            )
        )
    else:
        tilelang.testing.regression()
