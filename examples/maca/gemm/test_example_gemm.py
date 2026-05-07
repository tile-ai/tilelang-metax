import tilelang.testing
import example_gemm_autotune
import example_gemm_intrinsics
import example_gemm
import example_gemm_with_maca_async_copy


def test_example_gemm_autotune():
    # enable roller for fast tuning
    example_gemm_autotune.main(M=1024, N=1024, K=1024, with_roller=True)


def test_example_gemm_intrinsics():
    example_gemm_intrinsics.main(M=1024, N=1024, K=1024)


def test_example_gemm():
    example_gemm.main()


def test_example_gemm_with_maca_async_copy():
    example_gemm_with_maca_async_copy.main()


if __name__ == "__main__":
    tilelang.testing.main()
