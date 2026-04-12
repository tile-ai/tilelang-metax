import tilelang.testing

import example_dequant_gemv_fp16xint4
import example_dequant_gemm_fp4_hopper
import example_dequant_gemm_bf16_mxfp4_hopper
import example_dequant_gemm_w4a8


def test_example_dequant_gemv_fp16xint4():
    example_dequant_gemv_fp16xint4.main()


@tilelang.testing.pytest.mark.xfail
def test_example_dequant_gemm_fp4_hopper():
    example_dequant_gemm_fp4_hopper.main()


@tilelang.testing.pytest.mark.xfail
def test_example_dequant_gemm_bf16_mxfp4_hopper():
    example_dequant_gemm_bf16_mxfp4_hopper.main()


def test_example_dequant_gemm_w4a8():
    example_dequant_gemm_w4a8.main()


if __name__ == "__main__":
    tilelang.testing.main()
