import tilelang.language as T

from tilelang.intrinsics.maca_mma_macro_generator import TensorCoreIntrinEmitter


def test_maca_mma_fp8_dtype_metadata_uses_fp16_mma_input():
    fp8_info = TensorCoreIntrinEmitter.dtype_abbrv["float8_e4m3fnuz"]

    assert fp8_info["storage"] == "e4m3fnuz"
    assert fp8_info["mma"] == "fp8"
    assert fp8_info["mma_input_dtype"] == T.float16

    emitter = TensorCoreIntrinEmitter(
        a_dtype="float8_e4m3fnuz",
        b_dtype="float8_e4m3fnuz",
        accum_dtype="float32",
    )

    assert emitter.mma_input_dtype == T.float16
    assert emitter.mma_suffix == "16x16x16fp8"
