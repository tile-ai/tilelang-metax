import pytest

import tilelang.language as T
import tilelang.testing
from tilelang.maca.intrinsics.macro.mma_macro_generator import (
    TensorCoreIntrinEmitter,
    TensorCorePreshuffleIntrinEmitter,
)
from tilelang.maca.intrinsics.macro.mma_sp_macro_generator import SparseTensorCoreIntrinEmitter

DENSE_EMITTERS = [TensorCoreIntrinEmitter, TensorCorePreshuffleIntrinEmitter]
ALL_EMITTERS = [*DENSE_EMITTERS, SparseTensorCoreIntrinEmitter]


_UNSET = object()


def _emitter(emitter_cls, a_dtype=T.float16, b_dtype=T.float16, accum_dtype=_UNSET):
    common = {
        "a_dtype": a_dtype,
        "b_dtype": b_dtype,
        "a_transposed": False,
        "b_transposed": True,
        "block_row_warps": 2,
        "block_col_warps": 2,
        "warp_row_tiles": 32,
        "warp_col_tiles": 32,
    }
    # Left out of the kwargs when unset, so the emitter's own default is what gets exercised.
    if accum_dtype is not _UNSET:
        common["accum_dtype"] = accum_dtype
    if emitter_cls is SparseTensorCoreIntrinEmitter:
        return emitter_cls(e_dtype=T.uint8, warp_k=16, **common)
    return emitter_cls(chunk=32, **common)


@pytest.mark.parametrize("emitter_cls", ALL_EMITTERS)
@pytest.mark.parametrize("accum_dtype", [T.float16, T.bfloat16, T.float8_e4m3, T.float8_e5m2])
def test_narrow_float_accumulation_is_rejected(emitter_cls, accum_dtype):
    # codegen builds the builtin name as `"__builtin_mxc_mma_" + prefix` and casts C to
    # `dtype_map[accum_dtype + "x4"]` (src/maca/codegen/codegen_maca.cc). The float
    # matrix-core builtins take and return a float32x4, so any float accumulator narrower
    # than 32 bits hands them the wrong vector type and mxcc rejects the call — an opaque
    # error several stages downstream. On upstream, each of these fails to compile.
    with pytest.raises(ValueError, match="does not support .* accumulation"):
        _emitter(emitter_cls, accum_dtype=accum_dtype)


@pytest.mark.parametrize("emitter_cls", ALL_EMITTERS)
def test_fp32_accumulation_is_the_default(emitter_cls):
    emitter = _emitter(emitter_cls)

    assert emitter.accum_dtype == T.float32
    assert emitter.accum_dtype_abbrv == "fp32"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("emitter_cls", DENSE_EMITTERS)
def test_int32_accumulation_is_still_accepted(emitter_cls):
    # The int8 shapes accumulate in int32, so rejecting fp16 must not spill over onto them.
    # This is the only case that needs a live target: `_initialize_k_dim` calls
    # `determine_target()` on the 8-bit branch alone, to read `mcpu`.
    emitter = _emitter(emitter_cls, a_dtype=T.int8, b_dtype=T.int8, accum_dtype=T.int32)

    assert emitter.accum_dtype_abbrv == "int32"


if __name__ == "__main__":
    tilelang.testing.main()
