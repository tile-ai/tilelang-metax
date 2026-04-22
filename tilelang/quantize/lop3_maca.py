# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
from typing import Literal

from tilelang import language as T


decode_i4_to_f16 = """
#include "maca_fp16.h"
template <typename T1, typename T2, bool isSigned = false>
__device__ void decode_i4b_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    const unsigned char *src = reinterpret_cast<const unsigned char *>(_i4s);
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        int value = (src[i >> 1] >> ((i & 1) * 4)) & 0xF;
        if (isSigned && value >= 8) {
            value -= 16;
        }
        B_local_decode[i] = static_cast<T2>(value);
    }
}

template <typename T1, typename T2>
__device__ void decode_i4s_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    decode_i4b_to_f16<T1, T2, true>(_i4s, B_local_decode, N);
}

template <typename T1, typename T2>
__device__ void decode_i4u_to_f16(T1 *_i4u, T2 *B_local_decode, const int N = 8)
{
    decode_i4b_to_f16<T1, T2, false>(_i4u, B_local_decode, N);
}
"""

import_maca_c_map = {"i4_to_f16": decode_i4_to_f16}


def get_lop3_intrin_group(
    out_dtype: Literal[T.float16, T.int8, T.int4],
    source_format: Literal[T.int, T.uint] = T.uint,
    source_bit: int = 4,
    storage_dtype: Literal[T.int32, T.int8] = T.int8,
    with_scaling: bool = False,
    with_zeros: bool = False,
    zeros_mode: Literal["original", "rescale", "quantized"] = "original",
    storage_scope: str = "local",
) -> dict[str, str]:
    """Return MACA C fast-decode source compatible with lop3.get_lop3_intrin_group.

    MetaX C500 does not provide a LOP3-like instruction, so implemented entries use
    portable MACA C rather than NVIDIA PTX inline assembly. Future MetaX
    architectures can add optimized implementations behind the same interface.
    """
    out_dtype, source_format, storage_dtype = (
        T.dtype(out_dtype),
        T.dtype(source_format),
        T.dtype(storage_dtype),
    )
    if out_dtype not in [T.float16, T.int8, T.int4]:
        raise ValueError(f"Invalid out_dtype: {out_dtype}. Expected 'float16' or 'int8' or 'int4'.")
    if source_format not in [T.int, T.uint]:
        raise ValueError(f"Invalid source_format. Expected 'int' or 'uint', but got {source_format}, {type(source_format)}.")
    if storage_dtype not in [T.int8, T.int32]:
        raise ValueError(f"Invalid storage_dtype: {storage_dtype}. Expected 'int8' or 'int32'.")
    if with_zeros and source_format == T.int:
        raise ValueError(f"Zeros are not supported for signed integers, but got {source_format}")

    dtype_mapping = {T.float16: "f16", T.int4: "i4", T.int8: "i8", T.int32: "i32"}
    key = f"i{source_bit}_to_{dtype_mapping[out_dtype]}"
    if with_scaling:
        key += "_scale"
    if with_zeros:
        key += f"_zeros_{zeros_mode}"

    is_ladder_stage3 = (storage_scope == "warp") and with_scaling
    if is_ladder_stage3:
        key += "_offset"

    if key not in import_maca_c_map:
        raise NotImplementedError(f"MACA LOP3 fast decoding is not implemented for key '{key}'")

    if out_dtype == T.float16:
        d4f = "f16"
    elif out_dtype == T.int8:
        d4f = "i8s"
    elif out_dtype == T.int4:
        d4f = "i4s"
    else:
        raise ValueError(f"Unsupported target dtype: {out_dtype}")

    source_symbol = "u" if source_format == T.uint else "s"
    func_name = f"decode_i{source_bit}{source_symbol}_to_{d4f}"
    if with_scaling:
        func_name += "_scale"
    if with_zeros:
        func_name += f"_zeros_{zeros_mode}"
    if is_ladder_stage3:
        func_name += "_offset"

    return {
        "func_name": func_name,
        "c_source": import_maca_c_map[key],
    }
