# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.


decode_i4_to_f16 = """
#include "maca_fp16.h"
template <typename T1, typename T2, bool isSigned = false>
__device__ void decode_i4b_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    uint MEDIAN_NUM = isSigned ? 0x64086408 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        h[i] = ((i4s >> (4 * i)) & BOTTOM_MASK) | FP16_TOP_MAGIC_NUM;
        half2 tmp = __hsub2(*reinterpret_cast<half2*>(h + i), *reinterpret_cast<half2*>(&MEDIAN_NUM));
        h[i] = *reinterpret_cast<uint*>(&tmp);
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
