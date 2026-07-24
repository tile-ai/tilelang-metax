from tvm import DataType
from tvm.runtime import convert
from tvm.tirx import const
import tilelang.language as T


def mma_load_a_32x4_to_shared_16x8_layout(thread_id, local_id):
    row = 8 * (local_id % 2) + (thread_id // 4)
    col = 4 * (local_id // 2) + (thread_id % 4)
    return row, col


def mma_load_a_32x16_to_shared_16x32_layout(thread_id, local_id):
    row = 8 * (local_id % 8 // 4) + (thread_id // 4)
    col = 16 * (local_id // 8) + (thread_id % 4) * 4 + (local_id % 4)
    return row, col


def mma_load_a_32x8_to_shared_16x16_layout(thread_id, local_id):
    """
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =      groupID            for ai where  0 <= i < 2 || 4 <= i < 6
            groupID + 8         Otherwise

    col =  (threadID_in_group * 2) + (i & 0x1)          for ai where i <  4
    (threadID_in_group * 2) + (i & 0x1) + 8      for ai where i >= 4
    """
    row = (thread_id // 4) + 8 * (local_id % 4 // 2)
    col = (thread_id % 4) * 2 + (local_id % 2) + 8 * (local_id // 4)
    return row, col


def shared_16x8_to_mma_a_32x4_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 4)
    return thread_id, 2 * (j // 4) + (i // 8)


shared_16x8_to_mma_32x4_layout_sr_a = shared_16x8_to_mma_a_32x4_layout


def shared_16x16_to_mma_a_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


shared_16x16_to_mma_32x8_layout_sr_a = shared_16x16_to_mma_a_32x8_layout


def shared_16x32_to_mma_a_32x16_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (j // 16) + (i // 8) * 4 + j % 4


shared_16x32_to_mma_32x16_layout_sr_a = shared_16x32_to_mma_a_32x16_layout


def shared_16x4_to_local_64x1_layout_A(i, j):
    thread_id = j * 16 + i
    return thread_id, const(0)


def thread_id_shared_access_64x1_to_16x4_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = thread_id // 16
    return i, j


def shared_4x16_to_local_64x1_layout_B(i, j):
    thread_id = i * 16 + j
    return thread_id, const(0)


def thread_id_shared_access_64x1_to_4x16_layout_B(thread_id, local_id):
    i = thread_id // 16
    j = thread_id % 16
    return i, j


def shared_16x8_to_local_64x2_layout_A(i, j):
    thread_id = i + 16 * (j // 2)
    local = j % 2
    return thread_id, local


def thread_id_shared_access_64x2_to_16x8_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 2 + local_id
    return i, j


def shared_8x16_to_local_64x2_layout_B(i, j):
    thread_id = j + 16 * (i // 2)
    local = i % 2
    return thread_id, local


def thread_id_shared_access_64x2_to_8x16_layout_B(thread_id, local_id):
    i = (thread_id // 16) * 2 + local_id
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_C(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


def shared_16x16_to_ldmatrix_64x4_layout(ind):
    i, j = ind[0], ind[1]
    thread_id, local_id = shared_16x16_to_local_64x4_layout_C(i, j)
    return convert([thread_id, local_id])


def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local = j % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_B(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


shared_16x16_to_local_64x4_layout_m_n = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_k = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_m = shared_16x16_to_local_64x4_layout_B
shared_16x16_to_local_64x4_layout_k_n = shared_16x16_to_local_64x4_layout_B


def thread_id_shared_access_64x4_to_16x16_layout_C_m_n(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 4
    return i, j


def thread_id_shared_access_64x8_to_16x32_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 8 + local_id
    return i, j


def shared_16x32_to_local_64x8_layout_A(i, j):
    thread_id = i + 16 * (j // 8)
    local = j % 8
    return thread_id, local


def thread_id_shared_access_64x8_to_16x32_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 8
    j = thread_id % 16
    return i, j


def shared_16x32_to_local_64x8_layout_B(i, j):
    thread_id = j + (i // 8) * 16
    local = i % 8
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 16
    return i, j


def shared_16x64_to_local_64x16_layout_A(i, j):
    thread_id = i + 16 * (j // 16)
    local = j % 16
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 16
    j = thread_id % 16
    return i, j


def shared_16x64_to_local_64x16_layout_B(i, j):
    thread_id = j + (i // 16) * 16
    local = i % 16
    return thread_id, local


def shared_32x32_to_local_64x16_layout_C(i, j):
    thread_id = (i % 8 // 4) * 32 + j
    local_id = (i // 8) * 4 + i % 4
    return thread_id, local_id


def thread_id_shared_access_64x16_to_32x32_layout_C_n_m(thread_id, local_id):
    i = (thread_id // 32) * 4 + local_id % 4 + (local_id // 4) * 8
    j = thread_id % 32
    return i, j


def thread_id_shared_access_64x16_to_32x32_layout_C_m_n(thread_id, local_id):
    m = (thread_id // 32) * 4 + local_id % 4 + (local_id // 4) * 8
    n = thread_id % 32
    return m, n


def shared_32x32_to_local_64x16_layout_A(i, j):
    thread_id = i + 32 * (j // 16)
    local_id = j % 16
    return thread_id, local_id


def thread_id_shared_access_64x16_to_32x32_layout_A(thread_id, local_id):
    i = thread_id % 32
    j = (thread_id // 32) * 16 + local_id
    return i, j


def shared_32x32_to_local_64x16_layout_B(i, j):
    thread_id = j + 32 * (i // 16)
    local_id = i % 16
    return thread_id, local_id


def thread_id_shared_access_64x16_to_32x32_layout_B(thread_id, local_id):
    i = (thread_id // 32) * 16 + local_id
    j = thread_id % 32
    return i, j


def make_mma_swizzle_layout(shared_buf, vecSize=8):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    numBanks = 32
    bankBitWidth = 32
    SIMDWidth = 16

    innerDimLength = shape[-1]
    typeWidthInBit = DataType(dtype).bits

    elemsPerOneBanksRow = (numBanks * bankBitWidth) // typeWidthInBit
    perPhase = max(1, elemsPerOneBanksRow // innerDimLength)
    maxPhase = min(SIMDWidth // perPhase, innerDimLength // vecSize)

    def transform(row, col):
        phase = (row // perPhase) % maxPhase
        colOffSwizzled = ((col // vecSize) ^ phase) * vecSize
        colOffOrdered = col % vecSize
        colOff = colOffSwizzled + colOffOrdered
        return row, colOff

    return T.Layout(shape, transform)
