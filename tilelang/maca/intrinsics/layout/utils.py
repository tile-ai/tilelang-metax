from .mma_layout import thread_id_shared_access_64x4_to_16x16_layout_C_n_m


def mma_store_index_map(thread_id, local_id):
    return thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id)
