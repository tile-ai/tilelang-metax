"""Dense single-token MLA decode for the MetaX C500."""

from functools import lru_cache
import math

import tilelang
from tilelang.maca import language as T
from tilelang.maca.intrinsics import TensorCoreIntrinEmitter


def default_schedule(batch, query_heads, kv_ctx):
    """Select a head tile and a power-of-two split count for a C500 slice."""
    if batch < 1 or query_heads < 16 or query_heads % 16 or not 1 <= kv_ctx <= 65536:
        raise ValueError("expected a positive batch, query heads divisible by 16 and KV length in [1, 65536]")
    # Head tiling increases KV reuse but reduces independent head groups;
    # KV splits recover parallelism at the cost of partial-state traffic and merge.
    # The thresholds below are measured defaults, not hardware limits.
    tiles = (kv_ctx + 31) // 32
    max_splits = 1 << (tiles.bit_length() - 1)

    # Wider blocks reuse a KV tile across more query heads. Use them once
    # the product of independent head groups and key tiles is large enough.
    wide_groups = batch * (query_heads // 64)
    if query_heads % 64 == 0 and wide_groups * tiles >= 1024:
        # Near the crossover, one grid of about 64 blocks avoids the merge
        # traffic of extra splits. Longer workloads benefit from more blocks.
        block_target = 64 if wide_groups * tiles <= 3072 else min(512, 8 * tiles)
        blocks_per_group = (block_target + wide_groups - 1) // wide_groups
        split_limit = 1 << (max(1, tiles // 16).bit_length() - 1)
        return 64, min(1 << (blocks_per_group - 1).bit_length(), split_limit)

    medium_groups = batch * (query_heads // 32)
    if query_heads % 32 == 0 and medium_groups * tiles >= 128:
        minimum = (64 + medium_groups - 1) // medium_groups
        maximum = (512 + medium_groups - 1) // medium_groups
        target = max(minimum, min(maximum, max(1, tiles // 16)))
        return 32, min(1 << (target - 1).bit_length(), max_splits)

    # Start near 64 independent blocks. Once a split exceeds one unrolled
    # group, target four key tiles per split.
    groups = batch * (query_heads // 16)
    blocks_per_group = (64 + groups - 1) // groups
    occupancy_splits = 1 << (blocks_per_group - 1).bit_length()
    target = occupancy_splits if tiles <= 8 * occupancy_splits else (tiles + 3) // 4
    return 16, min(1 << (target - 1).bit_length(), max_splits)


@lru_cache(maxsize=None)
@tilelang.jit(target="maca", pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def mla_decode_maca(batch, kv_ctx, query_heads=16, num_split=None, head_tile=None):
    default_heads, default_splits = default_schedule(batch, query_heads, kv_ctx)
    # `heads` is the per-block query-head tile, not the total query head count.
    heads = default_heads if head_tile is None else head_tile
    num_split = default_splits if num_split is None else num_split
    assert heads in (16, 32, 64, 128)
    kv_heads, dim, pe_dim = 1, 512, 64
    # Two column waves cover 512 output columns. Each additional row group
    # lets another 16 heads consume the same shared KV tile.
    row_waves = heads // 16
    assert query_heads % heads == 0
    assert 1 <= num_split <= 2048 and num_split & (num_split - 1) == 0
    dtype, accum_dtype = T.float16, T.float32
    block_n, wave_size = 32, 64
    main_threads, merge_threads = row_waves * 128, 256
    num_tiles = (kv_ctx + block_n - 1) // block_n
    base_tiles, extra = divmod(num_tiles, num_split)
    max_tiles = base_tiles + int(extra > 0)
    # Bound unrolled KV tiles and independent merge chains in generated code.
    group_tiles = min(max_tiles, 8)
    chains = min(8, num_split)
    active_lanes = min(num_split, wave_size)
    reduction_steps = active_lanes.bit_length() - 1
    groups_per_band = min(8, num_split // chains)
    # At 64 heads, the packed FP16 Q tile alone is 72 KiB. A fragment
    # keeps that tile out of shared memory on the wide schedules, trading shared
    # capacity for registers. It does not guarantee lower total register usage.
    alloc_query = T.alloc_shared if heads == 16 else T.alloc_fragment
    # FP32 split state is packed as [partial PV: B,S,H,D], [sum P: B,S,H],
    # then [max QK: B,S,H]. Normalize after merging, not before storing partials.
    stats_offset = batch * num_split * query_heads * dim
    max_offset = stats_offset + batch * num_split * query_heads
    workspace_size = batch * num_split * query_heads * (dim + 2) if num_split > 1 else 1
    scale = math.log2(math.e) / math.sqrt(dim + pe_dim)

    pv_mma = TensorCoreIntrinEmitter(
        a_dtype=dtype,
        b_dtype=dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=False,
        block_row_warps=row_waves,
        block_col_warps=2,
        warp_row_tiles=16,
        warp_col_tiles=dim // 2,
        chunk=block_n,
    )
    _, pv_b_reverse = pv_mma.get_ldmatrix_index_map(is_b=True)

    @T.prim_func
    def main(
        q: T.Tensor([batch, query_heads, dim], dtype),
        q_pe: T.Tensor([batch, query_heads, pe_dim], dtype),
        kv: T.Tensor([batch, kv_ctx, kv_heads, dim], dtype),
        k_pe: T.Tensor([batch, kv_ctx, kv_heads, pe_dim], dtype),
        output: T.Tensor([batch, query_heads, dim], dtype),
    ):
        workspace = T.alloc_global([workspace_size], accum_dtype)

        with T.Kernel(num_split, batch * (query_heads // heads), threads=main_threads) as (sid, block):
            bid = block // (query_heads // heads)
            head_start = (block % (query_heads // heads)) * heads
            q_tile = alloc_query([heads, dim + pe_dim], dtype)
            key_shared = T.alloc_shared([block_n, dim + pe_dim], dtype)
            prob_shared = T.alloc_shared([heads, block_n], dtype)
            if heads == 16:
                T.annotate_layout(
                    {
                        q_tile: tilelang.layout.make_swizzled_layout(q_tile),
                        key_shared: tilelang.layout.make_swizzled_layout(key_shared),
                        prob_shared: tilelang.layout.make_swizzled_layout(prob_shared),
                    }
                )
            else:
                T.annotate_layout(
                    {
                        key_shared: tilelang.layout.make_swizzled_layout(key_shared),
                        prob_shared: tilelang.layout.make_swizzled_layout(prob_shared),
                    }
                )
            score = T.alloc_fragment([heads, block_n], accum_dtype)
            running_max = T.alloc_fragment([heads], accum_dtype)
            previous_max = T.alloc_fragment([heads], accum_dtype)
            rescale = T.alloc_shared([heads], accum_dtype)
            # A wave owns 16 x 256 output elements: 4096 / 64 lanes = 64 per lane.
            # Operand and accumulator fragments follow the emitter's lane mapping.
            output_acc = T.alloc_local([64], accum_dtype)
            pv_a = T.alloc_local([4], dtype)
            pv_b = T.alloc_local([64], dtype)
            ones_b = T.alloc_local([4], dtype)
            denom_acc = T.alloc_local([4], accum_dtype)
            tx_main = T.get_thread_binding()
            lane_main = tx_main % 64
            _, wave_main, wave_row = pv_mma.extract_thread_binding(tx_main)

            T.copy(q[bid, head_start : head_start + heads, :], q_tile[:, :dim])
            T.copy(q_pe[bid, head_start : head_start + heads, :], q_tile[:, dim : dim + pe_dim])
            T.fill(output_acc, 0)
            T.fill(denom_acc, 0)
            T.fill(ones_b, 1.0)
            T.fill(running_max, -T.infinity(accum_dtype))

            # Distribute whole tiles; idle splits retain (-inf, 0, 0) for the merge.
            first_tile = sid * base_tiles + T.min(sid, extra)
            for group in T.serial((max_tiles + group_tiles - 1) // group_tiles):
                for tile in T.unroll(group_tiles):
                    local_tile = group * group_tiles + tile
                    if local_tile < base_tiles or (local_tile == base_tiles and sid < extra):
                        kv_start = (first_tile + local_tile) * block_n
                        if kv_ctx % block_n and kv_start + block_n > kv_ctx:
                            for n, k in T.Parallel(block_n, dim):
                                key_shared[n, k] = T.if_then_else(kv_start + n < kv_ctx, kv[bid, kv_start + n, 0, k], 0)
                            for n, k in T.Parallel(block_n, pe_dim):
                                key_shared[n, dim + k] = T.if_then_else(kv_start + n < kv_ctx, k_pe[bid, kv_start + n, 0, k], 0)
                        else:
                            T.copy(kv[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, :dim])
                            T.copy(k_pe[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, dim : dim + pe_dim])
                        T.gemm(
                            q_tile,
                            key_shared,
                            score,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            clear_accum=True,
                        )
                        if kv_ctx % block_n:
                            for h, n in T.Parallel(heads, block_n):
                                if kv_start + n >= kv_ctx:
                                    score[h, n] = -T.infinity(accum_dtype)
                        # Online softmax: when the maximum grows, rescale BOTH
                        # the accumulated PV and sum(P) before adding the next tile.
                        if max_tiles > 1:
                            T.copy(running_max, previous_max)
                        T.reduce_max(score, running_max, dim=1, clear=True)
                        if max_tiles > 1:
                            for h in T.Parallel(heads):
                                running_max[h] = T.max(running_max[h], previous_max[h])
                                rescale[h] = T.exp2((previous_max[h] - running_max[h]) * scale)
                            T.sync_threads()
                            factor = rescale[wave_row * 16 + lane_main % 16]
                            for i in T.vectorized(4):
                                denom_acc[i] *= factor
                            for j in T.serial(16):
                                for i in T.vectorized(4):
                                    output_acc[j * 4 + i] *= factor
                        for h, n in T.Parallel(heads, block_n):
                            score[h, n] = T.exp2((score[h, n] - running_max[h]) * scale)
                        T.copy(score, prob_shared)
                        T.sync_threads()
                        for ki in T.serial(2):
                            pv_mma.ldmatrix_a(pv_a, prob_shared, ki)
                            for j in T.serial(16):
                                for local_id in T.vectorized(4):
                                    row, col = T.meta_var(pv_b_reverse(lane_main, local_id))
                                    pv_b[j * 4 + local_id] = key_shared[ki * 16 + row, wave_main * 256 + j * 16 + col]
                            for j in T.serial(16):
                                T.maca_mma(
                                    pv_mma.mma_suffix,
                                    "row",
                                    "row",
                                    "float16x4",
                                    "float16x4",
                                    "float32x4",
                                    pv_b.data,
                                    j,
                                    pv_a.data,
                                    0,
                                    output_acc.data,
                                    j,
                                    dtype="float32x4",
                                )
                            # Multiplying the same P fragment by ones adds its row sums
                            # without a second load from shared memory. The denominator
                            # therefore uses the same rounded FP16 P values as PV.
                            T.maca_mma(
                                pv_mma.mma_suffix,
                                "row",
                                "row",
                                "float16x4",
                                "float16x4",
                                "float32x4",
                                ones_b.data,
                                0,
                                pv_a.data,
                                0,
                                denom_acc.data,
                                0,
                                dtype="float32x4",
                            )
            # One split writes the normalized output directly; multiple splits
            # write unnormalized FP32 state for the second kernel.
            for j in T.serial(16):
                for local_id in T.vectorized(4):
                    row = wave_row * 16 + lane_main % 16
                    col = wave_main * 256 + j * 16 + (lane_main // 16) * 4 + local_id
                    if num_split == 1:
                        output[bid, head_start + row, col] = output_acc[j * 4 + local_id] / denom_acc[local_id]
                    else:
                        workspace[((bid * num_split + sid) * query_heads + head_start + row) * dim + col] = output_acc[j * 4 + local_id]
            if num_split > 1 and wave_main == 0 and lane_main < 16:
                workspace[stats_offset + (bid * num_split + sid) * query_heads + head_start + wave_row * 16 + lane_main] = denom_acc[0]
            if num_split > 1:
                for h in T.Parallel(heads):
                    workspace[max_offset + (bid * num_split + sid) * query_heads + head_start + h] = running_max[h]

        if num_split > 1:
            with T.Kernel(query_heads, batch, threads=merge_threads) as (merge_head, merge_batch):
                tx = T.get_thread_binding()
                lane, wave = tx % wave_size, tx // wave_size
                weights = T.alloc_shared((merge_threads // wave_size, num_split), accum_dtype)
                acc = T.alloc_local((chains, 2), accum_dtype)
                result = T.alloc_local((2,), accum_dtype)
                global_max = T.alloc_var(accum_dtype)
                total = T.alloc_var(accum_dtype)
                peer = T.alloc_var(accum_dtype)
                weight = T.alloc_var(accum_dtype)

                # Merge in a common softmax scale: w_s = exp2((m_s - max(m)) * scale).
                # Output = sum(w_s * partial_PV_s) / sum(w_s * sum_P_s).
                # Each wave computes its own weights and owns disjoint output columns.
                global_max = workspace[max_offset + (merge_batch * num_split + lane % num_split) * query_heads + merge_head]
                for shard in T.serial(1, (num_split + wave_size - 1) // wave_size):
                    index = (merge_batch * num_split + shard * wave_size + lane) * query_heads + merge_head
                    global_max = T.max(global_max, workspace[max_offset + index])
                for step in T.unroll(reduction_steps):
                    distance = (active_lanes // 2) >> step
                    peer = T.shfl_down(global_max, distance, wave_size)
                    if lane < distance:
                        global_max = T.max(global_max, peer)
                global_max = T.shfl_sync(global_max, 0, wave_size)
                weight = T.exp2(
                    (workspace[max_offset + (merge_batch * num_split + lane % num_split) * query_heads + merge_head] - global_max) * scale
                )
                if lane < num_split:
                    weights[wave, lane] = weight

                total = weight * workspace[stats_offset + (merge_batch * num_split + lane % num_split) * query_heads + merge_head]
                for shard in T.serial(1, (num_split + wave_size - 1) // wave_size):
                    split = shard * wave_size + lane
                    index = (merge_batch * num_split + split) * query_heads + merge_head
                    weight = T.exp2((workspace[max_offset + index] - global_max) * scale)
                    weights[wave, split] = weight
                    total += weight * workspace[stats_offset + index]
                for step in T.unroll(reduction_steps):
                    distance = (active_lanes // 2) >> step
                    peer = T.shfl_down(total, distance, wave_size)
                    if lane < distance:
                        total += peer
                total = T.shfl_sync(total, 0, wave_size)
                T.sync_warp()

                T.fill(acc, 0)
                for band in T.serial(num_split // (chains * groups_per_band)):
                    for group in T.unroll(groups_per_band):
                        for chain in T.unroll(chains):
                            split = (band * groups_per_band + group) * chains + chain
                            for vec in T.vectorized(2):
                                acc[chain, vec] += (
                                    workspace[((merge_batch * num_split + split) * query_heads + merge_head) * dim + tx * 2 + vec]
                                    * weights[wave, split]
                                )
                T.fill(result, 0)
                for chain in T.unroll(chains):
                    for vec in T.vectorized(2):
                        result[vec] += acc[chain, vec]
                for vec in T.vectorized(2):
                    output[merge_batch, merge_head, tx * 2 + vec] = result[vec] / total

    return main
