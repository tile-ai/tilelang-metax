"""Contiguous-KV adaptation of the upstream AMD-example Triton MLA kernel.

Derived from tile-ai/tilelang, MIT licensed. The main and merge math follow the
reference. Identity-page lookup and length tensors are replaced by contiguous
batch offsets and the shared KV length. Source:
https://github.com/tile-ai/tilelang/blob/4ed9b9edb5c9a1e05f52b3bd45e7116fea7ec9ab/examples/deepseek_mla/amd/benchmark_mla_decode_amd_triton.py
This module is an independent benchmark baseline, not the TileLang compute path.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mla_attn_kernel(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    O,
    sm_scale,
    stride_q_nope_bs,
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_LENGTH: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
):
    cur_batch = tl.program_id(1)
    cur_head_id = tl.program_id(0)
    split_kv_id = tl.program_id(2)

    cur_batch_seq_len = KV_LENGTH

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_q_nope = cur_batch * stride_q_nope_bs + cur_head[:, None] * stride_q_nope_h + offs_d_ckv[None, :]
    q_nope = tl.load(Q_nope + offs_q_nope)

    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)
    offs_q_pe = cur_batch * stride_q_pe_bs + cur_head[:, None] * stride_q_pe_h + offs_d_kpe[None, :]
    q_pe = tl.load(Q_pe + offs_q_pe)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=tl.float32)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_loc = cur_batch * KV_LENGTH + offs_n
        offs_k_c = kv_loc[None, :] * stride_kv_c_bs + offs_d_ckv[:, None]
        k_c = tl.load(Kv_c_cache + offs_k_c, mask=offs_n[None, :] < split_kv_end, other=0.0)

        qk = tl.dot(q_nope, k_c.to(q_nope.dtype))

        offs_k_pe = kv_loc[None, :] * stride_k_pe_bs + offs_d_kpe[:, None]
        k_pe = tl.load(K_pe_cache + offs_k_pe, mask=offs_n[None, :] < split_kv_end, other=0.0)

        qk += tl.dot(q_pe, k_pe.to(q_pe.dtype))
        qk *= sm_scale

        qk = tl.where(offs_n[None, :] < split_kv_end, qk, float("-inf"))

        v_c = tl.trans(k_c)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v_c.dtype), v_c)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
    offs_o = cur_batch * stride_o_b + cur_head[:, None] * stride_o_h + split_kv_id * stride_o_s + offs_d_ckv[None, :]
    tl.store(O + offs_o, acc / e_sum[:, None])
    offs_o_1 = cur_batch * stride_o_b + cur_head * stride_o_h + split_kv_id * stride_o_s + HEAD_DIM_CKV
    tl.store(O + offs_o_1, e_max + tl.log(e_sum))


@triton.jit
def _mla_softmax_reducev_kernel(
    Logits,
    O,
    stride_l_b,
    stride_l_h,
    stride_l_s,
    stride_o_b,
    stride_o_h,
    NUM_KV_SPLITS: tl.constexpr,
    KV_LENGTH: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_batch_seq_len = KV_LENGTH

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

    offs_l = cur_batch * stride_l_b + cur_head * stride_l_h + offs_d_ckv
    offs_l_1 = cur_batch * stride_l_b + cur_head * stride_l_h + HEAD_DIM_CKV

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            logits = tl.load(Logits + offs_l + split_kv_id * stride_l_s)
            logits_1 = tl.load(Logits + offs_l_1 + split_kv_id * stride_l_s)

            n_e_max = tl.maximum(logits_1, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(logits_1 - n_e_max)
            acc += exp_logic * logits

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d_ckv,
        acc / e_sum,
    )


def run(q, q_pe, kv, k_pe, splits, head_tile=16, key_tile=32, warps=8, scenario=""):
    batch, heads, dim = q.shape
    length = kv.shape[1]
    assert all(x.is_contiguous() for x in (q, q_pe, kv, k_pe))
    assert dim == 512 and q_pe.shape[-1] == 64
    assert heads % head_tile == 0 and length % (splits * key_tile) == 0
    partial = torch.empty((batch, heads, splits, dim + 1), device=q.device, dtype=torch.float32)
    output = torch.empty_like(q)
    _mla_attn_kernel[(heads // head_tile, batch, splits)](
        q,
        q_pe,
        kv,
        k_pe,
        partial,
        1 / math.sqrt(dim + q_pe.shape[-1]),
        q.stride(0),
        q.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        kv.stride(1),
        k_pe.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        BLOCK_H=head_tile,
        BLOCK_N=key_tile,
        NUM_KV_SPLITS=splits,
        KV_LENGTH=length,
        HEAD_DIM_CKV=dim,
        HEAD_DIM_KPE=q_pe.shape[-1],
        num_warps=warps,
        num_stages=1,
        scenario=scenario,
    )
    _mla_softmax_reducev_kernel[(batch, heads)](
        partial,
        output,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        output.stride(0),
        output.stride(1),
        NUM_KV_SPLITS=splits,
        KV_LENGTH=length,
        HEAD_DIM_CKV=dim,
        num_warps=warps,
    )
    return output
