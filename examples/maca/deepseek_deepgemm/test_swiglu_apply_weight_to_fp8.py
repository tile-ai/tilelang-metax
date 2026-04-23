# Modified from: https://github.com/deepseek-ai/DeepGEMM/blob/main/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py
import tilelang
import torch
from math import gcd
from tilelang import language as T
from typing import Any


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def get_sf_shape(
    num_tokens: int,
    hidden: int,
    num_per_channels: int,
    use_ue8m0: bool,
    use_col_major_sf: bool,
) -> tuple[int, int]:
    num_scales = ceil_div(hidden, num_per_channels)
    num_scales = ceil_div(num_scales, 4) if use_ue8m0 else num_scales

    # For col-major SF, TMA must be aligned into 16 bytes
    # For UE8M0, we must use col-major SF, and 4 UE8M0 are expanded into the inner dim (token)
    num_sf_tokens = num_tokens
    if use_col_major_sf:
        num_sf_tokens = align(num_tokens, 4)
        num_sf_tokens = num_sf_tokens * 4 if use_ue8m0 else num_sf_tokens

    return (num_scales, num_sf_tokens) if use_col_major_sf else (num_sf_tokens, num_scales)


def get_sf_and_inv(amax: float, round_sf: bool, use_ue8m0: bool) -> tuple[Any, Any]:
    sf = amax / 448.0
    if not round_sf:
        return sf, 448.0 / amax

    # Round into 2's power
    bits = T.reinterpret("uint32", sf)
    exp = (bits >> 23) & 0xFF
    man_bits = bits & ((1 << 23) - 1)
    exp_scale = T.reinterpret("int32", exp - 127 + (man_bits != 0))
    if use_ue8m0:  # noqa: SIM108
        sf = T.Cast("uint8", exp_scale + 127)
    else:
        sf = T.reinterpret("float", (127 + exp_scale) << 23)
    return sf, T.reinterpret("float", (127 - exp_scale) << 23)


@tilelang.jit
def _swiglu_apply_weight_to_fp8_tl(
    half_hidden: int,
    num_per_channels: int,
    use_col_major_scales: bool,
    round_scale: bool,
    ue8m0_scale: bool,
    num_ctas: int,
    has_topk_weights: bool,
    has_avail_tokens: bool,
    has_clamp_value: bool,
    output_bf16: bool,
    fast_math: bool,
) -> None:
    in_dtype = T.bfloat16
    w_dtype = T.float32
    out_dtype = T.float8_e4m3fn
    out_sf_dtype = T.uint8 if ue8m0_scale else T.float32
    num_dtype = T.int32
    num_tokens = T.dynamic("num_tokens")
    assert half_hidden % 16 == 0
    if not output_bf16:
        assert half_hidden % num_per_channels == 0

    num_block_h = max(half_hidden // gcd(half_hidden, 16 * 1024), 1)
    assert num_block_h <= num_ctas, "not supported hidden size"
    blk_h = half_hidden // num_block_h

    layout_h = blk_h // 16

    blk_n = 1024 // layout_h

    def local_layout(i, j):  # noqa: ANN001
        thread_id = i * layout_h + j // 16
        local_id = j % 16
        return thread_id, local_id

    def local_layout_3d(i, j, k):  # noqa: ANN001
        return local_layout(i, j * num_per_channels + k)

    @T.macro
    def main(
        bi: int,
        bh: int,
        num_ctas: int,
        x: T.Tensor[(num_tokens, half_hidden * 2), in_dtype],  # type: ignore
        topk_weights: T.Tensor[num_tokens, w_dtype],  # type: ignore
        avail_tokens: T.Tensor[1, num_dtype],  # type: ignore
        out: T.Tensor[(num_tokens, half_hidden), out_dtype],  # type: ignore
        out_sf: T.Tensor[get_sf_shape(num_tokens, half_hidden, num_per_channels, ue8m0_scale, use_col_major_scales), out_sf_dtype],  # type: ignore
        out_bf16: T.Tensor[(num_tokens, half_hidden), T.bfloat16],  # type: ignore
        clamp_value: T.float32,
    ):
        gate_frag = T.alloc_fragment((blk_n, blk_h), T.float32)
        up_frag = T.alloc_fragment((blk_n, blk_h), T.float32)
        y_frag = T.alloc_fragment((blk_n, blk_h // num_per_channels, num_per_channels), T.float32)
        y_f8_frag = T.alloc_fragment((blk_n, blk_h), out_dtype)
        T.annotate_layout(
            {
                gate_frag: T.Fragment(gate_frag.shape, forward_fn=local_layout),
                up_frag: T.Fragment(up_frag.shape, forward_fn=local_layout),
                y_frag: T.Fragment(y_frag.shape, forward_fn=local_layout_3d),
                y_f8_frag: T.Fragment(y_f8_frag.shape, forward_fn=local_layout),
            }
        )

        T.assume(0 <= bh * blk_h + blk_h <= half_hidden)

        for i, j in T.Parallel(blk_n, blk_h):
            gate_frag[i, j] = x[bi + i * num_ctas, bh * blk_h + j]
        for i, j in T.Parallel(blk_n, blk_h):
            up_frag[i, j] = x[bi + i * num_ctas, half_hidden + bh * blk_h + j]

        topk_weight = T.alloc_fragment((blk_n,), T.float32)
        for i in T.Parallel(blk_n):
            topk_weight[i] = topk_weights[bi + i * num_ctas] if has_topk_weights else 1.0

        zero = T.alloc_var(T.float32, 0.0)
        for i, j in T.Parallel(blk_n, blk_h):
            if has_clamp_value:
                up_frag[i, j] = T.min(clamp_value, T.max(-clamp_value, up_frag[i, j]))
                gate_frag[i, j] = T.min(clamp_value, gate_frag[i, j])
            y_frag[i, j // num_per_channels, j % num_per_channels] = (
                gate_frag[i, j] / (1 + T.exp(-gate_frag[i, j])) * up_frag[i, j] * topk_weight[i] + zero
            )  # HACK : + 0 for vectorize

        y_max_frag = T.alloc_fragment((blk_n, blk_h // num_per_channels), T.float32)
        sf_inv_frag = T.alloc_fragment((blk_n, blk_h // num_per_channels), T.float32)
        T.reduce_absmax(T.reshape(y_frag, (blk_n, blk_h // num_per_channels, num_per_channels)), y_max_frag)
        for i, j in T.Parallel(blk_n, blk_h // num_per_channels):
            clamped_amax = T.max(y_max_frag[i, j], 1e-4)
            sf, sf_inv = get_sf_and_inv(clamped_amax, round_scale, ue8m0_scale)
            i_index = bi + i * num_ctas
            j_index = blk_h // num_per_channels * bh + j
            # Store SF
            if ue8m0_scale:
                out_sf[j_index // 4, i_index * 4 + j_index % 4] = sf
            elif use_col_major_scales:
                out_sf[j_index, i_index] = sf
            else:
                out_sf[i_index, j_index] = sf
            sf_inv_frag[i, j] = sf_inv

        for i, j in T.Parallel(blk_n, blk_h):
            y_f8_frag[i, j] = y_frag[i, j // num_per_channels, j % num_per_channels] * sf_inv_frag[i, j // num_per_channels]

        for i, j in T.Parallel(blk_n, blk_h):
            out[bi + i * num_ctas, blk_h * bh + j] = y_f8_frag[i, j]

        if output_bf16:
            for i, j in T.Parallel(blk_n, blk_h):
                out_bf16[bi + i * num_ctas, blk_h * bh + j] = y_frag[i, j // num_per_channels, j % num_per_channels]

    @T.prim_func
    def _swiglu_apply_weight_to_fp8(
        x: T.Tensor[(num_tokens, half_hidden * 2), in_dtype],  # type: ignore
        topk_weights: T.Tensor[num_tokens, w_dtype],  # type: ignore
        avail_tokens: T.Tensor[1, num_dtype],  # type: ignore
        out: T.Tensor[(num_tokens, half_hidden), out_dtype],  # type: ignore
        out_sf: T.Tensor[get_sf_shape(num_tokens, half_hidden, num_per_channels, ue8m0_scale, use_col_major_scales), out_sf_dtype],  # type: ignore
        out_bf16: T.Tensor[(num_tokens, half_hidden), T.bfloat16],  # type: ignore
        clamp_value: T.float32,
    ):
        # we actually don't use this
        _ = num_tokens
        # simplest schedule: one token one block, but pipelined as persistent
        with T.Kernel(num_ctas, threads=1024) as cta_id:
            avail_tokens_l = avail_tokens[0] if has_avail_tokens else num_tokens
            # T.pdl_sync()  # avail_tokens must be const.
            T.assume(0 <= avail_tokens_l <= num_tokens)
            thread_idx = T.get_thread_binding()
            new_num_ctas = num_ctas // num_block_h
            if cta_id >= new_num_ctas * num_block_h:
                T.thread_return()
            for bi in T.serial(cta_id // num_block_h, avail_tokens_l - thread_idx // layout_h * new_num_ctas, new_num_ctas * blk_n):
                main(bi, cta_id % num_block_h, new_num_ctas, x, topk_weights, avail_tokens, out, out_sf, out_bf16, clamp_value)

    return _swiglu_apply_weight_to_fp8


def swiglu_apply_weight_to_fp8(
    x: torch.Tensor,
    topk_weights: torch.Tensor | None,
    avail_tokens: torch.Tensor | None,
    num_per_channels: int,
    use_col_major_scales: bool,
    round_scale: bool,
    ue8m0_scale: bool,
    clamp_value: float | None = None,
    fmt: str = "e4m3",
    num_sms: int | None = None,
    output_bf16: bool = False,
    fast_math: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert fmt == "e4m3"
    assert num_sms is not None

    num_tokens, hidden_size = x.shape
    assert hidden_size % (2 * num_per_channels) == 0

    y = torch.empty(
        (num_tokens, hidden_size // 2),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    y_sf = torch.empty(
        get_sf_shape(num_tokens, hidden_size // 2, num_per_channels, ue8m0_scale, use_col_major_scales),
        device=x.device,
        dtype=(torch.uint8 if ue8m0_scale else torch.float32),
    )

    y_bf16 = torch.empty((num_tokens, hidden_size // 2), device=x.device, dtype=torch.bfloat16) if output_bf16 else None

    if num_tokens > 0:
        _swiglu_apply_weight_to_fp8_tl.pass_configs = {
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: fast_math,
        }
        kernel = _swiglu_apply_weight_to_fp8_tl(
            hidden_size // 2,
            num_per_channels,
            use_col_major_scales,
            round_scale,
            ue8m0_scale,
            num_sms,
            topk_weights is not None,
            avail_tokens is not None,
            clamp_value is not None,
            output_bf16,
            fast_math,
        )
        kernel(x, topk_weights, avail_tokens.view(1) if avail_tokens is not None else None, y, y_sf, y_bf16, clamp_value or 0.0)

    if ue8m0_scale:
        if num_tokens == 0:
            y_sf.as_strided_(y_sf.size(), (0, 1))
        y_sf = y_sf.view(dtype=torch.int32)
    if output_bf16:
        return y, y_sf.T[:num_tokens], y_bf16
    else:
        return y, y_sf.T[:num_tokens]


def ref_program(x, topk_weights, num_tokens, half_hidden, clamp_value):
    gate = x[:, :half_hidden].float()
    up = x[:, half_hidden:].float()
    up = up.clamp(-clamp_value, clamp_value)
    gate = torch.minimum(gate, torch.tensor(clamp_value, dtype=gate.dtype, device="cuda"))
    y_ref = (gate / (1.0 + torch.exp(-gate))) * up * topk_weights.view(num_tokens, 1)
    return y_ref


def assert_swiglu_apply_weight_to_fp8(
    num_tokens: int = 64,
    half_hidden: int = 256,
    num_per_channels: int = 64,
    hidden_size: int = 512,
    clamp_value: float = 5.0,
    num_aps: int | None = None,
):
    assert hidden_size == half_hidden * 2

    if num_aps is None:
        num_aps = torch.cuda.get_device_properties().multi_processor_count

    x = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    topk_weights = torch.rand((num_tokens,), device="cuda", dtype=torch.float32)
    avail_tokens = torch.tensor(num_tokens, device="cuda", dtype=torch.int32)

    y_fp8, y_sf, y_bf16 = swiglu_apply_weight_to_fp8(
        x=x,
        topk_weights=topk_weights,
        avail_tokens=avail_tokens,
        num_per_channels=num_per_channels,
        use_col_major_scales=False,
        round_scale=False,
        ue8m0_scale=False,
        clamp_value=clamp_value,
        num_sms=num_aps,
        output_bf16=True,
        fast_math=False,
    )

    y_ref = ref_program(x, topk_weights, num_tokens, half_hidden, clamp_value)

    num_scales = half_hidden // num_per_channels

    assert y_bf16 is not None
    assert y_fp8.shape == (num_tokens, half_hidden)
    assert y_sf.shape == (num_scales, num_tokens)
    assert y_fp8.dtype == torch.float8_e4m3fn
    assert y_sf.dtype == torch.float32
    assert y_bf16.dtype == torch.bfloat16

    bf16_err = (y_bf16.float() - y_ref).abs()
    y_deq = y_fp8.float().reshape(num_tokens, num_scales, num_per_channels) * y_sf.T.float().reshape(num_tokens, num_scales, 1)
    y_deq = y_deq.reshape(num_tokens, half_hidden)
    fp8_rel_l2 = (torch.linalg.norm((y_deq - y_ref).reshape(-1)) / (torch.linalg.norm(y_ref.reshape(-1)) + 1e-6)).item()

    assert torch.isfinite(y_deq).all().item()
    assert bf16_err.max().item() < 0.03
    assert fp8_rel_l2 < 0.06

    print("Passed all tests!✅")


def main():
    assert_swiglu_apply_weight_to_fp8()


def test_swiglu_apply_weight_to_fp8():
    main()


if __name__ == "__main__":
    main()
