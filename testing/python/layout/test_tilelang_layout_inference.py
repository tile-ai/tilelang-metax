import re

import pytest
import tilelang
import tilelang.testing
import tvm
from tilelang import language as T
from tvm.target import Target


@tilelang.jit
def _tilelang_issue_layout_free_inference_choose_smallest_replication():
    @T.prim_func
    def main(A: T.Tensor((128, 4), T.float), B: T.Tensor((128, 4), T.float)):
        with T.Kernel(1, threads=128) as _:
            A_frag = T.alloc_fragment((128, 4), T.float)
            B_frag = T.alloc_fragment((128, 4), T.float)
            S_frag = T.alloc_fragment((4,), T.float)
            T.annotate_layout(
                {
                    A_frag: T.Fragment(A_frag.shape, lambda i, j: (i, j)),
                }
            )
            for i, j in T.Parallel(128, 4):
                A_frag[i, j] = S_frag[j]
            for i, j in T.Parallel(128, 4):
                B_frag[i, j] = S_frag[j]

    return main


def test_tilelang_issue_layout_free_inference_choose_smallest_replication():
    kernel = _tilelang_issue_layout_free_inference_choose_smallest_replication()
    source = kernel.get_kernel_source()
    assert "float S_frag[4];" in source, "S_frag is not in the source"
    assert "float B_frag[4];" in source, "B_frag is not in the source"
    assert "float A_frag[4];" in source, "A_frag is not in the source"


@tilelang.jit
def _invalid_fragment_write_owner_layout():
    def token_loop_layout_fn(i, j, rep):
        return i * 4 + rep % 4 + (rep // 4) * 32, j

    def scalar_buffer_layout_fn(i, j, rep):
        return i * 4 + j + rep * 32, 0

    loop_layout = T.Fragment((8, 4), forward_fn=token_loop_layout_fn, replicate=8)
    buffer_layout = T.Fragment((8, 4), forward_fn=scalar_buffer_layout_fn, replicate=2)

    @T.prim_func
    def main():
        with T.Kernel(1, threads=64):
            frag = T.alloc_fragment((8, 4), T.float32)
            T.annotate_layout({frag: buffer_layout})
            for i, j in T.Parallel(8, 4, loop_layout=loop_layout):
                frag[i, j] = T.float32(1.0)

    return main


def test_reject_fragment_write_from_non_owner_threads():
    with pytest.raises(Exception, match="Layout infer conflict"):
        _invalid_fragment_write_owner_layout()


@tilelang.jit
def _scalar_reduce_accumulator_owner_layout():
    @T.prim_func
    def main(out: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            acc = T.alloc_fragment((128,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            total[0] = T.float32(0.0)
            for k in T.Parallel(128):
                acc[k] = T.float32(1.0)
            T.reduce_sum(acc, total, clear=False)
            if T.get_thread_binding() == 0:
                out[0] = total[0]

    return main


def test_scalar_reduce_accumulator_owner_layout():
    kernel = _scalar_reduce_accumulator_owner_layout()
    source = kernel.get_kernel_source()
    assert "AllReduce" in source


def _lower_fully_replicated_readback(use_copy: bool) -> str:
    @T.prim_func
    def main(inp: T.Tensor((128,), T.float32), out: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((128,), T.float32)
            T.annotate_layout({fragment: tilelang.layout.make_fully_replicated_layout_fragment(fragment, 128)})

            for i in T.serial(128):
                fragment[i] = inp[i]

            if use_copy:
                T.copy(fragment, out)
            else:
                for i in T.Parallel(128):
                    out[i] = fragment[i]

    target = Target("maca")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER.value: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED.value: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK.value: True,
    }
    with tvm.transform.PassContext(config=pass_configs), target:
        artifact = tilelang.lower(main, target=target)
    return artifact.kernel_source


def _assert_single_thread_replicated_readback(src: str) -> None:
    assert not re.search(r"fragment\[[^\]\n]*threadIdx\.x", src), src
    assert not re.search(r"out\[[^\]\n]*threadIdx\.x", src), src
    assert "if (((int)threadIdx.x) == 0)" in src
    assert re.search(r"for \(int i(_\d+)? = 0; i(_\d+)? < 128; \+\+i(_\d+)?\)", src), src


@tilelang.testing.requires_cuda
def test_parallel_readback_from_fully_replicated_fragment_uses_rep_guard():
    src = _lower_fully_replicated_readback(use_copy=False)
    _assert_single_thread_replicated_readback(src)


@tilelang.testing.requires_cuda
def test_copy_readback_from_fully_replicated_fragment_uses_rep_guard():
    src = _lower_fully_replicated_readback(use_copy=True)
    _assert_single_thread_replicated_readback(src)


# [MACA] Regression for Parallel subregion stores into an MMA fragment under
# multi-warp Square (warp=64, threads=128 → often (2,1)): projected
# forward_thread for the lower-M half has min>0; without thread>=min guards in
# ParallelOp, non-owner threads corrupt the fragment before gemm.
@tilelang.jit(out_idx=[-1])
def _quadrant_parallel_fragment_write_gemm():
    """4x Parallel(32,32) writes into one 64x64 fragment then T.gemm.

    Reproduces the SSD microtile failure mode: non-common-index Parallel
    writes must project the full MMA fragment layout (thread + local index).
    """
    tile = 64
    half = 32

    @T.prim_func
    def main(
        A_in: T.Tensor((tile, tile), T.float16),
        B_in: T.Tensor((tile, tile), T.float16),
        C_out: T.Tensor((tile, tile), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_frag = T.alloc_fragment((tile, tile), T.float16)
            B_shared = T.alloc_shared((tile, tile), T.float16)
            C_frag = T.alloc_fragment((tile, tile), T.float32)
            T.annotate_layout({B_shared: tilelang.layout.make_swizzled_layout(B_shared)})
            T.clear(C_frag)

            for i, j in T.Parallel(tile, tile):
                B_shared[i, j] = B_in[i, j]

            # Four quadrant writes with non-common indices on A_frag.
            for i, j in T.Parallel(half, half):
                A_frag[i, j] = A_in[i, j]
            for i, j in T.Parallel(half, half):
                A_frag[i, half + j] = A_in[i, half + j]
            for i, j in T.Parallel(half, half):
                A_frag[half + i, j] = A_in[half + i, j]
            for i, j in T.Parallel(half, half):
                A_frag[half + i, half + j] = A_in[half + i, half + j]

            T.gemm(A_frag, B_shared, C_frag)

            for i, j in T.Parallel(tile, tile):
                C_out[i, j] = C_frag[i, j]

    return main


@tilelang.testing.requires_cuda
def test_quadrant_parallel_fragment_write_gemm_numeric():
    """[MACA] Numeric check for the Parallel fragment-owner lower-bound fix."""
    import torch

    kernel = _quadrant_parallel_fragment_write_gemm()
    tile = 64
    a = torch.randn(tile, tile, dtype=torch.float16, device="cuda")
    b = torch.randn(tile, tile, dtype=torch.float16, device="cuda")
    c = kernel(a, b)
    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
