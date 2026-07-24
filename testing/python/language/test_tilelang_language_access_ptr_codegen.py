import tilelang
import tilelang.language as T
import tilelang.testing
import pytest
from tilelang import tvm
from tilelang.maca.target import check_maca_availability


requires_maca = pytest.mark.skipif(not check_maca_availability(), reason="Requires MACA")


@tilelang.testing.skip_on_maca
@tilelang.testing.requires_cuda
def test_access_ptr_cp_async_codegen():
    """Smoke-test codegen for T.access_ptr -> tl.access_ptr -> tvm_access_ptr -> cp.async."""

    @T.prim_func
    def main(
        A: T.Tensor((64,), T.uint8),
        B: T.Tensor((64,), T.uint8),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((64,), T.uint8)
            T.ptx_cp_async(
                T.access_ptr(S[8], "w", 16),
                T.access_ptr(A[16], "r", 16),
                16,
            )
            # Keep the shared buffer live so the pointers remain in generated code.
            B[0] = S[8]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== access_ptr cp.async codegen ===")
    print(src)
    assert "cp_async_gs<16>" in src, "Expected cp_async_gs<16> in generated CUDA source"


@tilelang.testing.skip_on_maca
@tilelang.testing.requires_cuda
def test_vectorized_cp_async_num_elems_codegen():
    """Check vectorized tl.ptx_cp_async widens logical element counts."""

    @T.prim_func
    def main(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((64,), T.float16)
            for i in T.vectorized(4):
                T.ptx_cp_async(
                    T.access_ptr(S[i], "w", 1),
                    T.access_ptr(A[i], "r", 1),
                    1,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            B[0] = S[0]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== vectorized cp.async codegen ===")
    print(src)
    assert "cp_async_gs<8>" in src, "Expected vectorized cp.async to fold 4 x fp16 elems into cp_async_gs<8>"
    assert "cp_async_gs<2>" not in src, "Did not expect scalar cp.async width in generated CUDA source"


@tilelang.testing.skip_on_maca
@tilelang.testing.requires_cuda
def test_vectorized_int4_cp_async_num_elems_codegen():
    """Check subbyte tl.ptx_cp_async derives PTX bytes from logical element counts."""

    @T.prim_func
    def main(
        A: T.Tensor((128,), T.int4),
        B: T.Tensor((128,), T.int4),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((128,), T.int4)
            for i in T.vectorized(32):
                T.ptx_cp_async(
                    T.access_ptr(S[i], "w", 1),
                    T.access_ptr(A[i], "r", 1),
                    1,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            B[0] = S[0]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== vectorized int4 cp.async codegen ===")
    print(src)
    assert "cp_async_gs<16>" in src, "Expected 32 x int4 elems to fold into cp_async_gs<16>"


@tilelang.testing.skip_on_maca
@tilelang.testing.requires_cuda
def test_async_copy_tileop_lowers_to_cp_async():
    """Check T.async_copy always uses CPAsync path and does not auto-wait."""

    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((4,), T.float16)
            T.async_copy(A[0:4], S)
            T.copy(S, B[0:4])

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== async_copy -> cp.async codegen ===")
    print(src)
    assert "cp_async_gs<8>" in src, "Expected T.async_copy to lower to cp_async_gs<8>"
    assert "tl::cp_async_commit" in src, "Expected async_copy lowering to emit commit"
    assert "tl::cp_async_wait<0>" not in src, "Did not expect async_copy lowering to auto-emit wait"


@tilelang.testing.skip_on_maca
@tilelang.testing.requires_cuda
def test_async_copy_tileop_rejects_invalid_cp_async_scope():
    """Check T.async_copy rejects non global->shared patterns."""

    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S0 = T.alloc_shared((4,), T.float16)
            S1 = T.alloc_shared((4,), T.float16)
            T.copy(A[0:4], S0)
            # shared->shared cannot use cp.async and should fail for async_copy.
            T.async_copy(S0, S1)
            T.copy(S1, B[0:4])

    with pytest.raises(
        tvm.error.InternalError,
        match="T\\.async_copy only supports global->shared/shared\\.dyn copies",
    ):
        tilelang.compile(main, out_idx=[1], target="cuda")


@tilelang.testing.skip_on_maca
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_parallel_simt_copy_respects_enable_async_copy_config():
    """Check `tl.enable_async_copy=False` disables auto cp.async rewriting."""

    @T.prim_func
    def main(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128,), T.float32)
            for i in T.Parallel(128):
                S[i] = A[i]
            B[0] = S[0]

    kernel = tilelang.compile(
        main,
        out_idx=[1],
        target="cuda",
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: False},
    )
    src = kernel.get_kernel_source()
    print("=== Parallel SIMT copy (async disabled) codegen ===")
    print(src)
    assert "cp_async_gs<" not in src, "Did not expect cp_async_gs when async copy is disabled"


@tilelang.testing.skip_on_maca
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_async_copy_oob_lowers_to_predicated_cp_async_without_wait():
    """Check T.async_copy supports OOB via predicated cp.async and does not auto-wait."""

    M = 130
    K = 32
    block_m = 128
    block_k = 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, block_m)) as pid_m:
            S = T.alloc_shared((block_m, block_k), T.float16)
            T.async_copy(A[pid_m * block_m : (pid_m + 1) * block_m, 0:block_k], S)
            # Don't read S here (no wait). Keep B live so kernel has an output.
            B[0, 0] = A[0, 0]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== OOB async_copy -> predicated cp.async codegen ===")
    print(src)
    assert "cp_async_gs_conditional<" in src, "Expected predicated cp.async (zero-fill) in generated CUDA source"
    assert "tl::cp_async_commit" in src, "Expected async_copy lowering to emit commit"
    assert "tl::cp_async_wait<0>" not in src, "Did not expect async_copy lowering to auto-emit wait"


@requires_maca
def test_maca_global_atomic_add_preserves_logical_layout_indices_codegen():
    def make_dq_layout(dQ):
        return T.Layout(
            dQ.shape,
            lambda b, h, l, d: [
                b,
                h,
                l // 8,
                d // 8,
                (d % 2),
                4 * (l % 8) + (d % 8) // 2,
            ],
        )

    @T.prim_func
    def main(
        A: T.Tensor((64, 32), T.float16),
        B: T.Tensor((64, 128), T.float16),
        dQ: T.Tensor((1, 32, 512, 128), T.float32),
    ):
        with T.Kernel(32, 1, 1, threads=256) as (bx, by, bz):
            A_shared = T.alloc_shared((64, 32), T.float16)
            B_shared = T.alloc_shared((64, 128), T.float16)
            dq = T.alloc_fragment((32, 128), T.float32)
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(dq)
            for k in range(16):
                T.gemm(A_shared, B_shared, dq, transpose_A=True)
                T.atomic_add(dQ[bz, bx, k * 32 : (k + 1) * 32, :], dq)

    kernel = tilelang.compile(main, out_idx=None, target="maca")
    src = kernel.get_kernel_source()

    assert "AtomicAdd" in src
    assert "(k >> 4) + ((int)blockIdx.x)" not in src
    assert "((k & 15) * 4096)" not in src
    assert "((int)blockIdx.x) * 65536" in src
    assert "(k * 4096)" in src


@requires_maca
def test_maca_bsm_intrinsics_codegen():
    """Smoke-test codegen for the MACA BSM builtin wrappers."""

    @T.prim_func
    def main(
        A: T.Tensor((64,), T.uint8),
        B: T.Tensor((64,), T.uint8),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((64,), T.uint8)
            T.maca_ldg_b128_bsm_predicator(
                T.address_of(S[0]),
                T.address_of(A[0]),
                0,
                True,
                True,
                False,
                True,
                1,
                1,
                "MACA_ICMP_EQ",
            )
            T.maca_arrive_gvmcnt(4)
            T.maca_arrive_bsmcnt(2)
            T.maca_barrier_inst()
            B[0] = S[0]

    kernel = tilelang.compile(main, out_idx=[1], target="maca")
    src = kernel.get_kernel_source()
    print("=== MACA BSM builtin codegen ===")
    print(src)
    assert "__builtin_mxc_ldg_b128_bsm_predicator" in src
    assert "__builtin_mxc_arrive_gvmcnt(4)" in src
    assert "__builtin_mxc_arrive_bsmcnt(2)" in src
    assert "__builtin_mxc_barrier_inst();" in src
    assert '"MACA_ICMP_EQ"' not in src


@requires_maca
def test_maca_bsm_byte_view_feeds_gemm_codegen():
    """BSM byte staging can alias a half view consumed by MACA GEMM lowering."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 64), T.float16),
        B: T.Tensor((128, 64), T.float16),
        C: T.Tensor((128, 128), T.float16),
    ):
        with T.Kernel(1, 1, threads=256):
            A_shared = T.alloc_shared((128, 64), T.float16)
            B_storage = T.alloc_shared((128, 128), T.uint8)
            B_shared = T.view(B_storage, (128, 64), dtype=T.float16)
            C_local = T.alloc_fragment((128, 128), T.float32)
            T.copy(A, A_shared)
            T.clear(C_local)
            T.maca_ldg_b128_bsm_predicator(
                T.address_of(B_storage[0, 0]),
                T.address_of(B[0, 0]),
                0,
                True,
                True,
                False,
                True,
                1,
                1,
                "MACA_ICMP_EQ",
            )
            T.maca_arrive_gvmcnt(0)
            T.maca_barrier_inst()
            T.gemm(A_shared, B_shared, C_local, False, True)
            T.copy(C_local, C)

    kernel = tilelang.compile(main, out_idx=[2], target="maca")
    src = kernel.get_kernel_source()
    assert "__builtin_mxc_ldg_b128_bsm_predicator" in src
    assert "__builtin_mxc_arrive_gvmcnt(0)" in src
    assert "__builtin_mxc_barrier_inst();" in src
    assert "__builtin_mxc_mma_16x16x16f16" in src


if __name__ == "__main__":
    tilelang.testing.main()
