from __future__ import annotations

import os

from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
    make_linear_layout,
    make_maca_gemm_ab_layout,
    make_maca_gemm_fragment_c,
    make_swizzled_layout,
)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx
from tilelang import language as T
from tilelang.transform.simplify import _Simplify
from tilelang.maca.intrinsics.macro.mma_macro_generator import (
    TensorCoreIntrinEmitter,
)


def _get_maca_gemm_k_pack(default: int = 1) -> int:
    value = os.environ.get("TILELANG_MACA_GEMM_K_PACK")
    if value is None:
        return default
    try:
        k_pack = int(value)
    except ValueError as exc:
        raise ValueError(f"TILELANG_MACA_GEMM_K_PACK must be an integer, got {value!r}") from exc
    if k_pack < 1:
        raise ValueError(f"TILELANG_MACA_GEMM_K_PACK must be >= 1, got {k_pack}")
    return k_pack


def _get_maca_gemm_use_template(default: bool = False) -> bool:
    value = os.environ.get("TILELANG_MACA_GEMM_USE_TEMPLATE")
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "f", "no", "n", ""}


def _get_maca_gemm_consumer_surface(default: str = "direct_tl_gemm_ss") -> str:
    value = os.environ.get("TILELANG_MACA_GEMM_CONSUMER_SURFACE")
    if value is None or not value.strip():
        return default
    surface = value.strip().lower()
    if surface == "not_direct_tl_gemm_ss":
        return "wsm_aware"
    if surface not in {"direct_tl_gemm_ss", "wsm_aware"}:
        valid = "direct_tl_gemm_ss, wsm_aware, not_direct_tl_gemm_ss"
        raise ValueError(f"TILELANG_MACA_GEMM_CONSUMER_SURFACE must be one of {valid}, got {value!r}")
    return surface


def _make_maca_gemm_emitter(**kwargs):
    return TensorCoreIntrinEmitter(**kwargs)


def _resolve_maca_gemm_shared_layout(value: str, env_key: str):
    layout_name = value.strip().lower()
    layout_name = {
        "default": "swizzled",
        "auto": "swizzled",
    }.get(layout_name, layout_name)

    layout_factories = {
        "swizzled": make_swizzled_layout,
        "quarter": make_quarter_bank_swizzled_layout,
        "half": make_half_bank_swizzled_layout,
        "full": make_full_bank_swizzled_layout,
        "linear": make_linear_layout,
    }
    if layout_name not in layout_factories:
        valid = ", ".join(sorted(layout_factories))
        raise ValueError(f"{env_key} must be one of {valid}, got {value!r}")
    return layout_factories[layout_name]


def _get_maca_gemm_shared_layout():
    value = os.environ.get("TILELANG_MACA_GEMM_SHARED_LAYOUT")
    if value is None or not value.strip():
        return make_swizzled_layout
    return _resolve_maca_gemm_shared_layout(value, "TILELANG_MACA_GEMM_SHARED_LAYOUT")


def _get_maca_gemm_shared_layout_for_operand(operand: str):
    operand = operand.upper()
    specific_key = f"TILELANG_MACA_GEMM_SHARED_LAYOUT_{operand}"
    value = os.environ.get(specific_key)
    if value is None or not value.strip():
        return _get_maca_gemm_shared_layout()
    return _resolve_maca_gemm_shared_layout(value, specific_key)


def _format_maca_gemm_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def _make_maca_gemm_template_name(
    kind: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warp_m: int,
    num_warp_n: int,
    trans_a: bool,
    trans_b: bool,
    clear_accum: bool,
    k_pack: int,
    extra_template_args: tuple[int, ...] = (),
) -> str:
    template_args = [
        str(block_m),
        str(block_n),
        str(block_k),
        str(num_warp_m),
        str(num_warp_n),
        _format_maca_gemm_bool(trans_a),
        _format_maca_gemm_bool(trans_b),
        _format_maca_gemm_bool(clear_accum),
        str(k_pack),
        *(str(value) for value in extra_template_args),
    ]
    return f"tl::gemm_{kind}<" + ", ".join(template_args) + ">"


GEMM_INST_MMA = "maca.mma"
MACA_WSM_STAGE_BYTES = 0x4000
MACA_WSM_STAGE_COUNT = 4
MACA_WSM_WORKSPACE_BYTES = MACA_WSM_STAGE_BYTES * MACA_WSM_STAGE_COUNT


def _maca_dtype_name(dtype) -> str:
    return str(dtype).lower()


def _can_use_maca_gemm_wsm(
    *,
    trans_a: bool,
    trans_b: bool,
    num_warp_m: int,
    num_warp_n: int,
    k_pack: int,
    a_source_stride: int,
    in_dtype,
    accum_dtype,
) -> bool:
    return (
        not trans_a
        and not trans_b
        and num_warp_m == 1
        and num_warp_n == 1
        and k_pack == 8
        and a_source_stride % 8 == 0
        and _maca_dtype_name(in_dtype) in {"float16", "half"}
        and _maca_dtype_name(accum_dtype) in {"float", "float32"}
    )


class GemmMMA(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_MMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        k_pack = _get_maca_gemm_k_pack(self.k_pack)
        use_template = _get_maca_gemm_use_template(default=False)
        mma_emitter = _make_maca_gemm_emitter(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            k_pack=k_pack,
        )
        if use_template and self.is_gemm_ss():
            if self.trans_B:

                def shared_layout_a(buf):
                    return make_maca_gemm_ab_layout(buf, 1 if self.trans_A else 2)

                def shared_layout_b(buf):
                    return make_maca_gemm_ab_layout(buf, 2)

            else:
                shared_layout_a = make_linear_layout
                shared_layout_b = make_linear_layout

            c_layout = make_maca_gemm_fragment_c(
                int(self.M),
                int(self.N),
                int(warp_row_tiles),
                int(warp_col_tiles),
                self.C.dtype.bits,
            )
        else:
            shared_layout_a = _get_maca_gemm_shared_layout_for_operand("A")
            shared_layout_b = _get_maca_gemm_shared_layout_for_operand("B")
            c_layout = None
        if self.is_gemm_ss():
            return {
                self.A: shared_layout_a(self.A),
                self.B: shared_layout_b(self.B),
                self.C: c_layout if c_layout is not None else mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: shared_layout_a(self.A),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        local_thread_var = thread_var - thread_bounds.min
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_MMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        k_pack = _get_maca_gemm_k_pack(self.k_pack)
        mma_emitter = _make_maca_gemm_emitter(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            k_pack=k_pack,
            thread_var=local_thread_var,
        )

        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        warp_rows = mma_emitter.warp_rows
        warp_cols = mma_emitter.warp_cols
        local_size_a = mma_emitter.local_size_a
        local_size_b = mma_emitter.local_size_b
        block_K = mma_emitter.chunk
        micro_size_k = mma_emitter.micro_size_k
        k_pack = mma_emitter.k_pack
        macro_size_k = micro_size_k * k_pack
        # We use region for memory input to support strided gemm
        # T.gemm(A_shared[0:128, :], B_shared, C_local)
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        clear_accum = self.clear_accum
        use_template = _get_maca_gemm_use_template(default=False)
        consumer_surface = _get_maca_gemm_consumer_surface()

        assert block_K >= macro_size_k, f"block_K ({block_K}) must be >= macro_size_k ({macro_size_k})"
        assert block_K % macro_size_k == 0, f"block_K ({block_K}) must be divisible by macro_size_k ({macro_size_k})"

        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_gemm_ss():
            if use_template:
                extra_template_args: tuple[int, ...] = ()
                if consumer_surface == "wsm_aware":
                    a_source_stride = self.annotations.get("maca_wsm_a_stride", None)
                    if a_source_stride is None:
                        a_source_stride = int(self.K)
                    a_source_stride = int(a_source_stride)
                    if _can_use_maca_gemm_wsm(
                        trans_a=bool(self.trans_A),
                        trans_b=bool(self.trans_B),
                        num_warp_m=int(m_warp),
                        num_warp_n=int(n_warp),
                        k_pack=int(k_pack),
                        a_source_stride=a_source_stride,
                        in_dtype=self.a_dtype,
                        accum_dtype=self.accum_dtype,
                    ):
                        extra_template_args = (a_source_stride,)
                    else:
                        consumer_surface = "direct_tl_gemm_ss"
                gemm_kind = "ss_wsm" if consumer_surface == "wsm_aware" else "ss"
                op_instance = _make_maca_gemm_template_name(
                    gemm_kind,
                    int(self.M),
                    int(self.N),
                    int(self.K),
                    int(m_warp),
                    int(n_warp),
                    bool(self.trans_A),
                    bool(self.trans_B),
                    bool(clear_accum),
                    int(k_pack),
                    extra_template_args=extra_template_args,
                )

                if consumer_surface == "wsm_aware":
                    a_source_ptr = self.annotations.get("maca_wsm_a_source_ptr")
                    b_source_ptr = self.annotations.get("maca_wsm_b_source_ptr")
                    if a_source_ptr is None:
                        a_source_ptr = T.access_ptr(A_region, "r")
                    if b_source_ptr is None:
                        b_source_ptr = T.access_ptr(B_region, "r")

                    @T.prim_func
                    def _gemm_ss_wsm_template() -> None:
                        WSM = T.alloc_shared((MACA_WSM_WORKSPACE_BYTES,), T.uint8, scope="shared")
                        T.call_intrin(
                            "handle",
                            tirx.op.Op.get("tl.tl_gemm_wsm"),
                            op_instance,
                            T.access_ptr(A_region, "r"),
                            T.access_ptr(B_region, "r"),
                            T.access_ptr(C_region, "rw"),
                            T.address_of(WSM[0]),
                            a_source_ptr,
                            b_source_ptr,
                        )

                    return _Simplify(_gemm_ss_wsm_template, inline_let=True)

                @T.prim_func
                def _gemm_ss_template() -> None:
                    T.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tl_gemm"),
                        op_instance,
                        T.access_ptr(A_region, "r"),
                        T.access_ptr(B_region, "r"),
                        T.access_ptr(C_region, "rw"),
                    )

                return _Simplify(_gemm_ss_template, inline_let=True)

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a * k_pack), a_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b * k_pack), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                if self.mbar is not None:
                    T.maca_barrier_arrive_and_wait(self.mbar)
                num_iters = block_K // macro_size_k
                pipeline_stages = 4 if num_iters >= 4 else 0
                for ki in T.Pipelined(num_iters, num_stages=pipeline_stages):
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a * k_pack), a_dtype)

                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // macro_size_k)):
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            # alloc_buffers body
            # insert into parent block
            return _Simplify(_gemm_srr, inline_let=True)
        elif self.is_gemm_rs():
            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                B_local = T.alloc_local((warp_cols * local_size_b * k_pack), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // macro_size_k)):
                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_buf, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        elif self.is_gemm_rr():
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rrr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """

                for ki in T.serial(0, (block_K // macro_size_k)):
                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_buf, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rrr, inline_let=True)
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
