from pathlib import Path


def test_maca_builtin_registration_preserves_ptx_bulk_shared_effect_attr():
    repo_root = Path(__file__).resolve().parents[3]
    builtin_source = repo_root / "src" / "op" / "builtin.cc"

    source = builtin_source.read_text()
    ptx_bulk_segment = source.split("TIR_DEFINE_TL_BUILTIN(ptx_st_bulk_shared)", 1)[1].split(
        "TIR_DEFINE_TL_BUILTIN(maca_ldg_b128_bsm_predicator)",
        1,
    )[0]

    assert ".set_num_inputs(3)" in ptx_bulk_segment
    assert '.set_attr<TCallEffectKind>("TCallEffectKind",' in ptx_bulk_segment
    assert "Integer(CallEffectKind::kOpaque)" in ptx_bulk_segment


def test_maca_dense_template_normalizes_partitioned_fragments_before_gemm():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_header = repo_root / "src" / "tl_templates" / "maca" / "gemm.h"

    source = gemm_header.read_text()
    normalized = " ".join(source.split())

    assert "static CUTE_DEVICE auto remove_swizzle(Layout<Args...> const &layout)" in normalized
    assert "static CUTE_DEVICE auto remove_swizzle(ComposedLayout<Args...> const &layout)" in normalized
    assert source.count("CUTE_UNROLL") >= 2
    assert source.count("auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));") >= 2
    assert source.count("gemm(tiled_mma, tCrA(_, _, k), tCrB_view(_, _, k), acc);") >= 2
    assert "Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);" in source
    assert source.count("copy(tiled_copy_B, tCsB(_, _, 0), tCrB_copy_view(_, _, 0));") >= 2
    assert source.count("copy(tiled_copy_B, tCsB(_, _, k + 1), tCrB_copy_view(_, _, k + 1));") >= 2


def test_maca_dense_template_uses_maca_cute_composed_layout_accessor():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_header = repo_root / "src" / "tl_templates" / "maca" / "gemm.h"

    source = gemm_header.read_text()
    composed_overload = source.split("remove_swizzle(ComposedLayout<Args...> const &layout)", 1)[1].split(
        "\n  }\n\n  CUTE_DEVICE static void body",
        1,
    )[0]

    assert "layout.layout_b()" not in source
    assert "layout.layout_fn()" in source
    assert "return layout;" not in composed_overload


def test_maca_dense_template_uses_transpose_aware_shared_layouts():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_header = repo_root / "src" / "tl_templates" / "maca" / "gemm.h"
    gemm_mma = repo_root / "tilelang" / "maca" / "op" / "gemm" / "gemm_mma.py"

    header_source = gemm_header.read_text()
    lowering_source = gemm_mma.read_text()

    assert "using LinearSmemLayoutA = Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, _1>>;" in header_source
    assert "using LinearSmemLayoutB = Layout<Shape<Int<N>, Int<K>>, Stride<_1, Int<N>>>;" in header_source
    assert "typename std::conditional<trans_B, typename OperandATraits::Layout," in header_source
    assert "typename std::conditional<trans_B, typename OperandBTraits::Layout," in header_source
    template_section = lowering_source.split("if use_template and self.is_gemm_ss():", 1)[1].split("else:\n            shared_layout_a", 1)[
        0
    ]
    assert "if self.trans_B:" in template_section
    assert "make_maca_gemm_ab_layout(buf, 1 if self.trans_A else 2)" in template_section
    assert "make_maca_gemm_ab_layout(buf, 2)" in template_section
    assert "shared_layout_a = make_linear_layout" in template_section
    assert "shared_layout_b = make_linear_layout" in template_section


def test_maca_dense_template_remains_opt_in_by_default():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_mma = repo_root / "tilelang" / "maca" / "op" / "gemm" / "gemm_mma.py"

    source = gemm_mma.read_text()

    assert "def _get_maca_gemm_use_template(default: bool = False) -> bool:" in source
    assert "use_template = _get_maca_gemm_use_template(default=False)" in source
    assert source.count("use_template = _get_maca_gemm_use_template(default=False)") >= 2


def test_maca_dense_template_fragment_c_matches_cute_partition_order():
    repo_root = Path(__file__).resolve().parents[3]
    layout_source = repo_root / "src" / "layout" / "gemm_layouts.cc"

    source = layout_source.read_text()
    maca_fragment = source.split("Fragment makeGemmFragmentCMACA", 1)[1].split("Fragment makeGemmFragmentCHopper", 1)[0]

    assert "base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false)" in maca_fragment
    assert "thread_layout->Repeat({warp_m / 16, warp_n / 16}, false, false)" in maca_fragment
    assert "base_layout->Repeat({warp_m / 16, warp_n / 16}, false, true)" not in maca_fragment


def test_maca_wsm_template_declares_supported_contract():
    repo_root = Path(__file__).resolve().parents[3]
    wsm_header = repo_root / "src" / "tl_templates" / "maca" / "gemm_wsm.h"

    source = wsm_header.read_text()

    assert "static_assert(!trans_A && !trans_B" in source
    assert "static_assert(num_warp_m == 1 && num_warp_n == 1" in source
    assert "static_assert(kPack == 8" in source
    assert "static_assert(AStrideElements % 8 == 0" in source
    assert "static_assert(sizeof(A_type) == 2 && sizeof(B_type) == 2" in source
    assert "static_assert(sizeof(C_type) == 4" in source


def test_maca_wsm_lowering_names_workspace_size():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_mma = repo_root / "tilelang" / "maca" / "op" / "gemm" / "gemm_mma.py"

    source = gemm_mma.read_text()

    assert "MACA_WSM_STAGE_BYTES = 0x4000" in source
    assert "MACA_WSM_STAGE_COUNT = 4" in source
    assert "MACA_WSM_WORKSPACE_BYTES = MACA_WSM_STAGE_BYTES * MACA_WSM_STAGE_COUNT" in source
    assert "T.alloc_shared((MACA_WSM_WORKSPACE_BYTES,)" in source


def test_maca_wsm_lowering_falls_back_for_unsupported_contracts():
    repo_root = Path(__file__).resolve().parents[3]
    gemm_mma = repo_root / "tilelang" / "maca" / "op" / "gemm" / "gemm_mma.py"

    source = gemm_mma.read_text()

    assert "def _can_use_maca_gemm_wsm(" in source
    assert "not trans_a" in source
    assert "not trans_b" in source
    assert "num_warp_m == 1" in source
    assert "num_warp_n == 1" in source
    assert "k_pack == 8" in source
    assert "a_source_stride % 8 == 0" in source
    assert 'consumer_surface = "direct_tl_gemm_ss"' in source


def test_maca_template_long_k_regression_remains_on_generic_gemm_surface():
    repo_root = Path(__file__).resolve().parents[3]
    regression = repo_root / "examples" / "gemm" / "regression_example_gemm.py"

    source = regression.read_text()

    assert "bench_gemm_maca_baseline_m1664_n1024_k262144" in source
    assert "bench_gemm_maca_template_m1664_n1024_k262144" in source
    assert '"M": 1664' in source
    assert '"N": 1024' in source
    assert '"K": 262144' in source
    assert '"TILELANG_MACA_GEMM_USE_TEMPLATE": "1"' in source
    assert '"TILELANG_MACA_GEMM_K_PACK": "1"' in source
    assert source.count('"TILELANG_DEFAULT_TARGET": "maca"') == 3
    assert '"TILELANG_MACA_GEMM_USE_TEMPLATE": None' in source
    assert '"TILELANG_MACA_GEMM_K_PACK": None' in source
    assert "def _bench_gemm_maca_template_matmul" in source
    assert "def _run_bench_gemm_maca_baseline" in source
    assert "with _temporary_env(_MACA_BASELINE_ENV):" in source
    assert "def _run_bench_gemm_maca_template" in source
    assert "with _temporary_env(_MACA_TEMPLATE_ENV):" in source
    assert "maca-baseline-long-k" in source
    assert "maca-template-long-k" in source
    assert "maca-template-pair-long-k" in source
    assert '"regression_bench_gemm_maca_baseline_m1664_n1024_k262144"' in source
    assert '"regression_bench_gemm_maca_template_m1664_n1024_k262144"' in source
    assert "_MACA_GEMM_CHECK_SHAPE = (128, 128, 128)" in source
    assert "def _check_bench_gemm_maca_pair() -> None:" in source
    assert "kernel.get_profiler().assert_allclose(lambda a, b: a @ b, atol=1e-2, rtol=1e-2)" in source
    assert 'parser.add_argument("--check", action="store_true")' in source
    assert source.count("T.copy(A[by * block_M, ko * block_K], A_shared)") >= 2
    assert source.count("T.copy(B[ko * block_K, bx * block_N], B_shared)") >= 2
    assert source.count("T.gemm(A_shared, B_shared, C_local)") >= 2
    assert source.count("T.copy(C_local, C[by * block_M, bx * block_N])") >= 2


def test_maca_kpack2_long_k_regression_uses_non_template_lowering():
    repo_root = Path(__file__).resolve().parents[3]
    regression = repo_root / "examples" / "gemm" / "regression_example_gemm.py"
    gemm_mma = repo_root / "tilelang" / "maca" / "op" / "gemm" / "gemm_mma.py"

    regression_source = regression.read_text()
    lowering_source = gemm_mma.read_text()

    assert "bench_gemm_maca_kpack2_m1664_n1024_k262144" in regression_source
    assert '"TILELANG_MACA_GEMM_USE_TEMPLATE": None' in regression_source
    assert '"TILELANG_MACA_GEMM_K_PACK": "2"' in regression_source
    assert "maca-kpack2-long-k" in regression_source
    assert "maca-kpack2-pair-long-k" in regression_source
    assert "def _bench_gemm_maca_kpack2_matmul" in regression_source
    assert "def _run_bench_gemm_maca_kpack2" in regression_source
    assert "with _temporary_env(_MACA_KPACK2_ENV):" in regression_source
    assert "def _check_bench_gemm_maca_kpack2_pair() -> None:" in regression_source
    assert "(_MACA_KPACK2_ENV, _bench_gemm_maca_kpack2_matmul)" in regression_source
    assert regression_source.count("T.copy(A[by * block_M, ko * block_K], A_shared)") >= 3
    assert regression_source.count("T.copy(B[ko * block_K, bx * block_N], B_shared)") >= 3
    assert regression_source.count("T.gemm(A_shared, B_shared, C_local)") >= 3
    assert regression_source.count("T.copy(C_local, C[by * block_M, bx * block_N])") >= 3
    assert "def _get_maca_gemm_k_pack(default: int = 1) -> int:" in lowering_source
    assert 'os.environ.get("TILELANG_MACA_GEMM_K_PACK")' in lowering_source
    assert "k_pack = _get_maca_gemm_k_pack(self.k_pack)" in lowering_source
    assert "macro_size_k = micro_size_k * k_pack" in lowering_source
    assert "A_local = T.alloc_local((warp_rows * local_size_a * k_pack), a_dtype)" in lowering_source
    assert "B_local = T.alloc_local((warp_cols * local_size_b * k_pack), b_dtype)" in lowering_source


def test_maca_gemm_layout_entrypoint_uses_the_tirx_buffer_type():
    repo_root = Path(__file__).resolve().parents[3]
    layout_header = repo_root / "src" / "layout" / "layout.h"

    source = layout_header.read_text()

    assert "Layout makeMacaGemmABLayout(const tirx::Buffer &buffer, int kfactor);" in source
