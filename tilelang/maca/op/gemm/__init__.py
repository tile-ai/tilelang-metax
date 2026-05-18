"""MACA GEMM op registrations."""

from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from .gemm_mma import GEMM_INST_MMA, GemmMMA
from tilelang.utils.target import target_is_maca


register_gemm_impl("maca.mma", GEMM_INST_MMA, target_is_maca, GemmMMA)
