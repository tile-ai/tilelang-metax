"""MACA language dialect: common TileLang plus MACA extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.annotations import annotate_l2_hit_ratio, annotate_min_blocks_per_sm  # noqa: F401
from tilelang.language.builtin import (  # noqa: F401
    annotate_consumer_reg_alloc,
    annotate_producer_reg_dealloc,
    deallocate_tmem,
    dec_max_nreg,
    disable_warp_group_reg_alloc,
    get_warp_group_idx,
    inc_max_nreg,
    increase_descriptor_offset,
    ldg128,
    ldg256,
    ldg32,
    ldg64,
    lds128,
    lds32,
    lds64,
    match_all_sync,
    match_any_sync,
    set_max_nreg,
    shuffle_elect,
    stg128,
    stg256,
    stg32,
    stg64,
    sts128,
    sts32,
    sts64,
)
from tilelang.language.copy_op import maca_async_copy  # noqa: F401
from tilelang.language.kernel import CUDASourceCodeKernel  # noqa: F401

from .intrinsics import *  # noqa: F401,F403
from .intrinsics import __all__ as _INTRINSICS_ALL
from .pdl import *  # noqa: F401,F403
from .pdl import __all__ as _PDL_ALL
from .print import *  # noqa: F401,F403
from .print import __all__ as _PRINT_ALL
from .random import *  # noqa: F401,F403
from .random import __all__ as _RANDOM_ALL
from .tir import *  # noqa: F401,F403
from .tir import __all__ as _TIR_ALL
from .warpgroup import *  # noqa: F401,F403
from .warpgroup import __all__ as _WARPGROUP_ALL

_MACA_API_ALL = (
    "CUDASourceCodeKernel",
    "annotate_consumer_reg_alloc",
    "annotate_l2_hit_ratio",
    "annotate_min_blocks_per_sm",
    "annotate_producer_reg_dealloc",
    "deallocate_tmem",
    "dec_max_nreg",
    "disable_warp_group_reg_alloc",
    "get_warp_group_idx",
    "inc_max_nreg",
    "increase_descriptor_offset",
    "ldg128",
    "ldg256",
    "ldg32",
    "ldg64",
    "lds128",
    "lds32",
    "lds64",
    "match_all_sync",
    "match_any_sync",
    "set_max_nreg",
    "shuffle_elect",
    "stg128",
    "stg256",
    "stg32",
    "stg64",
    "sts128",
    "sts32",
    "sts64",
    "maca_async_copy",
)

__tilelang_dialect__ = "maca"
__all__ = tuple(
    dict.fromkeys(
        (
            *_COMMON_ALL,
            *_MACA_API_ALL,
            *_INTRINSICS_ALL,
            *_PDL_ALL,
            *_PRINT_ALL,
            *_RANDOM_ALL,
            *_TIR_ALL,
            *_WARPGROUP_ALL,
        )
    )
)

del _COMMON_ALL, _MACA_API_ALL, _INTRINSICS_ALL, _PDL_ALL, _PRINT_ALL, _RANDOM_ALL, _TIR_ALL, _WARPGROUP_ALL
