from __future__ import annotations
import os
import torch
import warnings
from functools import cache
from math import prod
from tilelang.contrib import nvcc
from tilelang.utils.tensor import is_float8_dtype, fp8_remove_negative_zeros_
from tilelang import env

# Include version information to ensure different versions use separate caches
from tilelang import __version__

# Define paths
compress_util = os.path.join(env.TILELANG_TEMPLATE_PATH, "tl_templates/cuda/compress_sm90.cu")
# Cache directory for compiled extensions
_CACHE_DIR = os.path.join(env.TILELANG_CACHE_DIR, "sparse_compressor", __version__)
os.makedirs(_CACHE_DIR, exist_ok=True)

original_cxx = os.environ.get("CXX")
from torch.utils.cpp_extension import load, _import_module_from_library

if getattr(torch.version, "maca", None):
    if original_cxx:
        os.environ["CXX"] = original_cxx
    else:
        os.environ.pop("CXX", None)


def _decompose_col_major(index_1d: int, basis: list[int]) -> list[int]:
    res = []
    for extent in basis:
        res.append(index_1d % extent)
        index_1d //= extent
    return res


def _gen_stride(shape_ik: list[int], order: list[int]) -> list[int]:
    stride_ik = [None for _ in range(len(shape_ik))]
    ordered = sorted(enumerate(order), key=lambda item: item[1])
    accum_shape = 1
    for idx, (axis, _) in enumerate(ordered):
        stride_ik[axis] = 1 if idx == 0 else accum_shape
        accum_shape *= shape_ik[axis]
    return stride_ik


@cache
def _sm90_metadata_layout_offsets(rows: int, cols: int, bits: int, block_k: int) -> tuple[int, ...]:
    if block_k > 128:
        block_k = 128

    block_k_atom = 512 // bits
    if block_k % block_k_atom != 0:
        raise ValueError(f"Tile K is too small, which should be at least {block_k_atom} for {bits}-bit sparse metadata")
    num_k = block_k // block_k_atom

    if bits == 32:
        shape_ik = [8, 2, 4, 8 // 8, 2, num_k]
        stride_ik = _gen_stride(shape_ik, [3, 1, 5, 0, 4, 2])
        shape_i, shape_k = shape_ik[:3], shape_ik[3:]
        stride_i, stride_k = stride_ik[:3], stride_ik[3:]
    elif bits == 16:
        shape_ik = [8, 2, 4, 16 // 8, 2, num_k]
        stride_ik = _gen_stride(shape_ik, [3, 1, 5, 0, 4, 2])
        shape_i, shape_k = shape_ik[:3], shape_ik[3:]
        stride_i, stride_k = stride_ik[:3], stride_ik[3:]
    elif bits == 8:
        shape_i, shape_k = [64], [block_k // 8]
        stride_i, stride_k = [block_k // 8], [1]
    else:
        raise NotImplementedError(f"Unsupported sparse metadata dtype width: {bits}")

    rep_i = (rows + 63) // 64
    rep_k = (cols + prod(shape_k) - 1) // prod(shape_k)
    rep_i_stride = prod(shape_i + shape_k)
    shape_i = shape_i + [rep_i]
    stride_i = stride_i + [rep_i_stride]
    rep_k_stride = prod(shape_i + shape_k)
    shape_k = shape_k + [rep_k]
    stride_k = stride_k + [rep_k_stride]

    offsets = []
    for i in range(rows):
        i_decomposed = _decompose_col_major(i, shape_i)
        i_offset = sum(i_decomposed[idx] * stride_i[idx] for idx in range(len(shape_i)))
        for k in range(cols):
            k_decomposed = _decompose_col_major(k, shape_k)
            k_offset = sum(k_decomposed[idx] * stride_k[idx] for idx in range(len(shape_k)))
            offsets.append(i_offset + k_offset)

    expected = rows * cols
    if sorted(offsets) != list(range(expected)):
        raise RuntimeError("SM90 sparse metadata layout is not a valid permutation")
    return tuple(offsets)


def _scatter_sm90_metadata_layout(meta_logical: torch.Tensor, bits: int, block_k: int) -> torch.Tensor:
    rows, cols = meta_logical.shape
    storage = torch.zeros(rows * cols, dtype=meta_logical.dtype)
    offsets = torch.tensor(_sm90_metadata_layout_offsets(rows, cols, bits, block_k), dtype=torch.long)
    storage[offsets] = meta_logical.reshape(-1)
    return storage.view(rows, cols)


def _compress_groups_2to4(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    non_zero_mask = groups != 0
    non_zero_count = non_zero_mask.sum(dim=-1)
    if torch.any(non_zero_count > 2):
        raise ValueError("compress_sm90 fallback expects 2:4 structured sparse input for 8/16-bit dtypes")

    sentinel = groups.shape[-1]
    logical_idx = torch.arange(sentinel, dtype=torch.int64).view(1, 1, sentinel)
    idx = torch.where(non_zero_mask, logical_idx, torch.full_like(logical_idx, sentinel))
    idx = idx.sort(dim=-1).values[..., :2]

    gathered = groups.gather(-1, idx.clamp(max=sentinel - 1))
    gathered = torch.where(idx < sentinel, gathered, torch.zeros_like(gathered))

    has_fewer_than_two = non_zero_count < 2
    has_no_non_zero = non_zero_count == 0
    single_last = (non_zero_count == 1) & (idx[..., 0] == sentinel - 1)
    first_value = gathered[..., 0].clone()

    idx[..., 1] = torch.where(has_fewer_than_two, torch.full_like(idx[..., 1], sentinel - 1), idx[..., 1])
    gathered[..., 1] = torch.where(has_fewer_than_two, torch.zeros_like(gathered[..., 1]), gathered[..., 1])
    idx[..., 0] = torch.where(has_no_non_zero, torch.zeros_like(idx[..., 0]), idx[..., 0])

    idx[..., 0] = torch.where(single_last, torch.zeros_like(idx[..., 0]), idx[..., 0])
    gathered[..., 0] = torch.where(single_last, torch.zeros_like(gathered[..., 0]), gathered[..., 0])
    idx[..., 1] = torch.where(single_last, torch.full_like(idx[..., 1], sentinel - 1), idx[..., 1])
    gathered[..., 1] = torch.where(single_last, first_value, gathered[..., 1])
    return gathered, idx.to(torch.uint8)


def _compress_groups_1to2(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    non_zero_mask = groups != 0
    non_zero_count = non_zero_mask.sum(dim=-1)
    if torch.any(non_zero_count > 1):
        raise ValueError("compress_sm90 fallback expects 1:2 structured sparse input for 32-bit dtypes")

    sentinel = groups.shape[-1]
    logical_idx = torch.arange(sentinel, dtype=torch.int64).view(1, 1, sentinel)
    idx = torch.where(non_zero_mask, logical_idx, torch.full_like(logical_idx, sentinel))
    idx = idx.sort(dim=-1).values[..., 0]
    gathered = groups.gather(-1, idx.clamp(max=sentinel - 1).unsqueeze(-1)).squeeze(-1)
    gathered = torch.where(idx < sentinel, gathered, torch.zeros_like(gathered))
    idx = torch.where(non_zero_count == 0, torch.zeros_like(idx), idx)
    meta = torch.where(idx == 0, torch.full_like(idx, 0b0100), torch.full_like(idx, 0b1110))
    return gathered.unsqueeze(-1), meta.to(torch.uint8)


def _compress_sm90_fallback(A: torch.Tensor, block_k: int, transposed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if not A.is_contiguous():
        raise ValueError("A need to be contiguous")
    if A.dim() != 2:
        raise ValueError("Might support batch dim in the future")

    input_tensor = A.t().contiguous() if transposed else A
    if is_float8_dtype(input_tensor.dtype):
        fp8_remove_negative_zeros_(input_tensor)

    work = input_tensor.to("cpu")
    m, k = work.shape
    bits = work.element_size() * 8

    if bits in (8, 16):
        if k % 4 != 0:
            raise ValueError(f"K dimension must be divisible by 4 for {bits}-bit sparse compression")
        groups = work.view(m, -1, 4)
        compressed_values, metadata_idx = _compress_groups_2to4(groups)
        nibbles = metadata_idx[..., 0] | (metadata_idx[..., 1] << 2)
        meta_logical = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    elif bits == 32:
        if k % 2 != 0:
            raise ValueError("K dimension must be divisible by 2 for 32-bit sparse compression")
        groups = work.view(m, -1, 2)
        compressed_values, nibbles = _compress_groups_1to2(groups)
        meta_logical = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    else:
        raise NotImplementedError(f"MACA sparse compression fallback does not support {bits}-bit inputs")

    A_sparse = compressed_values.reshape(m, -1).contiguous()
    E = _scatter_sm90_metadata_layout(meta_logical.contiguous(), bits, block_k)

    if transposed:
        A_sparse = A_sparse.t().contiguous()

    return A_sparse.to(A.device), E.to(A.device)


def _get_cached_lib():
    name = "compress_lib"

    if os.path.exists(os.path.join(_CACHE_DIR, f"{name}.so")):
        try:
            return _import_module_from_library(name, _CACHE_DIR, is_python_module=True)
        except Exception:
            pass

    # Set TORCH_CUDA_ARCH_LIST
    env._initialize_torch_cuda_arch_flags()

    # Compile if not cached or loading failed
    return load(
        name=name,
        sources=[compress_util],
        extra_cuda_cflags=[
            "-O2",
            "-std=c++17",
            "-lineinfo",
            f"-I{env.CUTLASS_INCLUDE_DIR}",
            f"-I{env.CUTLASS_INCLUDE_DIR}/../tools/util/include",
            "-arch=sm_90",
        ],
        build_directory=_CACHE_DIR,
    )


def compress_sm90(A: torch.Tensor, block_k: int, transposed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if block_k > 128:
        block_k = 128
        # Ref: https://github.com/NVIDIA/cutlass/blob/c2ad7c5b20f131c4ba33601860f1da3f9c9df0f3/include/cutlass/gemm/collective/builders/sm90_sparse_gmma_builder.inl#L145-L146
        warnings.warn(f"block_k {block_k} is too large, set to 128 for sm90 compression.", stacklevel=2)
    if getattr(torch.version, "maca", None):
        return _compress_sm90_fallback(A, block_k, transposed)
    # Load the library (will use cache if available)
    compress_lib = _get_cached_lib()

    return compress_lib.compress_sm90(A, block_k, transposed)


def compress_sm80(A: torch.Tensor, transposed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
    except ImportError as err:
        raise ImportError(
            "SparseSemiStructuredTensor is not available in this version of PyTorch. Please install a compatible version."
        ) from err
    orig_val = SparseSemiStructuredTensor._FORCE_CUTLASS
    try:
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
        if transposed is not False:
            raise NotImplementedError("transposed flag is deprecated by pytorch")
        compressed = to_sparse_semi_structured(A)
        return compressed.packed, compressed.meta
    finally:
        SparseSemiStructuredTensor._FORCE_CUTLASS = orig_val


def compress(A: torch.Tensor, transposed: bool, arch: str | None = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compress a tensor using the appropriate method based on the CUDA architecture.
    """
    if arch is None:
        arch = nvcc.get_target_compute_version()

    compute_version = nvcc.parse_compute_version(arch)

    if compute_version >= (9, 0):
        return compress_sm90(A, transposed=transposed, **kwargs)
    elif compute_version >= (8, 0):
        if transposed:
            A = A.t().contiguous()
        origin_dtype = A.dtype
        if is_float8_dtype(origin_dtype):
            fp8_remove_negative_zeros_(A)
            A = A.view(torch.int8)
        A_sp, E = compress_sm80(A, transposed=False)
        if is_float8_dtype(origin_dtype):
            A_sp = A_sp.view(origin_dtype)
        if transposed:
            A_sp = A_sp.t().contiguous()
        return A_sp, E
    else:
        raise ValueError(f"Unsupported CUDA compute version: {compute_version}. Supported versions are sm_80 and sm_90.")


def randn_semi_sparse(M: int, K: int, dtype=torch.float16, device="cuda", transposed: bool = False):
    """
    Generate a random semi-sparse tensor. The generated tensor will have 2:4 sparsity along the K dimension.
    Args:
        M (int): Number of rows
        K (int): Number of columns
        dtype: Data type of the tensor
        device: Device to create the tensor on
        transposed (bool): If True, returns a transposed tensor of shape (K, M)
    """
    elem, group = 2, 4
    if dtype == torch.float32:
        elem, group = 1, 2
    tensor = torch.randn((M, K), dtype=torch.float, device=device).view(M, -1, group)
    indice = tensor.topk(elem, dim=-1).indices
    tensor.scatter_(-1, indice, 0)
    tensor = tensor.view(M, K)
    if transposed:
        tensor = tensor.t().contiguous()
    return tensor.to(dtype)  # dtype like float8 might not have randn kernel


def randint_semi_sparse(M: int, K: int, low: int, high: int, dtype=torch.int32, device="cuda", transposed: bool = False):
    """
    Generate a random semi-sparse integer tensor. The generated tensor will have 2:4 sparsity along the K dimension.
    Args:
        M (int): Number of rows
        K (int): Number of columns
        low (int): Lower bound of the random integers
        high (int): Upper bound of the random integers
        dtype: Data type of the tensor
        device: Device to create the tensor on
        transposed (bool): If True, returns a transposed tensor of shape (K, M)
    """
    elem, group = 2, 4
    if dtype == torch.float32:
        elem, group = 1, 2
    tensor = torch.randint(low, high, (M, K), dtype=dtype, device=device).view(M, -1, group)
    indice = tensor.topk(elem, dim=-1).indices
    tensor.scatter_(-1, indice, 0)
    tensor = tensor.view(M, K)
    if transposed:
        tensor = tensor.t().contiguous()
    return tensor


def arange_semi_sparse(M: int, K: int, dtype=torch.float16, device="cuda", transposed: bool = False):
    """
    Generate a semi-sparse tensor with values from 0 to M*K-1. The generated tensor will have 2:4 sparsity along the K dimension.
    Args:
        M (int): Number of rows
        K (int): Number of columns
        dtype: Data type of the tensor
        device: Device to create the tensor on
        transposed (bool): If True, returns a transposed tensor of shape (K, M)
    """
    elem, group = 2, 4
    if dtype == torch.float32:
        elem, group = 1, 2
    tensor = torch.arange(M * K, dtype=dtype, device=device).view(M, -1, group)
    indice = tensor.topk(elem, dim=-1).indices
    tensor.scatter_(-1, indice, 0)
    tensor = tensor.view(M, K)
    if transposed:
        tensor = tensor.t().contiguous()
    return tensor
