"""Shared helpers for preparing TVM-FFI callable ABIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm.tirx import PrimFunc


_TORCH_FFI_CALLEE_ALLOCATED_OUTPUT_MIN_VERSION = (2, 8)


def get_torch_version() -> tuple[int, int]:
    """Return the running PyTorch major/minor version."""
    import torch

    version_parts = str(torch.__version__).split("+", 1)[0].split(".")
    try:
        return int(version_parts[0]), int(version_parts[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"Cannot parse PyTorch version {torch.__version__!r} for TVM-FFI ABI selection") from error


def torch_supports_tvm_ffi_callee_allocated_output_abi() -> bool:
    """Return whether PyTorch supports the callee-allocated TVM-FFI ABI."""
    return get_torch_version() > _TORCH_FFI_CALLEE_ALLOCATED_OUTPUT_MIN_VERSION


def should_use_tvm_ffi_callee_allocated_output_abi(target) -> bool:
    """Return whether ``target`` uses the callee-allocated TVM-FFI ABI.

    MACA Torch builds through 2.8 use the legacy Python-preallocated output
    ABI. CUDA and other targets retain their existing behavior.
    """
    target_kind = getattr(getattr(target, "kind", None), "name", None)
    if target_kind != "maca":
        return True

    return torch_supports_tvm_ffi_callee_allocated_output_abi()


def _normalize_output_indices(output_indices: list[int], num_params: int) -> list[int]:
    normalized = []
    for raw_index in output_indices:
        index = int(raw_index)
        if index < 0:
            index += num_params
        if index < 0 or index >= num_params:
            raise ValueError(f"out_idx index {raw_index} is out of range for a function with {num_params} parameters")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"out_idx contains duplicate tensor indices: {output_indices}")
    return normalized


def prepare_tvm_ffi_callee_allocated_outputs(
    func: PrimFunc,
    out_idx: list[int] | int | None,
    *,
    enable: bool = True,
) -> tuple[PrimFunc, list[int] | None]:
    """Resolve output indices and optionally expose them to TVM-FFI lowering.

    When disabled, output indices are still returned to the Python adapter, but
    the lowering-only attribute is removed so ``MakePackedAPI`` emits the
    legacy preallocated-output argument layout.
    """
    requested_indices = None if out_idx is None else ([out_idx] if isinstance(out_idx, int) else list(out_idx))
    attr_indices = None
    if func.attrs is not None and "tilelang_out_idx" in func.attrs:
        attr_indices = [int(index) for index in func.attrs["tilelang_out_idx"]]

    if attr_indices is not None:
        if requested_indices is not None:
            num_params = len(func.params)
            if _normalize_output_indices(requested_indices, num_params) != _normalize_output_indices(attr_indices, num_params):
                raise ValueError("out_idx does not match the PrimFunc's tilelang_out_idx attribute")
        output_indices = attr_indices
    else:
        output_indices = requested_indices or []
    if not output_indices:
        return func, None
    _normalize_output_indices(output_indices, len(func.params))
    if not enable:
        if attr_indices is not None:
            return func.without_attr("tilelang_out_idx"), output_indices
        return func, output_indices
    if attr_indices is not None:
        return func, output_indices
    return func.with_attr("tilelang_out_idx", output_indices), output_indices
