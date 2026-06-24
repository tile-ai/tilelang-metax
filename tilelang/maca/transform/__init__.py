"""MACA-specific transformation frontends."""

from .. import _ffi_api


def LowerMACAIntrin():
    """LowerMACAIntrin"""
    return _ffi_api.LowerMACAIntrin()  # type: ignore


__all__ = ["LowerMACAIntrin"]
