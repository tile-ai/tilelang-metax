"""MCRTC Backend for TileLang.

This module provides runtime compilation support using MACA's MCRTC API.
"""

import logging

__all__ = [
    "MCRTCKernelAdapter",
    "TLMCRTCSourceWrapper",
    "MCRTCLibraryGenerator",
    "is_mcrtc_available",
    "check_mcrtc_available",
]

logger = logging.getLogger(__name__)

is_mcrtc_available = False
MCRTC_UNAVAILABLE_MESSAGE = (
    "MACA MCRTC runtime is not available, MCRTC backend cannot be used. "
    "Please make sure MACA_PATH is configured and libmcruntime.so can be loaded."
)

try:
    from tilelang.contrib.mcrtc import mcrtc  # noqa: F401

    is_mcrtc_available = True
except (ImportError, OSError) as e:
    logger.debug(f"MCRTC runtime import failed: {e}")


def check_mcrtc_available():
    """Check if MCRTC backend is available."""
    if not is_mcrtc_available:
        raise ImportError(MCRTC_UNAVAILABLE_MESSAGE)


if is_mcrtc_available:
    from .adapter import MCRTCKernelAdapter
    from .libgen import MCRTCLibraryGenerator
    from .wrapper import TLMCRTCSourceWrapper
else:

    class MCRTCKernelAdapter:
        """Dummy MCRTCKernelAdapter that raises ImportError."""

        def __init__(self, *args, **kwargs):
            raise ImportError(MCRTC_UNAVAILABLE_MESSAGE)

        @classmethod
        def from_database(cls, *args, **kwargs):
            raise ImportError(MCRTC_UNAVAILABLE_MESSAGE)

    class TLMCRTCSourceWrapper:
        """Dummy TLMCRTCSourceWrapper that raises ImportError."""

        def __init__(self, *args, **kwargs):
            raise ImportError(MCRTC_UNAVAILABLE_MESSAGE)

    class MCRTCLibraryGenerator:
        """Dummy MCRTCLibraryGenerator that raises ImportError."""

        def __init__(self, *args, **kwargs):
            raise ImportError(MCRTC_UNAVAILABLE_MESSAGE)
