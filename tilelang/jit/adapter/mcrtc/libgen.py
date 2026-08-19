from __future__ import annotations

import importlib.util
import logging
import tempfile
from types import ModuleType
from typing import Any

from tvm.target import Target

from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.contrib.mcrtc import compile_maca
from tilelang.jit.adapter.mcrtc import check_mcrtc_available
from tilelang.jit.adapter.mcrtc.driver import load_module, unload_module

logger = logging.getLogger(__name__)


class MCRTCLibraryGenerator(LibraryGenerator):
    """Runtime compiler and loader for MCRTC-compiled MACA kernels."""

    host_func: str | None = None
    module: Any = None
    pymodule: ModuleType | None = None
    pypath: str | None = None

    def __init__(self, target: Target, verbose: bool = False):
        check_mcrtc_available()
        super().__init__(target, verbose)

    @staticmethod
    def import_from_file(module_name: str, file_path: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create module spec from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def update_host_func(self, host_func: str):
        self.host_func = host_func

    def load_lib(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path

        self.pypath = lib_path.replace(".mcbin", ".py")
        self.pymodule = self.import_from_file("kernel", self.pypath)

        with open(lib_path, "rb") as f:
            bitcode = f.read()

        self.module = load_module(bitcode)

    def compile_lib(self, timeout: float | None = None):
        if self.host_func is None:
            raise RuntimeError("Host function is not set, please call update_host_func() first.")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".maca", delete=False) as src:
            src.write(self.lib_code)
            self.srcpath = src.name

        self.libpath = self.srcpath.replace(".maca", ".mcbin")
        self.pypath = self.srcpath.replace(".maca", ".py")
        options = [item for flag in self.compile_flags for item in flag.split()] if self.compile_flags else []
        bitcode = compile_maca(self.lib_code, options=options, verbose=True)

        with open(self.libpath, "wb") as f:
            f.write(bitcode)

        with open(self.pypath, "w") as f:
            f.write(self.host_func)

    def __del__(self):
        if self.module is not None:
            try:
                unload_module(self.module)
            except Exception as e:
                logger.debug(f"Failed to unload MACA module: {e}")
            self.module = None
