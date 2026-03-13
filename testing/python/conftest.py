# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

import os
import pytest


def _parameterize_target(metafunc):
    # ENV variable TILELANG_TEST_TARGETS specify target names splited by ";"
    # default value is maca
    if "target" in metafunc.fixturenames:
        parametrized_args = [arg.strip() for mark in metafunc.definition.iter_markers("parametrize") for arg in mark.args[0].split(",")]
        if "target" not in parametrized_args:
            mark = pytest.mark.parametrize(
                "target",
                os.environ.get("TILELANG_TEST_TARGET", "maca").split(";"),
                scope="session",
            )
            metafunc.definition.add_marker(mark)


def pytest_generate_tests(metafunc):
    _parameterize_target(metafunc)
