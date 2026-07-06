import os
import pytest
import tilelang.testing
import example_mla_decode

_is_cutedsl = os.environ.get("TILELANG_TARGET", "").lower() == "cutedsl"


@tilelang.testing.requires_cuda
@pytest.mark.skipif(_is_cutedsl, reason="CuTeDSL backend does not support alloc_global yet")
def test_example_mla_decode():
    example_mla_decode.main()


if __name__ == "__main__":
    tilelang.testing.main()
