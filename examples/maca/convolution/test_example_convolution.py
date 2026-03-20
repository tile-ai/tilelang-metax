import tilelang.testing

import example_convolution
import example_convolution_autotune


# TODO(@cy): TMA with convolution must be fixed in future.
@tilelang.testing.pytest.mark.xfail
def test_example_convolution():
    example_convolution.main([])


def test_example_convolution_autotune():
    example_convolution_autotune.main()


if __name__ == "__main__":
    tilelang.testing.main()
