import tilelang.testing
import example_gemv


@tilelang.testing.pytest.mark.xfail
def test_example_gemv():
    example_gemv.main(do_bench=False)


if __name__ == "__main__":
    test_example_gemv()
