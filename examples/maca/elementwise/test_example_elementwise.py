import tilelang.testing
import example_elementwise_add


@tilelang.testing.pytest.mark.xfail
def test_example_elementwise_add():
    example_elementwise_add.main()


if __name__ == "__main__":
    tilelang.testing.main()
