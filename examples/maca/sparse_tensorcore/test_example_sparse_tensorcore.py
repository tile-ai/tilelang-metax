import tilelang.testing
import tilelang
import tilelang_example_sparse_tensorcore


@tilelang.testing.pytest.mark.xfail
def test_tilelang_example_sparse_tensorcore():
    tilelang_example_sparse_tensorcore.main()


if __name__ == "__main__":
    tilelang.testing.main()
