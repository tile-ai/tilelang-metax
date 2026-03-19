import tilelang.testing

from example_tilelang_gemm_streamk import main


# not fully supported on sm90
@tilelang.testing.pytest.mark.xfail
def test_example_tilelang_gemm_streamk():
    main()


if __name__ == "__main__":
    tilelang.testing.main()
