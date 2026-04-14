import tilelang.testing

from example_deepgemm_fp8_2xAcc import main


def test_deepgemm_fp8_2xAcc():
    main()


if __name__ == "__main__":
    tilelang.testing.main()
