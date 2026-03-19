# ruff: noqa
import tilelang.testing

from example_tilelang_nsa_fwd import main as main_fwd
from example_tilelang_nsa_decode import main as main_fwd_decode


@tilelang.testing.pytest.mark.skip("coredump")
def test_example_tilelang_nsa_fwd():
    main_fwd()


@tilelang.testing.pytest.mark.skip("coredump")
def test_example_tilelang_nsa_fwd_decode():
    main_fwd_decode()


if __name__ == "__main__":
    tilelang.testing.main()
