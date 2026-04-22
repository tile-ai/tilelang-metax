# ruff: noqa
import tilelang.testing

from example_tilelang_nsa_fwd import main as main_fwd
from example_tilelang_nsa_decode import main as main_fwd_decode
from example_tilelang_nsa_bwd import main as main_bwd


def test_example_tilelang_nsa_fwd():
    main_fwd()


def test_example_tilelang_nsa_fwd_decode():
    main_fwd_decode()


def test_example_tilelang_nsa_bwd():
    main_bwd()


if __name__ == "__main__":
    tilelang.testing.main()
