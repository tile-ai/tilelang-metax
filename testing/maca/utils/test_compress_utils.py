import torch

from tilelang.utils.sparse import compress_maca, randn_semi_sparse


def _test_compress(M, K, block_k, dtype):
    A = randn_semi_sparse(M, K, dtype=dtype, device="cuda")
    A_sparse, E = compress_maca(A, block_k, False)


def test_compress():
    _test_compress(1024, 1024, 128, torch.float16)
    _test_compress(1024, 1024, 64, torch.float16)
    _test_compress(1024, 1024, 32, torch.float16)

    _test_compress(1024, 1024, 128, torch.bfloat16)
    _test_compress(1024, 1024, 64, torch.bfloat16)
    _test_compress(1024, 1024, 32, torch.bfloat16)

    _test_compress(1024, 1024, 64, torch.float32)
    _test_compress(1024, 1024, 32, torch.float32)
    _test_compress(1024, 1024, 16, torch.float32)

    _test_compress(1024, 1024, 256, torch.float8_e4m3fn)
    _test_compress(1024, 1024, 128, torch.float8_e4m3fn)
    _test_compress(1024, 1024, 64, torch.float8_e4m3fn)

    _test_compress(1024, 1024, 256, torch.float8_e5m2)
    _test_compress(1024, 1024, 128, torch.float8_e5m2)
    _test_compress(1024, 1024, 64, torch.float8_e5m2)


if __name__ == "__main__":
    test_compress()
    print("All tests passed.")
