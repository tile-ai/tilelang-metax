import tilelang.testing

import block_sparse_attn_tilelang


@tilelang.testing.pytest.mark.skip("timeout")
def test_block_sparse_attn_tilelang():
    block_sparse_attn_tilelang.main()


if __name__ == "__main__":
    tilelang.testing.main()
