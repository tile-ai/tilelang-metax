import tilelang.testing
import torch

import example_vertical_slash_sparse_attn


def test_vs_sparse_attn_empty_vertical_and_slash_indexes():
    seqlens = torch.tensor([64], dtype=torch.int32, device="cuda")
    vertical = torch.empty((1, 1, 0), dtype=torch.int32, device="cuda")
    slash = torch.empty((1, 1, 0), dtype=torch.int32, device="cuda")

    block_count, block_offset, column_count, column_index = example_vertical_slash_sparse_attn._convert_vertical_slash_indexes_fallback(
        seqlens=seqlens,
        vertical_indexes=vertical,
        slash_indexes=slash,
        context_size=64,
        block_size_M=64,
        block_size_N=64,
    )

    assert torch.equal(block_count.cpu(), torch.zeros((1, 1, 1), dtype=torch.int32))
    assert torch.equal(column_count.cpu(), torch.zeros((1, 1, 1), dtype=torch.int32))
    assert block_offset.shape == (1, 1, 1, 0)
    assert column_index.shape == (1, 1, 1, 0)


def test_vs_sparse_attn_empty_vertical_indexes():
    seqlens = torch.tensor([64], dtype=torch.int32, device="cuda")
    vertical = torch.empty((1, 1, 0), dtype=torch.int32, device="cuda")
    slash = torch.tensor([[[0]]], dtype=torch.int32, device="cuda")

    block_count, block_offset, column_count, column_index = example_vertical_slash_sparse_attn._convert_vertical_slash_indexes_fallback(
        seqlens=seqlens,
        vertical_indexes=vertical,
        slash_indexes=slash,
        context_size=64,
        block_size_M=64,
        block_size_N=64,
    )

    assert torch.equal(column_count.cpu(), torch.zeros((1, 1, 1), dtype=torch.int32))
    assert column_index.shape == (1, 1, 1, 0)
    assert block_count[0, 0, 0].item() >= 0
    assert block_offset.shape == (1, 1, 1, 1)


def test_vs_sparse_attn_empty_slash_indexes():
    seqlens = torch.tensor([64], dtype=torch.int32, device="cuda")
    vertical = torch.tensor([[[0, 32]]], dtype=torch.int32, device="cuda")
    slash = torch.empty((1, 1, 0), dtype=torch.int32, device="cuda")

    block_count, block_offset, column_count, column_index = example_vertical_slash_sparse_attn._convert_vertical_slash_indexes_fallback(
        seqlens=seqlens,
        vertical_indexes=vertical,
        slash_indexes=slash,
        context_size=64,
        block_size_M=64,
        block_size_N=64,
    )

    assert torch.equal(block_count.cpu(), torch.zeros((1, 1, 1), dtype=torch.int32))
    assert block_offset.shape == (1, 1, 1, 0)
    assert torch.equal(column_count.cpu(), torch.tensor([[[2]]], dtype=torch.int32))
    assert torch.equal(column_index.cpu(), torch.tensor([[[[0, 32]]]], dtype=torch.int32))


def test_vs_sparse_attn():
    example_vertical_slash_sparse_attn.main()


if __name__ == "__main__":
    tilelang.testing.main()
