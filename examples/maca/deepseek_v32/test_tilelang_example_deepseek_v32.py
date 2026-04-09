# ruff: noqa
import tilelang
import tilelang.testing

import topk_selector
import fp8_lighting_indexer
import sparse_mla_fwd
import sparse_mla_fwd_pipelined
import sparse_mla_bwd


def test_example_topk_selector_maca_histogram_reset_plan():
    cfg = topk_selector._maca_histogram_reset_config(radix=256, block_size=256)

    cleared = set(range(cfg["thread_clear_limit"]))
    cleared.add(cfg["sentinel_bucket"])

    assert cleared == set(range(257))
    assert cfg["sentinel_thread"] == 0


def test_example_sparse_mla_fwd_maca_head_partition():
    cfg = sparse_mla_fwd._head_partition_config(head_kv=64, padded_H=64, is_maca=True)

    assert cfg["max_block_h"] == 16
    assert cfg["replicate_h"] == 4
    assert cfg["h_per_block"] == 16
    assert cfg["head_block_stride"] == 16


def test_example_topk_selector():
    topk_selector.test_topk_selector()


def test_example_fp8_lighting_indexer():
    fp8_lighting_indexer.test_fp8_lighting_indexer(S=512, SKV=1024, H=32, HKV=1, D=64, kv_stride=1)


def test_example_sparse_mla_fwd():
    # small shapes for testing
    sparse_mla_fwd.test_sparse_mla_fwd(S=256, SKV=1024, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


def test_example_sparse_mla_fwd_pipelined():
    # small shapes for testing
    sparse_mla_fwd_pipelined.test_sparse_mla_fwd_pipelined(S=256, SKV=512, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


def test_example_sparse_mla_bwd():
    sparse_mla_bwd.test_sparse_mla_bwd(S=256, SKV=512, H=64, HKV=1, DQKV=576, DV=512, topk=256, check_correctness=False)
    sparse_mla_bwd.test_sparse_mla_bwd(
        S=256, SKV=512, H=128, HKV=1, DQKV=576, DV=512, topk=256, check_correctness=False
    )  # test for large H


if __name__ == "__main__":
    tilelang.testing.main()
