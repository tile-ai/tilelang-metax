# ruff: noqa
import tilelang
import tilelang.testing

import topk_selector
import fp8_lighting_indexer
import sparse_mla_fwd
import sparse_mla_fwd_pipelined
import sparse_mla_bwd


@tilelang.testing.pytest.mark.xfail
def test_example_topk_selector():
    topk_selector.test_topk_selector()


@tilelang.testing.pytest.mark.xfail
def test_example_fp8_lighting_indexer():
    fp8_lighting_indexer.test_fp8_lighting_indexer(S=512, SKV=1024, H=32, HKV=1, D=64, kv_stride=1)


def test_example_sparse_mla_fwd():
    sparse_mla_fwd.test_sparse_mla_fwd(
        S=64, SKV=128, H=16, HKV=1, DQK=576, DV=512, topk=32, check_correctness=False, block_I=32, threads=128, num_stages=1
    )


@tilelang.testing.pytest.mark.xfail
def test_example_sparse_mla_fwd_pipelined():
    # small shapes for testing
    sparse_mla_fwd_pipelined.test_sparse_mla_fwd_pipelined(S=256, SKV=512, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


def test_example_sparse_mla_bwd():
    sparse_mla_bwd.test_sparse_mla_bwd(
        S=64,
        SKV=128,
        H=16,
        HKV=1,
        DQKV=288,
        DV=256,
        topk=32,
        check_correctness=False,
        fwd_block_I=32,
        fwd_num_stages=1,
        fwd_threads=128,
        bwd_block_size=32,
        bwd_split_store=2,
        bwd_num_stages=0,
        bwd_threads=128,
    )


if __name__ == "__main__":
    tilelang.testing.main()
