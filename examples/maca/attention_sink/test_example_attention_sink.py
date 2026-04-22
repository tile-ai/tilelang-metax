import tilelang.testing

import example_mha_sink_fwd_bhsd
import example_mha_sink_bwd_bhsd
import example_gqa_sink_bwd_bhsd
import example_gqa_sink_fwd_varlen
import example_gqa_sink_bwd_varlen


def test_example_mha_sink_fwd_bhsd_full_attn():
    example_mha_sink_fwd_bhsd.main()


def test_example_mha_sink_fwd_bhsd_sliding_window():
    example_mha_sink_fwd_bhsd.main(window_size=128)


def test_example_mha_sink_bwd_bhsd():
    example_mha_sink_bwd_bhsd.main()


def test_example_mha_sink_bwd_bhsd_sliding_window():
    example_mha_sink_bwd_bhsd.main(window_size=128)


def test_example_gqa_sink_bwd_bhsd():
    example_gqa_sink_bwd_bhsd.main()


def test_example_gqa_sink_bwd_bhsd_sliding_window():
    example_gqa_sink_bwd_bhsd.main(window_size=128)


def test_example_gqa_sink_varlen():
    example_gqa_sink_fwd_varlen.main()  # non-causal
    example_gqa_sink_bwd_varlen.main()  # causal


if __name__ == "__main__":
    tilelang.testing.main()
