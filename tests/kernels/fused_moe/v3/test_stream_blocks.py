# SPDX-License-Identifier: Apache-2.0
"""The column blocks a streamed expert may take, on the host."""

from vllm_torchtpu.kernels.fused_moe.v3.layer import stream_blocks_for


def test_blocks_are_divisors_in_whole_activation_blocks():
    assert stream_blocks_for(3072, 512) == [1536, 1024, 512]
    assert stream_blocks_for(4096, 512) == [2048, 1024, 512]


def test_block_scaled_blocks_cover_whole_scale_tiles():
    # 3072 in weight blocks of 512 has 6 scale rows: no block below the
    # whole width covers a whole tile of 8, so nothing can stream.
    assert stream_blocks_for(3072, 512, 512) == []
    # 4096 in blocks of 128 has 32: blocks of 8 rows are 1024 columns.
    assert stream_blocks_for(4096, 512, 128) == [2048, 1024]
