# SPDX-License-Identifier: Apache-2.0
"""Measured D1 schedule for the TPU v7 packed Engram specialization."""

from typing import NamedTuple


class EngramBlockSizes(NamedTuple):
    block_tokens: int
    group_tokens: int
    chunk_words: int
    dequant_words: int
    wkv_bn: int
    reduce_rows: int
    reduce_cols: int
    value_cols: int
    gate_rows: int
    quant_unroll: int
    vmem_limit_bytes: int


# Key: TPU version, tokens, projection K, projection N, streams, hidden dim.
# Values preserve the measured D1 double-buffered DMA and MXU schedule.
TUNED_BLOCK_SIZES = {
    (7, 288, 6144, 25600, 4, 5120): EngramBlockSizes(
        block_tokens=32,
        group_tokens=8,
        chunk_words=128,
        dequant_words=384,
        wkv_bn=2560,
        reduce_rows=16,
        reduce_cols=128,
        value_cols=128,
        gate_rows=32,
        quant_unroll=2,
        vmem_limit_bytes=64 * 1024**2,
    ),
}
