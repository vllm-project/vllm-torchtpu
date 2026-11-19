# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

# (TPU generation, score width, candidate count, records per candidate block).
PUBLISH_CANDIDATE_SHAPES = frozenset({(7, 16384, 2048, 8)})
# (generation, queries, token-score width, block size): (rows, output blocks).
PUBLISH_BLOCK_MAX_CONFIG = {(7, 96, 98816, 8): (96, 1024)} | {
    (7, rows, 1048576, 8): (16, 4096) for rows in (16, 32, 48, 64, 128)
}


@dataclass(frozen=True)
class StreamingTopKConfig:
    loop_unroll: int = 1
    bootstrap_multiplier: int = 4
    bitonic_ids: bool = False


# (generation, score width, topk): measured scan, bootstrap and output sorting.
STREAMING_TOPK_CONFIG = {
    (7, width, 512): StreamingTopKConfig(4, 8, True)
    for width in (16384, 32768, 49408, 98816, 262144, 524288, 1048576)
}
