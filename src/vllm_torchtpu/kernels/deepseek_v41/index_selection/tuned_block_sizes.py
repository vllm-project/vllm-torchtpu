# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

# (TPU generation, score width, candidate count, records per candidate block).
PUBLISH_CANDIDATE_SHAPES = frozenset({(7, 16384, 2048, 8)})


@dataclass(frozen=True)
class StreamingTopKConfig:
    loop_unroll: int = 1
    bootstrap_multiplier: int = 4
    bitonic_ids: bool = False


# (generation, score width, topk): measured scan, bootstrap and output sorting.
STREAMING_TOPK_CONFIG = {
    (7, width, 512): StreamingTopKConfig(4, 8, True)
    for width in (16384, 32768, 49408, 98816)
}
