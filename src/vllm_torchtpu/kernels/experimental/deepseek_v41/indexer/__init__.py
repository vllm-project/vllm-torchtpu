# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V4.1 experimental TPU Pallas indexer kernels."""

from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer.paged_indexer import (
    MAX_SCORING_TOKENS,
    MIN_SPARSECORE_KEYS,
    paged_indexer_topk,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer.sparsecore_topk import (
    BITS_NEG_INF,
    MAX_N,
    MAX_TOPK,
    SC_LANES,
    sparsecore_topk,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer.word_scores import (
    BF16_BYTES,
    DIM,
    MAX_BLOCK_ROWS,
    MIN_ROWS,
    ROW_WORDS,
    SLOTS,
    SUB_BLOCK_ROWS,
    WEIGHT_BYTES_PER_HEAD,
    WORD_BYTES,
    score_word_rows,
)

__all__ = [
    "BF16_BYTES",
    "BITS_NEG_INF",
    "DIM",
    "MAX_BLOCK_ROWS",
    "MAX_N",
    "MAX_SCORING_TOKENS",
    "MAX_TOPK",
    "MIN_ROWS",
    "MIN_SPARSECORE_KEYS",
    "ROW_WORDS",
    "SC_LANES",
    "SLOTS",
    "SUB_BLOCK_ROWS",
    "WEIGHT_BYTES_PER_HEAD",
    "WORD_BYTES",
    "paged_indexer_topk",
    "score_word_rows",
    "sparsecore_topk",
]
