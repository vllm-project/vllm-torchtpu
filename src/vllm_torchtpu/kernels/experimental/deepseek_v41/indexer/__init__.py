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
    GATHER_BLOCK_ROWS,
    MAX_SCORING_TOKENS,
    MAX_SPARSECORE_TOPK,
    MIN_SPARSECORE_KEYS,
    paged_indexer_topk,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer.word_scores import (
    DIM,
    MAX_BLOCK_ROWS,
    MIN_ROWS,
    ROW_WORDS,
    SLOTS,
    SUB_BLOCK_ROWS,
    score_word_rows,
)

__all__ = [
    "DIM",
    "GATHER_BLOCK_ROWS",
    "MAX_BLOCK_ROWS",
    "MAX_SCORING_TOKENS",
    "MAX_SPARSECORE_TOPK",
    "MIN_ROWS",
    "MIN_SPARSECORE_KEYS",
    "ROW_WORDS",
    "SLOTS",
    "SUB_BLOCK_ROWS",
    "paged_indexer_topk",
    "score_word_rows",
]
