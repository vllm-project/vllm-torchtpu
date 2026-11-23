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
"""Single-bank word-row cache and paged SWA window Pallas kernels for TPU7x."""

from vllm_torchtpu.kernels.experimental.deepseek_v41.cache.kernel import (
    DEFAULT_PAGE_TOKENS,
    GATHER_PAGE_BURST_ROWS,
    GATHER_SUB_BURST_ROWS,
    INDEXER_HEAD_DIM,
    INDEXER_KEY_WORDS,
    LATENT_DIM,
    NOPE_DIM,
    ROPE_DIM,
    ROPE_WORDS,
    ROW_BYTES,
    ROW_WORDS,
    VALID_SLICE_WORDS,
    WORD_BYTES,
    gather_cache_rows,
    pack_fp8_kv_rows,
    write_sub_rows,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.cache.swa_window import (
    gather_swa_windows,
)

__all__ = [
    "DEFAULT_PAGE_TOKENS",
    "GATHER_PAGE_BURST_ROWS",
    "GATHER_SUB_BURST_ROWS",
    "INDEXER_HEAD_DIM",
    "INDEXER_KEY_WORDS",
    "LATENT_DIM",
    "NOPE_DIM",
    "ROPE_DIM",
    "ROPE_WORDS",
    "ROW_BYTES",
    "ROW_WORDS",
    "VALID_SLICE_WORDS",
    "WORD_BYTES",
    "gather_cache_rows",
    "gather_swa_windows",
    "pack_fp8_kv_rows",
    "write_sub_rows",
]
