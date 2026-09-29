# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integer cost model for GLM-5.2 masked-dense MLA dispatch."""

from typing import NamedTuple


class MaskedDensePrefillCostModel(NamedTuple):
    analytic_fixed_ns: int
    analytic_token_ns: int
    bitmap_fixed_ns: int
    bitmap_token_ns: int
    q_kv_block_ns: int
    sparse_fixed_ns: int
    sparse_token_ns: int


# Geometry of the tuned GLM-5.2 kernel.
KV_BLOCK_SIZE = 1024
QUERY_BLOCK_SIZE = 32

# Conservative integer fits to the TPU7x medians produced by
# scripts/benchmark_masked_dense_vs_sparse_mla.py. The masked-dense estimate
# grows with the actual query-block x KV-block grid; sparse gather grows with
# query tokens and is otherwise nearly flat in KV length.
GLM52_TPU7X_PREFILL_COST_MODEL = MaskedDensePrefillCostModel(
    analytic_fixed_ns=215_000,
    analytic_token_ns=245,
    bitmap_fixed_ns=227_000,
    bitmap_token_ns=451,
    q_kv_block_ns=9_040,
    sparse_fixed_ns=259_000,
    sparse_token_ns=2_156,
)
