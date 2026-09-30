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
"""TPU-Friendly StreamIndex Top-K kernel."""

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    bref_override,
    config,
    kv_cache_append,
    kv_cache_utils,
    metadata,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import (
    DEFAULT_BUFFER_COUNT,
    KVLayout,
    MlaCase,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.kv_cache_append import (
    coalesced_kv_append,
    coalesced_kv_append_dcp,
    scatter_rows,
    scatter_rows_reference,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.kv_cache_utils import (
    cache_page_size,
    get_dcp_page_and_slot,
    get_dst_rows,
    get_page_and_slot,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
    DCP_AXIS_NAME,
    DEFAULT_VMEM_LIMIT_BYTES,
    convert_cache_to_seq_along_lane,
    cp_global_to_local,
    cp_local_length,
    cp_local_to_global,
    cp_owner_rank,
    cp_rank_as_data,
    get_kv_cache_shape,
    streamindex_topk,
    streamindex_topk_dcp,
)

__all__ = [
    "bref_override",
    "config",
    "kv_cache_append",
    "kv_cache_utils",
    "metadata",
    "MlaCase",
    "KVLayout",
    "DEFAULT_BUFFER_COUNT",
    "DEFAULT_VMEM_LIMIT_BYTES",
    "DCP_AXIS_NAME",
    "cache_page_size",
    "coalesced_kv_append",
    "coalesced_kv_append_dcp",
    "convert_cache_to_seq_along_lane",
    "cp_local_to_global",
    "cp_local_length",
    "cp_owner_rank",
    "cp_global_to_local",
    "cp_rank_as_data",
    "get_dcp_page_and_slot",
    "get_dst_rows",
    "get_kv_cache_shape",
    "get_page_and_slot",
    "scatter_rows",
    "scatter_rows_reference",
    "streamindex_topk",
    "streamindex_topk_dcp",
]
