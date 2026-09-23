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
    metadata,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import (
    DEFAULT_BUFFER_COUNT,
    KVLayout,
    MlaCase,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
    DCP_AXIS_NAME,
    DEFAULT_VMEM_LIMIT_BYTES,
    _select_owned_winners,
    cp_global_to_local,
    cp_local_length,
    cp_local_to_global,
    cp_owner_rank,
    cp_rank_as_data,
    streamindex_topk,
    streamindex_topk_dcp,
)

__all__ = [
    "bref_override",
    "config",
    "metadata",
    "MlaCase",
    "KVLayout",
    "DEFAULT_BUFFER_COUNT",
    "DEFAULT_VMEM_LIMIT_BYTES",
    "DCP_AXIS_NAME",
    "cp_local_to_global",
    "cp_local_length",
    "cp_owner_rank",
    "cp_global_to_local",
    "cp_rank_as_data",
    "_select_owned_winners",
    "streamindex_topk",
    "streamindex_topk_dcp",
]
