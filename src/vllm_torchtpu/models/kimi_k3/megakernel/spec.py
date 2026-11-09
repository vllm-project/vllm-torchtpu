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
"""TpuMonoSpec declaration for the Kimi K3 megakernel."""

from __future__ import annotations

from vllm_torchtpu.megakernel.runtime.contract import (
    DOMAIN_TP,
    FEATURE_KV_OFFLOAD,
    FEATURE_KV_TRANSFER,
    FEATURE_PREFIX_CACHING,
    FEATURE_SPEC_DECODE,
    PAR_DCP,
    PAR_DP,
    PAR_EP_IN_TP,
    PAR_TP,
    TpuMonoSpec,
)

from .layout import CONV_HEAD_ROWS, LAYOUT, NUM_LAYERS

KDA_LAYERS: int = len(LAYOUT.state_layers)
MLA_LAYERS: int = len(LAYOUT.attention_layers)
RECURRENT_HEAD_ROWS: int = LAYOUT.region("recurrent").rows
HIDDEN_SIZE: int = 7168
VERIFY_TOKENS: tuple[int, ...] = (8,)

SPEC = TpuMonoSpec(
    name="kimi_k3",
    version="1",
    create=".adapter:create",
    chips=("TPU7x",),
    tp_sizes=(),
    parallelism=frozenset(
        {
            PAR_TP,
            PAR_DP,
            PAR_DCP,
            PAR_EP_IN_TP,
        }
    ),
    features=frozenset(
        {
            FEATURE_SPEC_DECODE,
            FEATURE_PREFIX_CACHING,
            FEATURE_KV_OFFLOAD,
            FEATURE_KV_TRANSFER,
        }
    ),
    requires={},
    numerics="approx",
    collective_domain=DOMAIN_TP,
    layout_request=".adapter:layout_request",
    load_tensor=".adapter:load_tensor",
    decline_codes=(
        "runner_lifecycle_unwired",
        "unsupported_arch",
        "unsupported_tp_size",
        "unsupported_dcp_size",
        "unsupported_attention_dp",
        "invalid_geometry",
        "backbone_not_found",
        "unaligned_cache_layout",
        "missing_mxfp4_weights",
        "empty_buckets",
        "verify_commit_hook_unwired",
    ),
)

__all__ = [
    "CONV_HEAD_ROWS",
    "HIDDEN_SIZE",
    "KDA_LAYERS",
    "MLA_LAYERS",
    "NUM_LAYERS",
    "RECURRENT_HEAD_ROWS",
    "SPEC",
    "VERIFY_TOKENS",
]
