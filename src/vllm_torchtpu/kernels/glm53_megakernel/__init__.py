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
"""GLM-5.3-Flash fused 46-layer decode + EAGLE3 MTP megakernel on TPU v7x.

Importing this package is stdlib-only so the megakernel plugin registry can
inspect `SPEC` without importing JAX or PyTorch. Kernel, loader, and config
symbols are loaded lazily on attribute access.
"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.megakernel.core.runtime.contract import (
    FEATURE_AUX_HIDDEN_STATES,
    FEATURE_PREFIX_CACHING,
    FEATURE_SPEC_DECODE,
    PAR_TP,
    TpuMonoSpec,
)

SPEC = TpuMonoSpec(
    name="glm53",
    version="1.0",
    create="vllm_torchtpu.models.vllm.glm5_next.megakernel:create",
    chips=("TPU7x",),
    tp_sizes=(8,),
    parallelism=frozenset({PAR_TP}),
    features=frozenset(
        {
            FEATURE_SPEC_DECODE,
            FEATURE_PREFIX_CACHING,
            FEATURE_AUX_HIDDEN_STATES,
        }
    ),
    numerics="approx",
    top_level_options=frozenset(
        {
            "max_req",
            "cache_kpages",
            "profile_scopes",
            "debug",
            "timing",
        }
    ),
)

_KERNEL_ATTRS = frozenset(
    {
        "EXPERT_NAMES",
        "KERNEL_MTP_WEIGHT_NAMES",
        "KERNEL_WEIGHT_NAMES",
        "KV_GROUP",
        "MTP_EXPERT_NAMES",
        "TOPK_OUT",
        "expert_ep_to_tp",
        "experts_tp_to_kmajor",
        "kernel_cache_token_loc",
        "kv_bytes",
        "kv_words",
        "lane_topk",
        "latent_pool_shapes",
        "layer_tables",
        "make_decode_step",
        "make_spec_decode_step",
        "prepare_kernel_weights_local",
        "superpages_per_slot",
    }
)

_LOAD_ATTRS = frozenset(
    {
        "Glm53Checkpoint",
        "expected_rank_specs",
        "place_rank_weights_on_mesh",
        "requant_fp8_per_channel",
        "synthetic_rank_weights",
        "to_per_channel_fp8",
    }
)

__all__ = [
    "SPEC",
    "Glm53MegakernelConfig",
    *_KERNEL_ATTRS,
    *_LOAD_ATTRS,
]


def __getattr__(name: str) -> Any:
    if name == "Glm53MegakernelConfig":
        from .config import Glm53MegakernelConfig

        return Glm53MegakernelConfig
    if name in _KERNEL_ATTRS:
        from . import kernel

        return getattr(kernel, name)
    if name in _LOAD_ATTRS:
        from . import load

        return getattr(load, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
