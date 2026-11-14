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
"""Model-owned unified KV pools for Raiden offload.

Some models own a unified KV pool whose groups share one page size but not one
attention geometry (for example main KV, compressed keys, raw rings and
recurrent state in one pool). Each (group, block) key still names exactly one
page, so the model describes the pool's carrier rows itself and every group's
page moves as opaque bytes. A model directory registers its architectures here
from its register() hook; Raiden asks before its generic geometry resolution.
Nothing is registered by default.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelOwnedPool:
    # (vllm_config, kv_cache_config) -> (kernel_block_size, per_block_shape,
    # dtype, device_block_size), the tuple raiden_store.resolve_kernel_geometry
    # returns.
    geometry: Callable[[Any, Any], tuple]
    # kv_cache_config -> number of distinct pool storages a worker registers,
    # checked when the worker registers its KV caches; None skips the check.
    regions: Callable[[Any], int] | None = None


_POOLS: dict[str, ModelOwnedPool] = {}


def register_model_owned_pool(
    architectures: Iterable[str],
    geometry: Callable[[Any, Any], tuple],
    regions: Callable[[Any], int] | None = None,
) -> None:
    """Describe the KV pool of `architectures` to Raiden offload."""
    pool = ModelOwnedPool(geometry, regions)
    for architecture in architectures:
        _POOLS[architecture] = pool


def model_owned_pool(vllm_config: Any) -> ModelOwnedPool | None:
    """The registered pool of this config's architecture, if any."""
    model_config = getattr(vllm_config, "model_config", None)
    return _POOLS.get(getattr(model_config, "architecture", None))
