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
"""Architectures whose KV cache is several arrays of different shapes.

The offload connector registers such a cache array by array instead of
tiling one common row (see `raiden_store.resolve_kernel_geometry`). The
scheduler only sees a flattened KV cache config, where the per-layer shapes
are gone, so the scheduler and the workers decide from the architecture.

A model package declares its architectures here from its plugin `register`,
which runs in every process. Kept import-free so that call stays cheap.
"""

_MULTI_SHAPES_KV_ARCHITECTURES: set[str] = set()


def register_multi_shapes_kv_architectures(*architectures: str) -> None:
    """Declare that these architectures keep KV arrays of different shapes."""
    _MULTI_SHAPES_KV_ARCHITECTURES.update(architectures)


def is_multi_shapes_kv_architecture(architecture: str | None) -> bool:
    return architecture in _MULTI_SHAPES_KV_ARCHITECTURES
