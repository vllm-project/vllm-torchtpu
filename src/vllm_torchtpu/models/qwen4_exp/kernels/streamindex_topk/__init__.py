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
"""StreamIndex top-k for the QSA (query-sparse attention) indexer."""

from .streamindex_topk import get_kv_cache_shape, streamindex_topk
from .topk_backend import (
    BisectTopK,
    SparseCoreTopK,
    TensorCoreTopK,
    TopKBackend,
    default_topk_backend,
)

__all__ = [
    "BisectTopK",
    "SparseCoreTopK",
    "TensorCoreTopK",
    "TopKBackend",
    "default_topk_backend",
    "get_kv_cache_shape",
    "streamindex_topk",
]
