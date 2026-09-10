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
"""Whether a KV connector depends on how a KV page encodes K and V."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def connector_reads_kv_page_layout(connector_cls,
                                   vllm_config: "VllmConfig") -> bool:
    """Whether `connector_cls` depends on how a KV page encodes K and V.

    A connector that moves whole pages between a pool and a store on the same
    engine handles opaque byte rows and works with any page encoding. One that
    ships pages to another engine, or slices them by head, has to agree with
    the other side on the encoding. A connector says which it is through a
    `reads_kv_page_layout` classmethod; one without it is taken to read the
    layout.
    """
    reads = getattr(connector_cls, "reads_kv_page_layout", None)
    if reads is None:
        return True
    return bool(reads(vllm_config))
