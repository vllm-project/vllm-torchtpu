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
"""Hook for a drafter whose attention KV is DCP-sharded like the target's.

Under decode context parallelism a DFlash/DSpark drafter's attention keeps
its KV whole on every rank by default. A draft model that inherits
`DraftDcpAttentionProvider` may instead return a `DraftDcpAttention` for a
layer from ``make_draft_dcp_attention``: that layer's KV then stays
DCP-sharded like the target's, and
`PallasAttentionBackendImpl` routes the layer's context K/V write and its
attention to the returned implementation. Off by default: the base method
returns None.
"""

from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata


class DraftDcpAttention(Protocol):
    """Attention of one drafter layer over a DCP-sharded KV cache.

    ``layer`` is the attention layer; its ``draft_dcp_rank`` buffer holds
    the DCP rank (``int32 [1]``), so every rank traces one program. Build
    any custom ops at construction, outside the compiled forward.
    """

    def write(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Write context K/V (``[T, kv_heads, head_size]``, own KV heads)
        into the rank's share of the cache, in place."""
        ...

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        """Attention output ``[T, heads, head_size]`` of the query block
        (``query [T, heads, head_size]``) over the cached context plus the
        block's own ``key``/``value``. The block is not written."""
        ...


class DraftDcpAttentionProvider:
    """Mixin for a DFlash/DSpark draft model that supplies the attention of
    its layers whose KV is DCP-sharded like the target's.

    ``DFlashProposer.load_model`` calls ``make_draft_dcp_attention`` for
    each draft attention layer under DCP; a layer for which it returns None
    keeps its KV whole on every rank (the default). Once the layers are set
    up it calls ``prepare_sharded_context_kv`` once.
    """

    def make_draft_dcp_attention(
        self,
        layer: torch.nn.Module,
        attn_impl: Any,
        *,
        dcp: int,
        kv_head_replicas: int,
        max_query_len: int,
    ) -> DraftDcpAttention | None:
        """The DCP-sharded attention of ``layer`` (whose implementation is
        ``attn_impl``), or None to keep its KV whole on every rank.
        ``kv_head_replicas`` is the tensor-parallel ranks per draft KV head;
        ``max_query_len`` the longest query block."""
        return None

    def prepare_sharded_context_kv(
        self, group: Any, dcp: int, kv_head_replicas: int
    ) -> None:
        """Called once under DCP after the draft layers are set up."""
        return None
