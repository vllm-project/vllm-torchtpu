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

# Provenance: copied from src/vllm_torchtpu/models/vllm/qwen4_exp/ple_offload.py at
# 473dda4acce94f0d133b73352ede6c9f9819a7f2.
#   Owned by models/qwen4_exp from this copy on.
"""Host PLE lookup boundary for vLLM piecewise compilation."""

from typing import Any

import torch
from vllm.forward_context import get_forward_context

from .ple_storage import HostPLETable


@torch.library.custom_op("qwen4_exp::host_lookup", mutates_args=())
def host_lookup(
    ids: torch.Tensor,
    local_tokens: int,
    embedding_dim: int,
    fp8: bool,
    prefix: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[prefix]
    table = layer.ple_embedding.ngram_embedding
    assert isinstance(table, HostPLETable)
    raw = table.lookup_host(ids.cpu())
    return raw.reshape(local_tokens, embedding_dim).to(
        ids.device, non_blocking=raw.is_pinned()
    )


@host_lookup.register_fake
def _host_lookup_fake(ids, local_tokens, embedding_dim, fp8, prefix):
    return ids.new_empty(
        (local_tokens, embedding_dim),
        dtype=torch.int8 if fp8 else torch.int16,
    )


# Split variant of host_lookup (VLLM_TPU_PLE_HOST_OVERLAP): `start` begins the
# ids' device-to-host copy and records an event; the decoder layers before the
# PLE layer are dispatched next; `finish` then waits only for that copy, so the
# host gather and the upload overlap those layers on the device instead of
# draining it. Both are graph splitting ops, run eagerly in order.
_pending: dict[str, tuple[torch.Tensor, Any]] = {}


@torch.library.custom_op("qwen4_exp::host_lookup_start", mutates_args=())
def host_lookup_start(ids: torch.Tensor, prefix: str) -> torch.Tensor:
    host_ids = ids.to("cpu", non_blocking=True)
    event = torch.tpu.Event() if hasattr(torch, "tpu") else None
    if event is not None:
        event.record()
    _pending[prefix] = (host_ids, event)
    return ids.new_empty(0)


@host_lookup_start.register_fake
def _host_lookup_start_fake(ids, prefix):
    return ids.new_empty(0)


@torch.library.custom_op("qwen4_exp::host_lookup_finish", mutates_args=())
def host_lookup_finish(
    token: torch.Tensor,
    after: torch.Tensor,
    local_tokens: int,
    embedding_dim: int,
    fp8: bool,
    prefix: str,
) -> torch.Tensor:
    """``token`` orders this after ``host_lookup_start``; ``after`` (the hidden
    state of the last layer before the PLE layer) orders it after those layers'
    dispatch."""
    host_ids, event = _pending.pop(prefix)
    if event is not None:
        event.synchronize()
    layer = get_forward_context().no_compile_layers[prefix]
    table = layer.ple_embedding.ngram_embedding
    assert isinstance(table, HostPLETable)
    raw = table.lookup_host(host_ids)
    return raw.reshape(local_tokens, embedding_dim).to(
        after.device, non_blocking=raw.is_pinned()
    )


@host_lookup_finish.register_fake
def _host_lookup_finish_fake(token, after, local_tokens, embedding_dim, fp8, prefix):
    return after.new_empty(
        (local_tokens, embedding_dim),
        dtype=torch.int8 if fp8 else torch.int16,
    )
