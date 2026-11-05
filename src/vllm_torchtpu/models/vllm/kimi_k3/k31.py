# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi-K3.1: Kimi-K3 with a DSA indexer on every MLA layer, served by the
megakernel backend only."""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackend
from vllm_torchtpu.megakernel import backend as megakernel_backend

from .model import KimiK3ForConditionalGeneration


class KimiIndexKeyCache(torch.nn.Module, AttentionLayerBase):
    """The K3.1 MLA layer's index-key cache, which has no KV spec of its own:
    the runner allocates the megakernel's block-major index tensor and binds
    it under this name."""

    def __init__(self, vllm_config: Any, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def get_attn_backend(self) -> type[PallasMLAttentionBackend]:
        return PallasMLAttentionBackend

    def get_kv_cache_spec(self, vllm_config: Any) -> None:
        return None


class KimiIndexer(torch.nn.Module):
    """The weightless DSA indexer of a K3.1 MLA layer: its index-key cache."""

    def __init__(self, vllm_config: Any, prefix: str) -> None:
        super().__init__()
        self.k_cache = KimiIndexKeyCache(vllm_config, f"{prefix}.k_cache")


class KimiK31ForConditionalGeneration(KimiK3ForConditionalGeneration):
    """Kimi-K3.1 (``model_type`` ``kimi_k31``)."""

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        if megakernel_backend.role(vllm_config) is None:
            raise NotImplementedError(
                "Kimi-K3.1 is served by the megakernel backend only "
                "(--load-format megakernel)"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
