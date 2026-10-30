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
"""Diffusion attention backends for TPU execution.

``TpuFlashAttentionBackend`` (the default) runs unmasked, non-causal
bf16 attention through a Pallas flash kernel tuned for diffusion
sequence lengths. Masked, causal, or non-bf16 calls fall back with a
one-time warning to ``TpuSDPAImpl`` (PyTorch SDPA).
"""

from functools import cache as run_once
from typing import Any

import jax
import torch
from torch_tpu._internal import pallas
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

from vllm_torchtpu.kernels.diffusion_attention.kernel import flash_attention_bshd
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class TpuSDPAImpl(SDPAImpl):
    """TPU implementation of SDPA attention for diffusion models."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            query, key, value, attn_metadata, mask_mode="broadcast_k"
        )


class TpuSDPABackend(SDPABackend):
    """TPU attention backend for diffusion models using PyTorch SDPA."""

    @staticmethod
    def get_name() -> str:
        return "TPU_SDPA"

    @staticmethod
    def get_impl_cls() -> type[TpuSDPAImpl]:
        return TpuSDPAImpl


@run_once
def _flash_attention_op():
    """Builds the torch custom op wrapping the Pallas kernel (once, eagerly)."""

    def diffusion_flash_attention(
        q: jax.Array, k: jax.Array, v: jax.Array, scale: float
    ) -> jax.Array:
        return flash_attention_bshd(q, k, v, scale=scale)

    op = pallas.jax_op("pallas::diffusion_flash_attention", diffusion_flash_attention)
    op.register_fake(lambda q, k, v, scale: q.new_empty((*q.shape[:3], v.shape[-1])))
    return op


class TpuFlashAttentionImpl(TpuSDPAImpl):
    """Pallas flash attention for diffusion transformers on TPU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Build the op outside any compiled region.
        self._op = _flash_attention_op()

    def _needs_sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: Any | None,
    ) -> bool:
        attn_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        # TODO: Revisit this fallback once torch_tpu SDPA lets callers select its
        # flash kernel and supports masks; until then masked, causal and non-bf16
        # calls take PyTorch SDPA.
        if self.causal:
            reason = "causal=True"
        elif attn_mask is not None:
            reason = "attn_mask is not None"
        elif query.dtype != torch.bfloat16:
            reason = f"query.dtype={query.dtype} (requires torch.bfloat16)"
        else:
            return False
        logger.warning_once(
            "TpuFlashAttentionImpl falling back to PyTorch SDPA (%s).",
            reason,
        )
        return True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        if query.shape[2] % key.shape[2] != 0:
            # Neither path supports it: vLLM-Omni's SDPA raises the same way.
            raise ValueError(
                f"num_heads ({query.shape[2]}) must be a multiple of "
                f"num_kv_heads ({key.shape[2]})"
            )
        if self._needs_sdpa_fallback(query, key, attn_metadata):
            return super().forward(query, key, value, attn_metadata)
        return self._op(query, key, value, self.softmax_scale)


class TpuFlashAttentionBackend(SDPABackend):
    """TPU diffusion attention backend using a Pallas flash kernel."""

    @staticmethod
    def get_name() -> str:
        return "TPU_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type[TpuFlashAttentionImpl]:
        return TpuFlashAttentionImpl
