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
sequence lengths. Masked, causal, or non-bf16 calls delegate to
``TpuSDPAImpl``, which runs PyTorch SDPA by default and, when
``VLLM_TPU_OMNI_SPLIT_SPLASH=1``, dispatches eligible prefix-masked
joint ``[text, image]`` calls to LSE-merged Pallas Split-Splash. On the
PyTorch SDPA path, ``TpuSDPAImpl`` runs query-broadcast masked calls under
``sdpa_kernel(SDPBackend.MATH)``, which on torch_tpu selects the StableHLO
SDPA lowering.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import cache as run_once
from typing import Any

import jax
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_tpu._internal import pallas
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.diffusion_attention.kernel import flash_attention_bshd
from vllm_torchtpu.kernels.diffusion_attention.split_splash_attention import (
    is_split_splash_supported,
    split_splash_attention,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@run_once
def _get_split_splash_op():
    """Builds the torch custom op wrapping prefix-masked Split-Splash."""

    def core(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: jax.Array,
        prefix_len: int,
        scale: float,
    ) -> jax.Array:
        return split_splash_attention(
            q, k, v, mask, prefix_len=prefix_len, softmax_scale=scale
        )

    op = pallas.jax_op("pallas::omni_split_splash_masked", core)
    op.register_fake(lambda q, k, v, mask, prefix_len, scale: torch.empty_like(q))
    return op


def _prefix_mask_len_for_split_splash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata | None,
    causal: bool,
) -> int | None:
    if (
        not envs.VLLM_TPU_OMNI_SPLIT_SPLASH
        or attn_metadata is None
        or query.device.type != "tpu"
        or query.ndim != 4
        or key.shape != query.shape
        or value.shape != query.shape
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        return None
    if (
        attn_metadata.joint_attn_mask is not None
        or attn_metadata.joint_query is not None
    ):
        return None
    mask = attn_metadata.attn_mask
    if (
        mask is None
        or mask.ndim != 2
        or mask.shape != (query.shape[0], key.shape[1])
        or mask.dtype is not torch.bool
        or mask.device != query.device
    ):
        return None
    prefix_len = attn_metadata.extra.get("prefix_mask_len")
    if type(prefix_len) is not int:
        return None
    if not is_split_splash_supported(
        q_seq_len=query.shape[1],
        kv_seq_len=key.shape[1],
        head_dim=query.shape[3],
        q_heads=query.shape[2],
        kv_heads=key.shape[2],
        causal=causal,
        prefix_len=prefix_len,
    ):
        return None
    return prefix_len


class TpuSDPAImpl(SDPAImpl):
    """TPU implementation of SDPA (and opt-in Split-Splash) for diffusion models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if envs.VLLM_TPU_OMNI_SPLIT_SPLASH and not torch.compiler.is_compiling():
            _get_split_splash_op()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        scale = (
            query.shape[-1] ** -0.5
            if self.softmax_scale is None
            else float(self.softmax_scale)
        )
        prefix_len = _prefix_mask_len_for_split_splash(
            query, key, value, attn_metadata, self.causal
        )
        if prefix_len is not None:
            assert attn_metadata is not None and attn_metadata.attn_mask is not None
            return _get_split_splash_op()(
                query, key, value, attn_metadata.attn_mask, prefix_len, scale
            )
        # ---------------------------------------------------------------------
        # TODO: Remove the math-backend pin, and its canary
        # test_torch_tpu_still_misreads_query_broadcast_attn_mask, once the
        # pinned torch_tpu release has the fix for
        # google-pytorch/torch_tpu#5677 (on torch_tpu main, not in 0.2.0).
        #
        # Problem: torch_tpu 0.2.0's default SDPA lowering, its flash kernel,
        #   misreads an attn_mask of size 1 along the query dim, such as a
        #   (B, 1, 1, K) or (B, H, 1, K) key-padding mask, bool or additive.
        #   The output is wrong even for a mask that keeps every key, and NaN
        #   for some shapes. A mask with a full query dim is read correctly.
        # Fix: (B, 1, 1, K) key-padding masks — along with 1D [K], 2D [B, K]
        #   (reshaped to [B, 1, 1, K]), and any 3D/4D mask with shape[-2] == 1
        #   — now take the SDPBackend.MATH path, which on torch_tpu selects
        #   the StableHLO SDPA lowering (not ATen math). This path materializes
        #   the full [B, H, Sq, Sk] scores, so an OOM or slowdown on long
        #   LTX-2/Wan sequences can be traced to this pin.
        # Effect: masked calls match a float64 reference. Before, LTX-2 under
        #   Ulysses, which pads the audio latents to a multiple of the SP size
        #   and sends a key-padding mask to audio self-attention and
        #   audio-to-video attention, got a NaN row from audio-to-video
        #   attention at per-rank batch >= 2 (LTX-2.3 multi-guidance at
        #   CFG-parallel 2): blank video and NaN audio. At batch 1 the audio
        #   was finite but wrong.
        # ---------------------------------------------------------------------
        mask = attn_metadata.attn_mask if attn_metadata is not None else None
        query_broadcast_mask = mask is not None and (
            mask.ndim < 3 or mask.shape[-2] == 1
        )
        with sdpa_kernel(SDPBackend.MATH) if query_broadcast_mask else nullcontext():
            return self._forward_impl(
                query, key, value, attn_metadata, mask_mode="broadcast_k"
            )


class TpuSDPABackend(SDPABackend):
    """TPU diffusion attention backend using PyTorch SDPA (plus opt-in Split-Splash)."""

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
        # calls take TpuSDPAImpl (PyTorch SDPA, or opt-in Split-Splash).
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
        if (
            _prefix_mask_len_for_split_splash(
                query, key, value, attn_metadata, self.causal
            )
            is not None
        ):
            return super().forward(query, key, value, attn_metadata)
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
