# SPDX-License-Identifier: Apache-2.0
"""Diffusion attention backends for TPU execution.

``TpuFlashAttentionBackend`` (the default) runs unmasked, non-causal
bf16 attention through a Pallas flash kernel tuned for diffusion
sequence lengths. Key-padded calls that declare a valid K/V prefix
(``extra["valid_kv_length"]`` plus the matching ``[B, S_kv]`` boolean key
mask, as in MiniMax-H3's ``[real, pad]`` packing) run on the same kernel
with the padding keys masked in-kernel. Other masked, causal, or non-bf16
calls fall back with a one-time warning to ``TpuSDPAImpl`` (PyTorch SDPA).
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


@run_once
def _flash_attention_kv_len_op():
    """Builds the key-padded variant of the Pallas op (once, eagerly)."""

    def diffusion_flash_attention_kv_len(
        q: jax.Array, k: jax.Array, v: jax.Array, kv_len: jax.Array, scale: float
    ) -> jax.Array:
        return flash_attention_bshd(q, k, v, scale=scale, kv_len=kv_len)

    op = pallas.jax_op(
        "pallas::diffusion_flash_attention_kv_len", diffusion_flash_attention_kv_len
    )
    op.register_fake(
        lambda q, k, v, kv_len, scale: q.new_empty((*q.shape[:3], v.shape[-1]))
    )
    return op


class TpuFlashAttentionImpl(TpuSDPAImpl):
    """Pallas flash attention for diffusion transformers on TPU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Build the op outside any compiled region.
        self._op = _flash_attention_op()

    def _key_padding_kv_len(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: Any | None,
    ) -> torch.Tensor | None:
        """Per-row count of leading real keys for a prefix key-padded call.

        Returns ``attn_mask.sum(-1)`` as a ``(B,)`` int32 device tensor for
        non-causal bf16 calls whose metadata has an int
        ``extra["valid_kv_length"]`` (vLLM-Omni's contiguous valid K/V
        prefix) and the matching ``(B, S_kv)`` boolean key mask. Returns
        ``None`` for every other call, which keeps its existing path.
        """
        # ---------------------------------------------------------------------
        # Upstream vLLM-Omni Root Cause Tracking:
        # Upstream: MiniMaxH3Attention._run_packed_attention packs [real, pad]
        # rows. For a backend with neither supports_prefix_kv_slicing nor
        # supports_packed_mask_free (SDPABackend, which this backend
        # inherits), it passes attn_mask = arange(S)[None] < used and
        # extra["valid_kv_length"] = used whenever used < S
        # (minimax_h3_transformer.py:587-622 at vllm-omni e1d77a3).
        #
        # Problem on TPU: the sequence is padded unless the used length is a
        # multiple of 64, so for practically every request all 50 DiT block
        # attentions took the PyTorch SDPA fallback ("TpuFlashAttentionImpl
        # falling back to PyTorch SDPA (attn_mask is not None)" in every
        # v7x-8 run). SDPA materializes the (S, S) scores per head: ~4 GB
        # per layer at 12k tokens with 7 heads per TensorCore.
        #
        # Fix: run the Pallas kernel with the real-key count as a scalar
        # prefetch and mask padding keys in-kernel. The count is a device
        # value, so shapes stay static and no length recompiles. Every query
        # row attends over the real keys, as with SDPA's broadcast key mask.
        # The supports_* flags stay False: flipping them changes how every
        # model that reads them packs its inputs.
        #
        # REMOVAL / FIX CONDITION:
        # Remove when torch_tpu SDPA runs a flash kernel with masks, or when
        # vLLM-Omni lets a backend use valid_kv_length without building the
        # dense mask (then take kv_len from it instead of the mask).
        # ---------------------------------------------------------------------
        if attn_metadata is None or self.causal:
            return None
        attn_mask = getattr(attn_metadata, "attn_mask", None)
        extra = getattr(attn_metadata, "extra", None)
        if not isinstance(attn_mask, torch.Tensor) or not isinstance(extra, dict):
            return None
        valid_kv_length = extra.get("valid_kv_length")
        batch, kv_seq_len = key.shape[0], key.shape[1]
        if (
            type(valid_kv_length) is not int
            or not 0 < valid_kv_length <= kv_seq_len
            or attn_mask.dtype != torch.bool
            or tuple(attn_mask.shape) != (batch, kv_seq_len)
            or query.dtype != torch.bfloat16
            or key.dtype != torch.bfloat16
        ):
            return None
        return attn_mask.sum(dim=-1, dtype=torch.int32)

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
        kv_len = self._key_padding_kv_len(query, key, attn_metadata)
        if kv_len is not None:
            return _flash_attention_kv_len_op()(
                query, key, value, kv_len, self.softmax_scale
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
