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

``TpuBlockSparseAttentionBackend`` runs training-free block-sparse Pallas
FlashAttention (``block_sparse_attention_bshd``) on video DiT self-attention
layers after optional dense warmup steps, delegating warmup steps, skipped
layers, and short sequences to ``TpuFlashAttentionImpl``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache as run_once
from typing import Any
from weakref import WeakSet

import jax
import torch
from torch_tpu._internal import pallas
from vllm.model_executor.models.utils import extract_layer_index
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.data import parse_kv_cache_skip_selector

from vllm_torchtpu.kernels.diffusion_attention.block_sparse import (
    LANES as _SPARSE_LANES,
)
from vllm_torchtpu.kernels.diffusion_attention.block_sparse import (
    block_sparse_attention_bshd,
)
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
def _block_sparse_attention_op():
    """Builds the torch custom op wrapping the Pallas block-sparse kernel."""

    def diffusion_block_sparse_attention(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        scale: float,
        valid_len: int,
        prefix_tokens: int,
        block: int,
        max_density: float,
        tau: float,
        sink_stride: int,
    ) -> jax.Array:
        return block_sparse_attention_bshd(
            q,
            k,
            v,
            scale=scale,
            valid_len=valid_len,
            prefix_tokens=prefix_tokens,
            block=block,
            max_density=max_density,
            tau=tau,
            sink_stride=sink_stride,
        )

    op = pallas.jax_op(
        "pallas::diffusion_block_sparse_attention",
        diffusion_block_sparse_attention,
    )
    op.register_fake(lambda q, k, v, *_args: q.new_empty((*q.shape[:3], v.shape[-1])))
    return op


class TpuFlashAttentionImpl(TpuSDPAImpl):
    """Pallas flash attention for diffusion transformers on TPU."""

    sdpa_fallback_count: int = 0

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
        TpuFlashAttentionImpl.sdpa_fallback_count += 1
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
            attn_metadata is not None
            and attn_metadata.attn_mask is None
            and not self.causal
            and attn_metadata.extra
        ):
            valid_kv_length = attn_metadata.extra.get("valid_kv_length")
            if valid_kv_length is not None:
                if isinstance(valid_kv_length, bool) or not isinstance(
                    valid_kv_length, int
                ):
                    raise ValueError("valid_kv_length must be a Python integer")
                if not 0 < valid_kv_length <= key.shape[1]:
                    raise ValueError(
                        "valid_kv_length must be within the K/V sequence, "
                        f"got {valid_kv_length} for length {key.shape[1]}"
                    )
                if valid_kv_length < key.shape[1]:
                    key = key[:, :valid_kv_length]
                    value = value[:, :valid_kv_length]
        if self._needs_sdpa_fallback(query, key, attn_metadata):
            return super().forward(query, key, value, attn_metadata)
        return self._op(query, key, value, self.softmax_scale)


class TpuFlashAttentionBackend(SDPABackend):
    """TPU diffusion attention backend using a Pallas flash kernel."""

    supports_prefix_kv_slicing: bool = True

    @staticmethod
    def get_name() -> str:
        return "TPU_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type[TpuFlashAttentionImpl]:
        return TpuFlashAttentionImpl


_ACTIVE_BLOCK_SPARSE_IMPLS: WeakSet[TpuBlockSparseAttentionImpl] = WeakSet()


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


def refresh_block_sparse_step_gates(
    step_idx: int | None,
    total_steps: int | None,
) -> None:
    """Update per-layer boolean step gates outside ``torch.compile`` regions.

    Keeping ``_sparse_step_active`` as a plain ``bool`` rather than reading the
    integer ``denoise_step_idx`` inside compiled DiT blocks ensures TorchDynamo
    compiles at most two graphs per shape (one dense warmup graph and one sparse
    graph) instead of recompiling on every integer step index ``0..N-1``.
    """
    for impl in tuple(_ACTIVE_BLOCK_SPARSE_IMPLS):
        impl._update_step_gate(step_idx, total_steps)


class TpuBlockSparseAttentionImpl(TpuFlashAttentionImpl):
    """Pallas block-sparse FlashAttention for diffusion transformers on TPU."""

    sparse_call_count: int = 0
    dense_call_count: int = 0

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
        role: str = "self",
        **extra_impl_args: Any,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            qkv_layout=qkv_layout,
            backend_kwargs=None,
            role=role,
            **extra_impl_args,
        )
        bk = dict(backend_kwargs or {})
        self.prefix = prefix
        self.role = role
        self.layer_index = _try_extract_layer_index(prefix)

        raw_sparsity = bk.get("sparsity")
        self.sparsity = float(raw_sparsity) if raw_sparsity is not None else 0.75
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(
                "TpuBlockSparseAttention sparsity must be in [0, 1), "
                f"got {self.sparsity}"
            )
        self.max_density = 1.0 - self.sparsity

        self.start_step = int(bk.get("start_step", 0))
        self.end_step = int(bk.get("end_step", 0))
        parsed_skip = parse_kv_cache_skip_selector(bk.get("skip_layers"))
        self.skip_layers: frozenset[int] = frozenset(parsed_skip or ())

        raw_block = bk.get("block_size")
        self._explicit_block_size = raw_block is not None
        self.block_size = int(raw_block) if raw_block is not None else 1024
        if self.block_size <= 0 or self.block_size % _SPARSE_LANES != 0:
            raise ValueError(
                f"block_size must be a positive multiple of {_SPARSE_LANES}, "
                f"got {self.block_size}"
            )
        self.tau = float(bk.get("tau", 1.0))
        raw_stride = bk.get("sink_stride")
        self._explicit_sink_stride = raw_stride is not None
        self.sink_stride = (
            int(raw_stride)
            if raw_stride is not None
            else max(4, 8192 // self.block_size)
        )

        self._layer_permanently_dense = (
            self.causal
            or self.sparsity <= 0.0
            or (self.role != "self")
            or (self.layer_index is not None and self.layer_index in self.skip_layers)
        )
        self._sparse_step_active = (
            not self._layer_permanently_dense and self.start_step == 0
        )
        self._sparse_op = _block_sparse_attention_op()
        _ACTIVE_BLOCK_SPARSE_IMPLS.add(self)

    def _update_step_gate(
        self,
        step_idx: int | None,
        total_steps: int | None,
    ) -> None:
        if self._layer_permanently_dense:
            self._sparse_step_active = False
            return
        if step_idx is None:
            self._sparse_step_active = self.start_step == 0
            return
        if step_idx < self.start_step:
            self._sparse_step_active = False
            return
        if (
            self.end_step > 0
            and total_steps is not None
            and step_idx >= total_steps - self.end_step
        ):
            self._sparse_step_active = False
            return
        self._sparse_step_active = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        if query.shape[2] % key.shape[2] != 0:
            raise ValueError(
                f"num_heads ({query.shape[2]}) must be a multiple of "
                f"num_kv_heads ({key.shape[2]})"
            )
        attn_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        extra = (
            attn_metadata.extra
            if (attn_metadata is not None and attn_metadata.extra)
            else None
        )
        valid_len = query.shape[1]
        prefix_tokens = 0
        eff_max_density = self.max_density
        if extra is not None:
            raw_valid = extra.get("valid_kv_length")
            if raw_valid is not None:
                if isinstance(raw_valid, bool) or not isinstance(raw_valid, int):
                    raise ValueError("valid_kv_length must be a Python integer")
                if not 0 < raw_valid <= key.shape[1]:
                    raise ValueError(
                        "valid_kv_length must be within the K/V sequence, "
                        f"got {raw_valid} for length {key.shape[1]}"
                    )
                valid_len = raw_valid
            raw_prefix = extra.get("prefix_tokens", 0)
            if isinstance(raw_prefix, int) and not isinstance(raw_prefix, bool):
                prefix_tokens = max(0, min(raw_prefix, valid_len))
            h3_sparsity = extra.get("vsa_h3_sparsity")
            if h3_sparsity is not None:
                eff_max_density = 1.0 - float(h3_sparsity)

        eff_block = (
            self.block_size
            if self._explicit_block_size
            else (2048 if valid_len >= 16384 else 1024)
        )
        eff_stride = (
            self.sink_stride
            if self._explicit_sink_stride
            else (8 if eff_block >= 1024 else max(4, 8192 // eff_block))
        )

        can_sparse = (
            self._sparse_step_active
            and not self._layer_permanently_dense
            and attn_mask is None
            and query.dtype == torch.bfloat16
            and key.dtype == torch.bfloat16
            and value.dtype == torch.bfloat16
            and query.shape[1] == key.shape[1]
            and query.shape[-1] % _SPARSE_LANES == 0
            and valid_len > 3 * eff_block
            and eff_max_density < 1.0
        )
        if not can_sparse:
            if not torch.compiler.is_compiling():
                TpuBlockSparseAttentionImpl.dense_call_count += 1
            return super().forward(query, key, value, attn_metadata)

        if not torch.compiler.is_compiling():
            TpuBlockSparseAttentionImpl.sparse_call_count += 1
        return self._sparse_op(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            float(self.softmax_scale),
            int(valid_len),
            int(prefix_tokens),
            int(eff_block),
            float(eff_max_density),
            float(self.tau),
            int(eff_stride),
        )


class TpuBlockSparseAttentionBackend(SDPABackend):
    """TPU diffusion block-sparse attention backend using Pallas."""

    supports_prefix_kv_slicing: bool = True

    @staticmethod
    def get_name() -> str:
        return "TPU_BLOCK_SPARSE"

    @staticmethod
    def get_impl_cls() -> type[TpuBlockSparseAttentionImpl]:
        return TpuBlockSparseAttentionImpl
