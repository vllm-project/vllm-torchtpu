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

``tpu_ring_flash_attention`` runs vLLM-Omni's ring attention
(``ring_degree > 1``) through the same kernel, rotating K/V between ranks
inside one Pallas op.
"""

import functools
import math
from functools import cache as run_once
from typing import Any, cast

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

from vllm_torchtpu.kernels.diffusion_attention.kernel import (
    flash_attention_bshd,
    ring_flash_attention_bshd,
)
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


_NON_RING_FLASH_PREFS = frozenset(
    {
        "sdpa",
        "torch",
        "torch_sdpa",
        "tpu_sdpa",
        "cudnn_attn",
        "flashinfer_attn",
        "trtllm_attn",
        "sage_attn",
        "sage_attn_3",
        "flash_attn_hub",
        "flash_attn_3_hub",
    }
)


@functools.cache
def _ring_attention_op(outer_size: int, ring_size: int, uly_size: int):
    """Builds the torch custom op wrapping the Pallas ring attention kernel.

    vLLM-Omni lays out sequence-parallel ranks as (cfg, ring, ulysses), so
    ring partners are ``uly_size`` ranks apart. Ulysses has already moved
    heads before ``run_attention``; each ``uly`` shard runs its own ring over
    its slice of heads.
    """
    mesh = cast(Any, torch).tpu.pallas.get_pallas_mesh(
        axis_names=("batch", "ring", "uly"),
        mesh_shape=(outer_size, ring_size, uly_size),
    )
    spec = PartitionSpec("batch", "ring", "uly", None, None, None, None)

    def _body(q: jax.Array, k: jax.Array, v: jax.Array, scale: float) -> jax.Array:
        out = ring_flash_attention_bshd(
            q[0, 0, 0],
            k[0, 0, 0],
            v[0, 0, 0],
            ring_axis="ring",
            scale=scale,
        )
        return out[None, None, None]

    def _fn(q: jax.Array, k: jax.Array, v: jax.Array, scale: float) -> jax.Array:
        return jax.shard_map(
            functools.partial(_body, scale=scale),
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )(q, k, v)

    op_name = f"pallas::diffusion_ring_attn_{outer_size}_{ring_size}_{uly_size}"
    # jax_op's stub types partition specs as tuple[tuple[str, ...], ...].
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        op_name,
        _fn,
        mesh=mesh,
        input_partition_specs=(spec, spec, spec, None),
        output_partition_specs=(spec,),
    )
    op.register_fake(lambda q, k, v, scale: q.new_empty((*q.shape[:6], v.shape[-1])))
    return op


def tpu_ring_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    *,
    ring_size: int,
    uly_size: int = 1,
    outer_size: int | None = None,
) -> torch.Tensor:
    """Runs Pallas ring flash attention on TPU over (B, S, H, D) inputs."""
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError(
            f"num_heads ({query.shape[2]}) must be a multiple of "
            f"num_kv_heads ({key.shape[2]})"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    if outer_size is None:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else ring_size * uly_size
        )
        outer_size = max(1, world_size // (ring_size * uly_size))
    op = _ring_attention_op(outer_size, ring_size, uly_size)
    q = query.contiguous().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    k = key.contiguous().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    v = value.contiguous().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    out = op(q, k, v, float(scale))
    return out.reshape(*query.shape[:3], value.shape[-1])


def can_run_tpu_ring_attention(
    ring_runner: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any | None,
    *,
    causal: bool = False,
) -> bool:
    """Whether a RingParallelAttention call fits the TPU Pallas ring kernel."""
    pref = getattr(ring_runner, "attn_backend_pref", None)
    if pref is not None and str(pref).lower() in _NON_RING_FLASH_PREFS:
        return False
    if (
        causal
        or query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
        or query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
    ):
        return False
    if (
        key.shape[2] <= 0
        or query.shape[2] % key.shape[2] != 0
        or key.shape[2] != value.shape[2]
        or query.shape[-1] != key.shape[-1]
        or value.shape[-1] % 8 != 0
    ):
        return False
    if attn_metadata is not None:
        if (
            getattr(attn_metadata, "attn_mask", None) is not None
            or getattr(attn_metadata, "joint_query", None) is not None
            or getattr(attn_metadata, "joint_key", None) is not None
            or getattr(attn_metadata, "joint_value", None) is not None
        ):
            return False
        extra = getattr(attn_metadata, "extra", None) or {}
        if (
            extra.get("valid_kv_length") is not None
            or extra.get("gate_compress") is not None
            or extra.get("kv_cache_dtype") not in (None, "float")
        ):
            return False
    sp_group = getattr(ring_runner, "_sp_group", None)
    return int(getattr(sp_group, "ring_world_size", 0)) >= 1


_checked_ring_layouts: set[tuple[int, int, int]] = set()


def check_ring_rank_layout(sp_group: Any, ring_size: int, uly_size: int) -> None:
    """Raises unless the ring group matches the op's (cfg, ring, uly) mesh.

    The mesh is row-major, so it pairs the right ranks only when vLLM-Omni's
    tensor and pipeline parallel sizes are 1 (it orders tp innermost).
    """
    dist = torch.distributed
    ring_group = getattr(sp_group, "ring_group", None)
    if not dist.is_initialized() or not isinstance(ring_group, dist.ProcessGroup):
        return
    key = (id(ring_group), ring_size, uly_size)
    if key in _checked_ring_layouts:
        return
    block = ring_size * uly_size
    ranks = dist.get_process_group_ranks(ring_group)
    base = ranks[0] if ranks else 0
    expected = [base + r * uly_size for r in range(ring_size)]
    if (
        dist.get_world_size() % block != 0
        or ranks != expected
        or base // block != expected[-1] // block
    ):
        raise NotImplementedError(
            f"TPU ring attention expects ring group ranks {expected} "
            f"(cfg x ring x ulysses layout), got {ranks} in world size "
            f"{dist.get_world_size()}; tensor/pipeline parallel > 1 is not "
            "supported with TPU ring attention."
        )
    _checked_ring_layouts.add(key)
