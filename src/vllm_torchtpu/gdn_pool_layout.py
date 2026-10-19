# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Architectures whose recurrent layers read and write their state through the
# attention-shaped pool. The Qwen3 and Qwen4Exp families build on
# QwenGatedDeltaNetAttention, which the TPU backend replaces with
# VllmGatedDeltaNetAttention; that subclass
# accepts a single pooled buffer instead of a separate conv/ssm pair.
# KimiLinearForCausalLM runs its KDA layers through the gather/scatter pooled
# op (build_kimi_pooled_kda_op, around the fused conv1d + GDN v3 kernel)
# instead. Other hybrid architectures unpack two state tensors unconditionally,
# so they stay on the per-layer KV caches until they grow a pooled path.
# Extend this set in the same change that adds one.
QWEN_GDN_ARCHITECTURES = frozenset(
    {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    }
)
POOLED_GDN_ARCHITECTURES = QWEN_GDN_ARCHITECTURES | frozenset(
    {
        # Kimi-Linear and Kimi-K3 run their KDA layers through the gather/scatter
        # pooled op (build_kimi_pooled_kda_op, around the fused conv1d + GDN v3
        # kernel). Kimi-K3's TP32 geometry is validated: prefix-cache hits are
        # byte-identical to misses and GSM8K scores 100/100 with and without the
        # pool.
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    }
)

# The pooled GDN kernel's state dtypes inside the unified pool are fixed for
# Conv, while SSM follows the layer's declared MambaSpec dtype. Everything
# that describes pool bytes — the kernel's state plan and the Raiden reshard
# manifest — must take its dtypes and sizes from here, so no two consumers can
# disagree about the physical layout.
#
# These are torch dtype *names*, not torch.dtype objects: this module is
# imported by the JAX kernel path and by deviceless tests, so it must stay
# importable without torch.
DEFAULT_POOLED_GDN_CONV_STATE_DTYPE = "torch.bfloat16"
DEFAULT_POOLED_GDN_SSM_STATE_DTYPE = "torch.float32"
_ITEMSIZE_BY_DTYPE = {"torch.bfloat16": 2, "torch.float32": 4}


def pooled_gdn_state_dtypes(
    dtypes: tuple[object, ...],
) -> tuple[object, object]:
    """Resolve declared Conv/SSM dtypes to their pooled physical dtypes."""
    return DEFAULT_POOLED_GDN_CONV_STATE_DTYPE, dtypes[1]


def pooled_gdn_state_itemsize(dtype: object) -> int:
    name = str(dtype)
    if not name.startswith("torch."):
        name = f"torch.{name}"
    try:
        return _ITEMSIZE_BY_DTYPE[name]
    except KeyError as exc:
        raise ValueError(f"unsupported pooled GDN state dtype: {dtype}") from exc


def unified_kv_layout_enabled_for_architecture(architecture: str | None) -> bool:
    """Whether `architecture` runs on the attention-shaped unified KV pool.

    Split out of `platforms.tpu_block_size_utils.unified_kv_layout_enabled`
    (which delegates here) so the GDN layer can ask the same question without
    importing `vllm_torchtpu.platforms` -- that package's __init__ pulls in
    TpuPlatform, which imports the layers back.
    """
    from vllm_torchtpu import envs as tpu_envs

    override = tpu_envs.TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
    if override is not None:
        return override
    return architecture in POOLED_GDN_ARCHITECTURES


def pooled_gdn_conv_state_bytes(*, kernel_size: int, conv_dim: int) -> int:
    """Live bytes of one slot's conv region: (kernel_size - 1) dense bf16
    rows of conv_dim channels. The kernel keeps no spec-decode widening in
    the pool (verify windows roll back via per-slot checkpoints instead)."""
    return (
        (kernel_size - 1)
        * conv_dim
        * pooled_gdn_state_itemsize(DEFAULT_POOLED_GDN_CONV_STATE_DTYPE)
    )


def pooled_gdn_ssm_state_bytes(
    *,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: object = DEFAULT_POOLED_GDN_SSM_STATE_DTYPE,
) -> int:
    """Live bytes of one slot's head-major, dense SSM region in ``dtype``."""
    return num_v_heads * head_k_dim * head_v_dim * pooled_gdn_state_itemsize(dtype)


@dataclass(frozen=True)
class PooledGDNStateLayout:
    ssm_bytes: int
    conv_bytes: int
    token_bytes: int
    ssm_tokens: int
    conv_tokens: int

    @property
    def required_tokens(self) -> int:
        return self.ssm_tokens + self.conv_tokens

    @property
    def required_bytes(self) -> int:
        return self.required_tokens * self.token_bytes


def derive_pooled_gdn_state_layout(
    *,
    ssm_bytes: int,
    conv_bytes: int,
    token_bytes: int,
) -> PooledGDNStateLayout:
    """Derive the physical token extent of pooled GDN state regions.

    If ssm_bytes cannot be evenly divided into FA token rows, its last row
    is padded.
    """
    if min(ssm_bytes, conv_bytes, token_bytes) <= 0:
        raise ValueError("pooled GDN byte sizes must be positive")

    ssm_tokens = (ssm_bytes + token_bytes - 1) // token_bytes
    conv_tokens = 1
    while conv_tokens * token_bytes < conv_bytes:
        conv_tokens *= 2

    return PooledGDNStateLayout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=token_bytes,
        ssm_tokens=ssm_tokens,
        conv_tokens=conv_tokens,
    )


def prefill_checkpoint_enabled(vllm_config: VllmConfig) -> bool:
    """Enables checkpoints for pooled GDN with block-aligned prefix hits."""
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    if (
        cache.mamba_cache_mode != "align"
        or parallel.decode_context_parallel_size != 1
        or not unified_kv_layout_enabled_for_architecture(
            vllm_config.model_config.architecture
        )
    ):
        return False
    assert cache.mamba_block_size is not None
    # vLLM v0.31.0 can hash a partial-hit state copy at the next block boundary
    # without writing it when an internal checkpoint skips that boundary.
    return cache.prefix_match_unit is None or cache.prefix_match_unit >= (
        cache.mamba_block_size * parallel.prefill_context_parallel_size
    )


@dataclass(frozen=True)
class PrefillCheckpointUnits:
    """Units of the prefill checkpoint rule, as the engine core applies it.

    All token sizes are global across PCP ranks: mamba_block_size is
    MambaSpec.block_size * pcp_world_size, the column stride of the Mamba
    block table.
    """

    hash_block_size: int
    mamba_block_size: int
    alignment: int
    drop_eagle_block: bool

    def fill_offsets(
        self, out: np.ndarray, seq_lens: np.ndarray, query_lens: np.ndarray
    ) -> None:
        """Populates per-request logical query checkpoint offsets into out (0 if none).

        seq_lens (chunk ends) and query_lens (scheduled tokens) cover active
        requests; any padded tail in out is zeroed.
        """
        # Lazy import to avoid a circular import while vllm.platforms initializes.
        from vllm.model_executor.layers.mamba.checkpoint import (
            compute_mamba_prefill_checkpoints,
        )

        out[:] = 0
        # A checkpoint lies at least one hash block past the query start.
        rows = np.flatnonzero(query_lens > self.hash_block_size)
        if rows.size:
            offsets, _ = compute_mamba_prefill_checkpoints(
                [int(n) for n in seq_lens[rows]],
                [int(n) for n in query_lens[rows]],
                hash_block_size=self.hash_block_size,
                mamba_block_size=self.mamba_block_size,
                checkpoint_alignment=self.alignment,
                drop_eagle_block=self.drop_eagle_block,
            )
            out[rows] = np.asarray(offsets, dtype=np.int32)
