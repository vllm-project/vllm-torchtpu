# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

# Architectures whose recurrent layers read and write their state through the
# attention-shaped pool. These build on QwenGatedDeltaNetAttention, which the
# TPU backend replaces with VllmGatedDeltaNetAttention; that subclass is what
# accepts a single pooled buffer instead of a separate conv/ssm pair. Other
# hybrid architectures unpack two state tensors unconditionally, so they stay
# on the per-layer KV caches until they grow a pooled path. Extend this set in
# the same change that adds one.
POOLED_GDN_ARCHITECTURES = frozenset({
    "Qwen3NextForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
})

# The pooled GDN kernel's state dtypes inside the unified pool are fixed:
# conv rows are bf16 and the SSM region is fp32, independent of the layer's
# declared MambaSpec dtypes (which only drive vLLM's page-size accounting).
# Everything that describes pool bytes — the kernel's state plan and the
# Raiden reshard manifest — must take its dtypes and sizes from here, so no
# two consumers can disagree about the physical layout.
#
# These are torch dtype *names*, not torch.dtype objects: this module is
# imported by the JAX kernel path and by deviceless tests, so it must stay
# importable without torch.
POOLED_GDN_CONV_STATE_DTYPE = "torch.bfloat16"
POOLED_GDN_SSM_STATE_DTYPE = "torch.float32"

# Derived from the dtypes above rather than declared separately, so changing
# a state dtype cannot leave a stale byte width behind.
_ITEMSIZE_BY_DTYPE = {"torch.bfloat16": 2, "torch.float32": 4}
POOLED_GDN_CONV_STATE_ITEMSIZE = _ITEMSIZE_BY_DTYPE[
    POOLED_GDN_CONV_STATE_DTYPE]
POOLED_GDN_SSM_STATE_ITEMSIZE = _ITEMSIZE_BY_DTYPE[POOLED_GDN_SSM_STATE_DTYPE]


def unified_kv_layout_enabled_for_architecture(
        architecture: str | None) -> bool:
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
    return (kernel_size - 1) * conv_dim * POOLED_GDN_CONV_STATE_ITEMSIZE


def pooled_gdn_ssm_state_bytes(*, num_v_heads: int, head_k_dim: int,
                               head_v_dim: int) -> int:
    """Live bytes of one slot's SSM region: fp32, head-major and dense."""
    return (num_v_heads * head_k_dim * head_v_dim *
            POOLED_GDN_SSM_STATE_ITEMSIZE)


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
