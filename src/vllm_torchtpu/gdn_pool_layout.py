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
    """Derive the physical token extent of pooled GDN state regions."""
    if min(ssm_bytes, conv_bytes, token_bytes) <= 0:
        raise ValueError("pooled GDN byte sizes must be positive")
    if ssm_bytes % token_bytes:
        raise ValueError(
            "pooled GDN SSM state must occupy complete FA token rows")

    ssm_tokens = ssm_bytes // token_bytes
    conv_tokens = 1
    while (conv_tokens * token_bytes < conv_bytes
           or ssm_tokens % conv_tokens != 0):
        conv_tokens *= 2
        if conv_tokens > ssm_tokens:
            raise ValueError(
                "pooled GDN conv tile cannot be placed after the SSM region")

    return PooledGDNStateLayout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=token_bytes,
        ssm_tokens=ssm_tokens,
        conv_tokens=conv_tokens,
    )
