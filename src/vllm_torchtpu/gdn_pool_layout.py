# SPDX-License-Identifier: Apache-2.0
"""GDN state layout and byte calculation helpers for unified KV caches.

Re-exports core primitives from `llm_tpu_core.kv_cache.gdn_pool_layout`.
"""

from vllm_torchtpu.llm_tpu_core.kv_cache.gdn_pool_layout import (
    DEFAULT_POOLED_GDN_CONV_STATE_DTYPE,
    DEFAULT_POOLED_GDN_SSM_STATE_DTYPE,
    POOLED_GDN_ARCHITECTURES,
    QWEN_GDN_ARCHITECTURES,
    PooledGDNStateLayout,
    derive_pooled_gdn_state_layout,
    pooled_gdn_conv_state_bytes,
    pooled_gdn_ssm_state_bytes,
    pooled_gdn_state_dtypes,
    pooled_gdn_state_itemsize,
    unified_kv_layout_enabled_for_architecture,
)

__all__ = [
    "DEFAULT_POOLED_GDN_CONV_STATE_DTYPE",
    "DEFAULT_POOLED_GDN_SSM_STATE_DTYPE",
    "POOLED_GDN_ARCHITECTURES",
    "QWEN_GDN_ARCHITECTURES",
    "PooledGDNStateLayout",
    "derive_pooled_gdn_state_layout",
    "pooled_gdn_conv_state_bytes",
    "pooled_gdn_ssm_state_bytes",
    "pooled_gdn_state_dtypes",
    "pooled_gdn_state_itemsize",
    "unified_kv_layout_enabled_for_architecture",
]
