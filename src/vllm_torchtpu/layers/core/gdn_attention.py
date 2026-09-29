# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Bridge the torch gdn_attention_core op for gated deltanet attention TPU impl.

Re-exports core primitives from `llm_tpu_core.layers.gdn_attention`.
"""

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.layers import gdn_attention as _core_gdn
from vllm_torchtpu.llm_tpu_core.layers.gdn_attention import (
    _build_v3_pool_state_plan,
    _derive_pcp_ragged_exchange_descriptors,
    _derive_pcp_rank_major_reorder_indices,
    _exchange_pcp_token_shards_for_head_shards,
    _gdn_pcp_tp_state_axis,
    _gdn_tp_axis_name,
    _pcp_rank_token_count_before,
    _ragged_exchange_pcp_token_shards_for_head_shards,
    _reorder_gdn_qkv_for_pcp,
    _select_replicated_shard_for_pcp_rank,
    _validate_pcp_ragged_exchange_layout_support,
    run_jax_gdn_attention,
    run_jax_gdn_attention_pcp_tp_prefill,
    run_jax_gdn_attention_pooled,
    run_jax_gdn_attention_pooled_local,
    run_jax_gdn_attention_pooled_pcp_prefill,
    run_jax_gdn_attention_pooled_pcp_prefill_projection,
)


class _GdnAttentionModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        if hasattr(_core_gdn, name):
            setattr(_core_gdn, name, value)


sys.modules[__name__] = _GdnAttentionModuleWrapper(sys.modules[__name__])

__all__ = [
    "_build_v3_pool_state_plan",
    "_derive_pcp_ragged_exchange_descriptors",
    "_derive_pcp_rank_major_reorder_indices",
    "_exchange_pcp_token_shards_for_head_shards",
    "_gdn_pcp_tp_state_axis",
    "_gdn_tp_axis_name",
    "_pcp_rank_token_count_before",
    "_ragged_exchange_pcp_token_shards_for_head_shards",
    "_reorder_gdn_qkv_for_pcp",
    "_select_replicated_shard_for_pcp_rank",
    "_validate_pcp_ragged_exchange_layout_support",
    "run_jax_gdn_attention",
    "run_jax_gdn_attention_pcp_tp_prefill",
    "run_jax_gdn_attention_pooled",
    "run_jax_gdn_attention_pooled_local",
    "run_jax_gdn_attention_pooled_pcp_prefill",
    "run_jax_gdn_attention_pooled_pcp_prefill_projection",
]
