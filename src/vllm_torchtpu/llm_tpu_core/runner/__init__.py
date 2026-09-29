# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""TPU runner device operations, padding, and core runtime interfaces."""

from vllm_torchtpu.llm_tpu_core.runner.device_ops import (
    LogprobsResult,
    compute_logits_from_hidden_states,
    compute_selected_logits,
    gather_logprobs,
    reset_read_offsets,
    reset_read_offsets_impl,
    rollback_offsets_migrate,
    rollback_offsets_migrate_impl,
    rollback_offsets_seed,
    rollback_offsets_seed_impl,
    sample_from_logits,
    substitute_placeholder_token,
)
from vllm_torchtpu.llm_tpu_core.runner.device_ops import (
    rollback_offsets_migrate_compiled as _rollback_offsets_migrate_compiled,
)
from vllm_torchtpu.llm_tpu_core.runner.device_ops import (
    rollback_offsets_seed_compiled as _rollback_offsets_seed_compiled,
)
from vllm_torchtpu.llm_tpu_core.runner.padding import (
    MIN_NUM_SEQS,
    get_num_slices_per_kv_cache_update_block,
    get_padded_num_kv_cache_update_slices,
    get_padded_num_reqs_with_upper_limit,
    get_padded_token_len,
    get_req_paddings,
    prev_power_of_2,
)

__all__ = [
    "LogprobsResult",
    "compute_logits_from_hidden_states",
    "compute_selected_logits",
    "gather_logprobs",
    "reset_read_offsets",
    "reset_read_offsets_impl",
    "rollback_offsets_migrate",
    "_rollback_offsets_migrate_compiled",
    "rollback_offsets_migrate_impl",
    "rollback_offsets_seed",
    "_rollback_offsets_seed_compiled",
    "rollback_offsets_seed_impl",
    "sample_from_logits",
    "substitute_placeholder_token",
    "MIN_NUM_SEQS",
    "get_num_slices_per_kv_cache_update_block",
    "get_padded_num_kv_cache_update_slices",
    "get_padded_num_reqs_with_upper_limit",
    "get_padded_token_len",
    "get_req_paddings",
    "prev_power_of_2",
]
