# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""In-place Pallas partial slot update for KV cache pools (b/546309078).

Re-exports core primitives from `llm_tpu_core.kernels.partial_kv_update`.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.kernels.partial_kv_update import (
    _blockspec_kernel,
    _fake_scatter_op,
    _jax_scatter_op,
    _make_sublane_block_kernel,
    _pallas_scatter_torch_op,
    _scatter_slots_2d,
    _scatter_slots_nd,
    pallas_index_copy_,
    pallas_scatter_slots,
)

__all__ = [
    "_blockspec_kernel",
    "_fake_scatter_op",
    "_jax_scatter_op",
    "_make_sublane_block_kernel",
    "_pallas_scatter_torch_op",
    "_scatter_slots_2d",
    "_scatter_slots_nd",
    "pallas_index_copy_",
    "pallas_scatter_slots",
]
