# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.kimi_k3.collectives`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.kimi_k3.collectives import (
    CYCLES,
    gather_moe_inputs,
    gather_project,
    mla_chip_layout,
    mla_preprocess,
    pack_mla_gate,
    project_reduce_scatter,
    ring_collective,
    ring_tables,
)

__all__ = [
    "CYCLES",
    "gather_moe_inputs",
    "gather_project",
    "mla_chip_layout",
    "mla_preprocess",
    "pack_mla_gate",
    "project_reduce_scatter",
    "ring_collective",
    "ring_tables",
]
