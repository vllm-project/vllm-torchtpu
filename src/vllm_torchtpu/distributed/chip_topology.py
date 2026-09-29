# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.distributed.chip_topology`.
"""

from vllm_torchtpu.llm_tpu_core.distributed.chip_topology import (
    ChipTopology,
    get_chip_topology,
    hierarchical_moe_split,
    logger,
)

__all__ = [
    "ChipTopology",
    "get_chip_topology",
    "hierarchical_moe_split",
    "logger",
]
