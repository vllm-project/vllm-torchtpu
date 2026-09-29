# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper for pipeline-parallel shift collectives.

Re-exports core primitives from `vllm_torchtpu.llm_tpu_core.distributed.pp_shift`.
"""

from vllm_torchtpu.llm_tpu_core.distributed.pp_shift import (
    AXIS,
    get_or_create_pp_mesh,
    pp_permute_op,
    pp_positions,
)

__all__ = [
    "AXIS",
    "get_or_create_pp_mesh",
    "pp_positions",
    "pp_permute_op",
]
