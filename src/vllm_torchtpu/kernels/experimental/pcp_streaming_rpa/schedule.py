# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Runtime schedules for PCP streaming prefill RPA.

Re-exports core primitives from
`llm_tpu_core.kernels.experimental.pcp_streaming_rpa.schedule`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.experimental.pcp_streaming_rpa.schedule import (
    RuntimeScheduleField,
    TilePlanField,
    build_pcp_streaming_schedule_inputs_from_metadata_jax,
)

__all__ = [
    "RuntimeScheduleField",
    "TilePlanField",
    "build_pcp_streaming_schedule_inputs_from_metadata_jax",
]
