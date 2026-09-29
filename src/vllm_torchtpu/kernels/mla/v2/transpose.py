# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.mla.v2.transpose`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.mla.v2.transpose import (
    get_reshape_dimension,
    identity_fn_generator,
    logger,
    pin_vmem_custom_call,
    prev_closest_valid_divisor,
    xpose_full,
    xpose_pipeline,
)

__all__ = [
    "get_reshape_dimension",
    "identity_fn_generator",
    "logger",
    "pin_vmem_custom_call",
    "prev_closest_valid_divisor",
    "xpose_full",
    "xpose_pipeline",
]
