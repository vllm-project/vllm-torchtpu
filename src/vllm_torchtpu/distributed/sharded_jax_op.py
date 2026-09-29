# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.distributed.sharded_jax_op`.
"""

from vllm_torchtpu.llm_tpu_core.distributed.sharded_jax_op import (
    logger,
    sharded_jax_op,
)

__all__ = [
    "logger",
    "sharded_jax_op",
]
