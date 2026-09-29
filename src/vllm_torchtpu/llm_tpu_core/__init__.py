# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""LLM TPU Core subsystem.

Contains all lower-level acceleration primitives (Pallas kernels, layers,
distributed ops, and KV cache materializer) for Google Cloud TPU hardware,
with strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core import (
    config,
    distributed,
    kernels,
    kv_cache,
    layers,
    logger,
)

__version__ = "0.1.0"

__all__ = [
    "config",
    "distributed",
    "kernels",
    "kv_cache",
    "layers",
    "logger",
]
