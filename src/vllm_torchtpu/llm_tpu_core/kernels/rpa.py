# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Pallas Ragged Paged Attention (RPA) kernel interfaces for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.kernels.experimental.batched_rpa.configs import (
    AttentionScope,
    BlockSizes,
    KVLayout,
    ModelConfigs,
    RpaCase,
    RpaConfigs,
    ServingConfigs,
)


def get_rpa_kernel_version() -> str:
    """Return the active Pallas RPA kernel version."""
    return "v3"


__all__ = [
    "AttentionScope",
    "BlockSizes",
    "KVLayout",
    "ModelConfigs",
    "RpaCase",
    "RpaConfigs",
    "ServingConfigs",
    "get_rpa_kernel_version",
]
