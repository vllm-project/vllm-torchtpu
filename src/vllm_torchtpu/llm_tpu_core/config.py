# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Configuration dataclasses, protocols, and environment flags for llm_tpu_core.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class KVCacheLayout(str, Enum):
    NHD = "NHD"
    HND = "HND"


def get_kv_cache_layout() -> str:
    """Returns the configured KV cache layout from environment."""
    return os.environ.get("VLLM_KV_CACHE_LAYOUT", "NHD").upper()


def is_hnd_layout() -> bool:
    """Returns True if the current KV cache layout is HND."""
    return get_kv_cache_layout() == "HND"


def is_block_major_enabled() -> bool:
    """Returns True if block-major KV layout is enabled via environment."""
    return os.environ.get("VLLM_TPU_BLOCK_MAJOR_KV", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def is_gdn_conv_qk_pair_layout_enabled() -> bool:
    """Returns True if GDN conv QK pair layout is enabled via environment."""
    return os.environ.get("TPU_GDN_CONV_QK_PAIR_LAYOUT", "0").lower() in (
        "1",
        "true",
        "yes",
    )


@dataclass(frozen=True)
class CoreCompilationConfig:
    """Minimal compilation parameters for hardware-adapted TPU kernels."""

    backend: str = "openxla"
    use_pallas: bool = True
    fuse_layers: bool = True
