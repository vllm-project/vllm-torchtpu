# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Bridge the torch gdn_attention_core op for gated deltanet attention TPU impl.

Re-exports core primitives from `llm_tpu_core.layers.gdn_attention`.
"""

from vllm_torchtpu.llm_tpu_core.layers.gdn_attention import (
    run_jax_gdn_attention,
)

__all__ = [
    "run_jax_gdn_attention",
]
