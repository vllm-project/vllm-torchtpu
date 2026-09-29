# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.streamindex_topk.streamindex_topk`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.streamindex_topk import (
    streamindex_topk as _mod,
)

globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")})
