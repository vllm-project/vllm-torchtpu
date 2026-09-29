# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.quantized_matmul.kernel`.
"""

import importlib

_mod = importlib.import_module(
    "vllm_torchtpu.llm_tpu_core.kernels.quantized_matmul.kernel"
)

globals().update(
    {
        k: getattr(_mod, k)
        for k in dir(_mod)
        if not (k.startswith("__") and k.endswith("__"))
    }
)
