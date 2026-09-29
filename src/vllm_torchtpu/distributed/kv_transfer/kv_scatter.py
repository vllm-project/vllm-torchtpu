# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.distributed.kv_transfer.kv_scatter`.
"""

import importlib

_mod = importlib.import_module(
    "vllm_torchtpu.llm_tpu_core.distributed.kv_transfer.kv_scatter"
)

globals().update(
    {
        k: getattr(_mod, k)
        for k in dir(_mod)
        if not (k.startswith("__") and k.endswith("__"))
    }
)
