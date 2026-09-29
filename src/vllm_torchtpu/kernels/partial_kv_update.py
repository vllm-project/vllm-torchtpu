# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""In-place Pallas partial slot update for KV cache pools (b/546309078).

Re-exports core primitives from `llm_tpu_core.kernels.partial_kv_update`.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.kernels.partial_kv_update import *  # noqa: F403
