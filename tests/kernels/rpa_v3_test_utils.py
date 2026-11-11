# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared test helpers for RPA v3 and TurboQuant unit and TPU parity tests."""

from __future__ import annotations

import jax
import pytest
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)

from vllm_torchtpu.layers.core import turboquant as tq


def require_tpu() -> None:
    """Fail the test immediately if JAX's active backend is not TPU."""
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def make_tq_layout_and_tables(
    preset: str, head_dim: int
) -> tuple[tq.TQLayout, tq.TQTables]:
    """Construct `(TQLayout, TQTables)` for `preset` (with or without prefix)."""
    dtype_name = preset if preset.startswith("turboquant_") else f"turboquant_{preset}"
    cfg = TurboQuantConfig.from_cache_dtype(dtype_name, head_dim)
    layout = tq.TQLayout.from_tq_config(cfg)
    centroids = get_centroids(head_dim, cfg.centroid_bits).numpy()
    return layout, tq.make_tables(layout, centroids)
