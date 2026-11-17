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
"""Shared TurboQuant layout and table builders for unit and TPU kernel tests."""

from __future__ import annotations

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)

from vllm_torchtpu.layers.core import turboquant as tq


def make_tq_layout_and_tables(
    preset: str, head_dim: int
) -> tuple[tq.TQLayout, tq.TQTables]:
    full = preset if preset.startswith("turboquant_") else f"turboquant_{preset}"
    cfg = TurboQuantConfig.from_cache_dtype(full, head_dim)
    layout = tq.TQLayout.from_tq_config(cfg)
    tables = tq.make_tables(layout, get_centroids(head_dim, cfg.centroid_bits).numpy())
    return layout, tables
