# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The decode megakernel's per-rank weights.

``presharded`` is a rank cache: ``rankNN/<family>.bin`` raw arrays in the decode
fast profile's per-rank layout (``kernels.layout.Layout.weight_specs``: TP32,
expert TP4 block64 W4A8 experts, packed shared expert) with ``rankNN/index.json``
(shape and dtype per family) and the checkpoint's ``source-config.json``. Each
worker reads its kernel rank's files; every tensor is created once on the
device in its final dtype (FP8 as FP8).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import ml_dtypes  # noqa: F401  (registers the bfloat16 / float8 numpy dtypes)
import numpy as np
import torch

from vllm_torchtpu.megakernel import bridge
from vllm_torchtpu.megakernel.qwen38.kernels import Config
from vllm_torchtpu.megakernel.qwen38.kernels.layout import Layout

FORMAT = "qwen38-megakernel-rank-cache-v2"


def decode_layout(presharded: Path, page: int) -> Layout:
    """The decode fast profile of the cache's model, on ``page``-token KV pages."""
    manifest = json.loads((Path(presharded) / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{presharded} is not a {FORMAT} Qwen3.8 decode rank cache")
    config = Config.from_checkpoint(
        json.loads((Path(presharded) / "source-config.json").read_text())
    )
    return Layout.for_config(
        config,
        tp=32,
        expert_tp=4,
        context_parallel=8,
        precision="torchtpu_w4a8_b64",
        expert_reduction="residual",
        dense_output_tile=8192,
        kv_page_size=page,
    )


def _iter_rank(presharded: Path, rank: int, names: Iterable[str]):
    directory = Path(presharded) / f"rank{rank:02d}"
    index = json.loads((directory / "index.json").read_text())
    for name in names:
        entry = index[name]
        array = np.fromfile(directory / f"{name}.bin", dtype=np.dtype(entry["dtype"]))
        yield None, name, array.reshape(entry["shape"])


def load(
    presharded: Path, rank: int, layout: Layout
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """``({family: [1, ...] device tensor}, load stats)`` of kernel rank ``rank``."""
    weights: dict[str, torch.Tensor] = {}

    def store(_, name, tensor):
        weights[name] = tensor

    started = time.perf_counter()
    stats = bridge.stream_to_device(
        _iter_rank(presharded, rank, layout.weight_names), store
    )
    stats["wall"] = time.perf_counter() - started
    return weights, stats
