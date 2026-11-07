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
"""TPU numerics parity for in-graph RoPE against CPU-built reference tables."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers import rotary_embedding as rope_lib

from vllm_torchtpu import rope_patch
from vllm_torchtpu.layers.adapter import rotary_embedding as tpu_rope

pytestmark = pytest.mark.tpu_test

_CHUNK_SIZE = 32768


@pytest.mark.parametrize(
    ("name", "head_size", "max_pos", "rope_params"),
    [
        (
            "standard",
            128,
            262144,
            {"rope_type": "default", "rope_theta": 10000.0},
        ),
        (
            "proportional",
            256,
            262144,
            {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        ),
        (
            "llama3",
            128,
            262144,
            {
                "rope_type": "llama3",
                "rope_theta": 500000.0,
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        ),
    ],
    ids=["standard", "proportional", "llama3"],
)
def test_in_graph_cos_sin_matches_cpu_reference_table_on_tpu(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    head_size: int,
    max_pos: int,
    rope_params: dict[str, Any],
) -> None:
    import torch_tpu  # noqa: F401

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    dev = torch.device("tpu")

    with set_current_vllm_config(config):
        # Build the CPU reference table single-threaded: on 112-thread hosts,
        # multi-threaded `einsum` + `cos`/`sin` inside `_compute_cos_sin_cache`
        # intermittently perturbs one OpenMP worker slice by ~1.5e-4, flipping
        # >1e-4 of bf16 rounding ties independently of the TPU path under test.
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.device("cpu"):
                ref = rope_lib.get_rope(
                    head_size,
                    max_position=max_pos,
                    is_neox_style=True,
                    rope_parameters=dict(rope_params),
                    dtype=torch.float32,
                )
        finally:
            torch.set_num_threads(previous_threads)
        ref_table = ref.cos_sin_cache.clone()

        rope_lib._ROPE_DICT.clear()
        with dev:
            rope = rope_lib.get_rope(
                head_size,
                max_position=max_pos,
                is_neox_style=True,
                rope_parameters=dict(rope_params),
                dtype=torch.bfloat16,
            )

    assert type(rope).__module__ == tpu_rope.__name__
    assert rope_patch.patch_in_graph_rotary_embedding(rope) == 1

    inv_freq: tuple[float, ...] = rope._in_graph_rope_inv_freq
    cache_dtype: torch.dtype = rope._in_graph_rope_cache_dtype

    @torch.compile(backend="tpu", dynamic=False)
    def eval_chunk(pos: torch.Tensor) -> torch.Tensor:
        return rope_patch._in_graph_cos_sin(pos, inv_freq, cache_dtype, torch.bfloat16)

    outs = [
        eval_chunk(
            torch.arange(s, s + _CHUNK_SIZE, dtype=torch.int32, device=dev)
        ).cpu()
        for s in range(0, max_pos, _CHUNK_SIZE)
    ]
    ingraph_bf16 = torch.cat(outs, dim=0)

    mismatch = float((ingraph_bf16 != ref_table.to(torch.bfloat16)).float().mean())
    max_abs = float((ingraph_bf16.float() - ref_table).abs().max())
    assert mismatch < 1e-4, f"{name}: bf16 mismatch {mismatch:.6g} >= 1e-4"
    assert max_abs <= 2**-8, f"{name}: max abs diff {max_abs:.6g} > 2**-8"
