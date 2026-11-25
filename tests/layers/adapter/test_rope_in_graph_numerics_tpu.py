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
    ],
    ids=["standard", "proportional"],
)
def test_patched_forward_matches_cpu_reference_table_on_tpu(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    head_size: int,
    max_pos: int,
    rope_params: dict[str, Any],
) -> None:
    """Compiled in-graph cos/sin reproduce the CPU-built bf16 table on TPU.

    Covers both standard RoPE (where device `pow` ~8-ulp drift flips several
    percent of bf16 entries by 262,144 positions) and zero-padded proportional
    RoPE (Gemma-4, where recomputing `1 / base ** (2i / d)` on device would
    overwrite the zero frequencies).
    """
    import torch_tpu  # noqa: F401

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    # Multi-threaded CPU `cos`/`sin` can perturb one thread's slice of the
    # reference table by more than the threshold below.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with set_current_vllm_config(config), torch.device("cpu"):
            rope = rope_lib.get_rope(
                head_size,
                max_position=max_pos,
                is_neox_style=True,
                rope_parameters=dict(rope_params),
                dtype=torch.bfloat16,
            )
            # `_maybe_compact_neox_rope_cache` narrows `rope.cos_sin_cache` on
            # proportional NeoX; compare against the full CPU-built table.
            ref_table = rope._compute_cos_sin_cache().to(torch.bfloat16)
    finally:
        torch.set_num_threads(previous_threads)
    assert ref_table.shape == (max_pos, head_size)
    assert rope_patch.patch_in_graph_rotary_embedding(rope) == 1

    dev = torch.device("tpu")
    # A NeoX rotation of `[1, ..., 1 | 0, ..., 0]` returns `[cos | sin]`.
    half = head_size // 2
    probe = torch.cat(
        (torch.ones(_CHUNK_SIZE, half), torch.zeros(_CHUNK_SIZE, half)), dim=-1
    )
    probe = probe.to(torch.bfloat16).to(dev)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def cos_sin_rows(positions: torch.Tensor) -> torch.Tensor:
        return rope(positions, probe)[0]

    rows = torch.cat(
        [
            cos_sin_rows(
                torch.arange(start, start + _CHUNK_SIZE, dtype=torch.int32, device=dev)
            ).cpu()
            for start in range(0, max_pos, _CHUNK_SIZE)
        ]
    )
    mismatch = float((rows != ref_table).float().mean())
    assert mismatch < 1e-4, f"{name}: bf16 mismatch {mismatch:.6g} >= 1e-4"
