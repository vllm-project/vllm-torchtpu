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
"""Unit tests for TPU generation parsing and Gen 8 RPA v3 default block sizes."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.collectives import (
    all_gather_matmul_tuned_block_sizes as agmm_blocks,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa_v3_kernel
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import util as rpa_v3_util


@pytest.mark.parametrize(
    ("device_kind", "expected_version"),
    [
        ("TPU v5 lite", 5),
        ("TPU v5e", 5),
        ("TPU v5p", 5),
        ("TPU v6 lite", 6),
        ("TPU v6e", 6),
        ("TPU7x", 7),
        ("TPU v7", 7),
        ("TPU8i", 8),
        ("TPU v8i", 8),
        ("TPU8t", 8),
        ("TPU v8t", 8),
        ("cpu", -1),
        ("NVIDIA H100", -1),
    ],
)
@pytest.mark.parametrize("module", [agmm_blocks, rpa_v3_util])
def test_get_tpu_version(monkeypatch, module, device_kind, expected_version):
    monkeypatch.setattr(
        module.jax,
        "devices",
        lambda: [SimpleNamespace(device_kind=device_kind)],
    )
    assert module.get_tpu_version() == expected_version


@pytest.mark.parametrize(
    "rpa_case",
    [rpa_v3_kernel.RpaCase.DECODE, rpa_v3_kernel.RpaCase.MIXED],
)
def test_rpa_v3_default_block_sizes_supports_gen8(monkeypatch, rpa_case):
    monkeypatch.setattr(rpa_v3_kernel, "get_tpu_version", lambda: 7)
    gen7_blocks = rpa_v3_kernel.get_default_block_sizes(
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        actual_num_q_heads=32,
        actual_num_kv_heads=8,
        head_dim=128,
        page_size=16,
        max_num_tokens=2048,
        max_num_seqs=64,
        pages_per_seq=64,
        case=rpa_case,
    )
    monkeypatch.setattr(rpa_v3_kernel, "get_tpu_version", lambda: 8)
    gen8_blocks = rpa_v3_kernel.get_default_block_sizes(
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        actual_num_q_heads=32,
        actual_num_kv_heads=8,
        head_dim=128,
        page_size=16,
        max_num_tokens=2048,
        max_num_seqs=64,
        pages_per_seq=64,
        case=rpa_case,
    )
    assert gen8_blocks == gen7_blocks
