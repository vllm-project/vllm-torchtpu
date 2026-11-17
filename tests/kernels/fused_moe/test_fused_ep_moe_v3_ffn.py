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
"""Fused EP MoE v3's FFN kernel build (``ffn_kernel.ffn_kernel``) on CPU.

The build reads ``layout.SHARD_INDEX_OPERAND``: in the torch bridge's mode
the shard's index is one more scalar-prefetch operand. The memoized build
must key on it, or a process that builds in one mode and calls in the other
gets a program with the wrong operand list. The kernel's arithmetic runs on
TPU in ``test_fused_ep_moe_v3.py``.
"""

import importlib

import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import layout

pytestmark = pytest.mark.cpu_test

ffn_kernel = importlib.import_module("vllm_torchtpu.kernels.fused_moe.v3.ffn_kernel")

# The build test_fused_ep_moe_v3.py's small world takes: 16 experts over 8
# shards, 24 rows per shard at top-k 4, hidden 1024, intermediate 128.
BUILD = dict(
    experts_per_shard=2,
    tile_rows=32,
    hidden=1024,
    inter=128,
    num_shards=8,
    weight_format="fp8",
    routed_rows=928,
    result_rows="bf16",
    sorted_rows=True,
)


@pytest.fixture
def tpu7x(monkeypatch):
    """A TPU7x chip record (the build sizes its VMEM from it) and an empty
    build cache."""

    def record():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    monkeypatch.setattr(pltpu, "get_tpu_info", record)
    monkeypatch.setattr(tpu_info, "get_tpu_info", record)
    monkeypatch.setattr(ffn_kernel, "_BUILD_CACHE", {})


def test_a_build_is_reused_within_its_operand_mode_only(tpu7x, monkeypatch):
    monkeypatch.setattr(layout, "SHARD_INDEX_OPERAND", False)
    own_mode = ffn_kernel.ffn_kernel(**BUILD)
    assert ffn_kernel.ffn_kernel(**BUILD) is own_mode
    monkeypatch.setattr(layout, "SHARD_INDEX_OPERAND", True)
    operand_mode = ffn_kernel.ffn_kernel(**BUILD)
    assert operand_mode is not own_mode
    assert ffn_kernel.ffn_kernel(**BUILD) is operand_mode
    monkeypatch.setattr(layout, "SHARD_INDEX_OPERAND", False)
    assert ffn_kernel.ffn_kernel(**BUILD) is own_mode
