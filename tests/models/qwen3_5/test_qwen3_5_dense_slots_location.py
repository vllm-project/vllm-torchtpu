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
"""The dense-slots layer lives with the Qwen3.5 kernels: its source is part of
the runtime compile-cache key, and the shared fused EP v3 adapter imports it
only on the path the Qwen3.5 layout table turns on."""

import ast
from pathlib import Path

import pytest

import vllm_torchtpu
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3

pytestmark = pytest.mark.cpu_test

PACKAGE = Path(vllm_torchtpu.__file__).resolve().parent


def test_runtime_cache_key_covers_the_dense_slots_layer():
    """It runs behind a custom-op boundary like the shared kernels, so a change
    to it must invalidate cached executables."""
    repo_root = PACKAGE.parent
    hashed = {
        p.relative_to(repo_root).as_posix()
        for p in tpu_compiler._iter_runtime_cache_key_files(repo_root)
    }
    assert "vllm_torchtpu/models/qwen3_5/kernels/dense_slots.py" in hashed


def test_the_shared_adapter_does_not_import_it_at_module_level():
    tree = ast.parse(Path(fused_moe_ep_v3.__file__).read_text())
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "qwen3_5" in ast.unparse(node)
    ]
    assert not top_level


def test_the_layer_has_its_own_barrier_group():
    """It runs in one decode program with the small all-reduce and
    reduce-scatter and with fused EP v3's kernels; a kernel that shared its
    collective id would share its barrier semaphore."""
    from vllm_torchtpu.kernels.experimental.small_collectives import (
        all_reduce,
        common,
        reduce_scatter,
    )
    from vllm_torchtpu.kernels.fused_moe.v3 import layout
    from vllm_torchtpu.models.qwen3_5.kernels import dense_slots

    assert dense_slots.COLLECTIVE_ID_DENSE == common.DENSE_SLOTS_COLLECTIVE_ID
    ids = (
        dense_slots.COLLECTIVE_ID_DENSE,
        all_reduce.COLLECTIVE_ID,
        reduce_scatter.COLLECTIVE_ID,
        layout.COLLECTIVE_ID_FFN,
        layout.COLLECTIVE_ID_TRANSPORT,
    )
    assert len(set(ids)) == len(ids), ids
