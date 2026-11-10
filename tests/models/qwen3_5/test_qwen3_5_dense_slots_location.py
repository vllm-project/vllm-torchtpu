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


def test_the_v3_package_no_longer_has_it():
    assert not (PACKAGE / "kernels/fused_moe/v3/dense_slots.py").exists()
