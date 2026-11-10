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
"""Host-only test: the runner's read-offset scatters clamp host-provided
indices into the read-offset buffer before the (unchecked) scatter uses them.

The runner module needs torch-tpu, so the helpers are extracted with ``ast``.
"""

import ast
import pathlib
import textwrap

import pytest

RUNNER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/vllm_torchtpu/runner/tpu_runner.py"
)


def _extract(path, names):
    src = path.read_text()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = textwrap.dedent(ast.get_source_segment(src, node))
    assert set(out) == set(names), set(names) - set(out)
    return out


def test_read_offset_scatters_clamp_indices():
    torch = pytest.importorskip("torch")
    names = [
        "_clamp_offsets_index",
        "_rollback_offsets_seed_impl",
        "_rollback_offsets_migrate_impl",
        "_spec_seed_and_rollback_impl",
    ]
    g = {"torch": torch}
    for code in _extract(RUNNER, names).values():
        exec(code, g)  # noqa: S102
    offs = torch.arange(8, dtype=torch.int32) + 100
    bad = torch.tensor([3, 1 << 20, -5], dtype=torch.long)
    g["_rollback_offsets_seed_impl"](offs, bad)
    assert offs.tolist() == [0, 101, 102, 0, 104, 105, 106, 0]
    offs = torch.arange(8, dtype=torch.int32) + 100
    g["_rollback_offsets_migrate_impl"](
        offs,
        torch.tensor([1, 99], dtype=torch.long),
        torch.tensor([2, -3], dtype=torch.long),
    )
    assert offs.tolist() == [107, 101, 101, 103, 104, 105, 106, 107]
