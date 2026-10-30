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
"""Host-only tests: host-provided mamba copy / read-offset indices are clamped
into their buffer before the (unchecked) copy kernel or scatter uses them.

The modules need torch-tpu, so the helpers are extracted with ``ast``.
"""

import ast
import pathlib
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2] / "src/vllm_torchtpu"
COPY_OP = ROOT / "layers/adapter/custom_ops/mamba_state_copy_op.py"
RUNNER = ROOT / "runner/tpu_runner.py"


def _extract(path, names):
    src = path.read_text()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = textwrap.dedent(ast.get_source_segment(src, node))
    assert set(out) == set(names), set(names) - set(out)
    return out


def test_copy_indices_clamped_into_pool():
    jnp = pytest.importorskip("jax.numpy")
    g = {"jax": pytest.importorskip("jax"), "jnp": jnp}
    exec(_extract(COPY_OP, ["_clamp_rows"])["_clamp_rows"], g)  # noqa: S102
    idx = jnp.array([0, 5, 9, -1, 2**30, -(2**31)], dtype=jnp.int32)
    out = g["_clamp_rows"](idx, 10)
    assert out.dtype == idx.dtype
    assert out.tolist() == [0, 5, 9, 0, 9, 0]


def test_both_copy_programs_clamp_src_and_dst():
    src = COPY_OP.read_text()
    for fn in ("def _copy_fn(", "def _rows_fn("):
        body = src[src.index(fn) : src.index("copy_blocks(", src.index(fn))]
        assert "src = _clamp_rows(src, pool.shape[0])" in body, fn
        assert "dst = _clamp_rows(dst, pool.shape[0])" in body, fn


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
