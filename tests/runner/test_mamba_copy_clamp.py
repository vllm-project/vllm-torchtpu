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
"""Host-only tests: host-provided mamba copy indices are clamped into the pool
before the (unchecked) copy kernel uses them.

The module needs torch-tpu, so the helper is extracted with ``ast``.
"""

import ast
import pathlib
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2] / "src/vllm_torchtpu"
COPY_OP = ROOT / "layers/adapter/custom_ops/mamba_state_copy_op.py"


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
