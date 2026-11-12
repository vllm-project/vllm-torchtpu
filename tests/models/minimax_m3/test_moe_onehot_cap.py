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
"""Capped one-hot EP permute (`fused_moe_gmm._onehot_row_cap`) against the
full one, on CPU: the GMM kernel is replaced by a per-row reference and the
module is loaded with its TPU-only imports stubbed."""

import importlib.util
import pathlib
import sys
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SRC = pathlib.Path(__file__).parents[3] / "src/vllm_torchtpu"


def _load():
    def file_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    saved = dict(sys.modules)
    pkg = types.ModuleType("vllm_torchtpu")
    pkg.__path__ = []
    sys.modules["vllm_torchtpu"] = pkg
    envs = file_module("vllm_torchtpu.envs", _SRC / "envs.py")
    pkg.envs = envs
    logger_mod = types.ModuleType("vllm_torchtpu.logger")
    logger_mod.init_logger = lambda name: mock.MagicMock()
    sys.modules["vllm_torchtpu.logger"] = logger_mod
    for name in (
        "vllm_torchtpu.kernels",
        "vllm_torchtpu.kernels.experimental",
        "vllm_torchtpu.kernels.experimental.moe_dense_expert",
        "vllm_torchtpu.kernels.megablox",
        "vllm_torchtpu.kernels.megablox.gmm_v2",
        "vllm_torchtpu.kernels.megablox.moe_onehot_unpermute",
        "vllm_torchtpu.kernels.sparse_core",
        "vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce",
        "vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2",
        "vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper",
        "vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3",
        "vllm_torchtpu.kernels.sparse_core.ragged_gather_v2",
        "vllm_torchtpu.kernels.sparse_core.support",
    ):
        sys.modules[name] = mock.MagicMock()
    try:
        mod = file_module("fused_moe_gmm_under_test", _SRC / "layers/core/fused_moe_gmm.py")
    finally:
        for k in list(sys.modules):
            if k not in saved:
                sys.modules.pop(k)
        sys.modules.update(saved)
    return mod, envs


fmg, envs = _load()


def _gmm_mlp(x, w1, w2, group_sizes, **kw):
    """Per-row reference of `gmm_v2_mlp` (silu-and-mul, then down)."""
    ends = jnp.cumsum(group_sizes)
    rows = jnp.arange(x.shape[0])
    expert = jnp.minimum(jnp.searchsorted(ends, rows, side="right"), w1.shape[0] - 1)
    valid = rows < ends[-1]
    xf = x.astype(jnp.float32)
    h = jnp.einsum("rk,rkn->rn", xf, w1[expert].astype(jnp.float32))
    half = h.shape[1] // 2
    a = jax.nn.silu(h[:, :half]) * h[:, half:]
    out = jnp.einsum("rk,rkn->rn", a, w2[expert].astype(jnp.float32))
    return jnp.where(valid[:, None], out, jnp.nan).astype(x.dtype)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(fmg, "gmm_v2_mlp", _gmm_mlp)
    monkeypatch.setattr(fmg, "_is_fp4_weight", lambda w: True)
    monkeypatch.setattr(fmg, "get_packing_factor", lambda *a: 1)


def _run(monkeypatch, tokens, cap_min_rows, local_bias, seed):
    monkeypatch.setenv("TPU_MOE_ONEHOT_CAP_MIN_ROWS", str(cap_min_rows))
    monkeypatch.setenv("TPU_MOE_ONEHOT_CAP_FRACTION", "0.25")
    rng = np.random.default_rng(seed)
    hidden, inter, experts, local, topk = 64, 32, 128, 16, 4
    start = 32
    x = jnp.asarray(rng.standard_normal((tokens, hidden)), jnp.float32)
    w1 = jnp.asarray(rng.standard_normal((local, hidden, 2 * inter)) * 0.1, jnp.float32)
    w2 = jnp.asarray(rng.standard_normal((local, inter, hidden)) * 0.1, jnp.float32)
    # Distinct experts per token; `local_bias` raises the share of this rank's.
    logits = rng.standard_normal((tokens, experts))
    logits[:, start : start + local] += local_bias
    ids = np.argsort(-logits, axis=1)[:, :topk].astype(np.int32)
    w = rng.random((tokens, topk)).astype(np.float32)
    w[-1] = 0.0  # a padded token
    fn = fmg.fused_moe_func.__wrapped__ if hasattr(fmg.fused_moe_func, "__wrapped__") else fmg.fused_moe_func
    out = fn(
        x, w1, w2, None, None, None, None, jnp.asarray(w), jnp.asarray(ids),
        jnp.asarray(start, jnp.int32), topk=topk, activation="silu", use_ep=True,
        use_sparse_core=True, onehot_moe_permute_threshold=2048, skip_padded_tokens=True,
    )
    local_routes = int(((ids >= start) & (ids < start + local) & (w != 0)).sum())
    return np.asarray(out), local_routes


@pytest.mark.parametrize("tokens", [64, 384, 512])
@pytest.mark.parametrize("local_bias", [0.0, 6.0])
def test_capped_matches_full(monkeypatch, tokens, local_bias):
    full, routes = _run(monkeypatch, tokens, 0, local_bias, seed=tokens)
    seen = []
    prepare = fmg.prepare_routed_gmm_inputs

    def spy(*args, **kwargs):
        seen.append(kwargs.get("row_cap"))
        return prepare(*args, **kwargs)

    monkeypatch.setattr(fmg, "prepare_routed_gmm_inputs", spy)
    capped, _ = _run(monkeypatch, tokens, 16, local_bias, seed=tokens)
    cap = -(-int(tokens * 4 * 0.25) // 8) * 8
    assert sorted(seen, key=str) == sorted([cap, None], key=str)  # both cond branches
    # local_bias 6 overflows the cap: the full branch of the cond runs.
    assert (routes > cap) == (local_bias > 0)
    np.testing.assert_allclose(capped, full, rtol=1e-6, atol=1e-6)
    assert np.isfinite(full).all()


def test_cap_gate():
    assert fmg._onehot_row_cap(1536, True, True, 2048) is None  # env unset -> off
