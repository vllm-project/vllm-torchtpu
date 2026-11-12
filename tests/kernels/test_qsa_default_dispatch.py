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
"""Normal default dispatch requires no external profile or configuration."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.test_qsa_prefill_tc_packing import inputs
from tests.kernels.test_qsa_selector import profile_for
from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
from vllm_torchtpu.kernels.sparse_gqa_prefill import selector

pytest_plugins = ["tests.kernels.test_qsa_prefill_tc_packing"]


def test_default_builder_is_used_and_sc_reachable(core_calls, monkeypatch):
    xs = inputs(256, [0, 256], columns=10)
    calls = []

    def model(*a, **kw):
        calls.append((a, kw))
        return profile_for(xs, sc_fixed_ms=0.0)

    monkeypatch.setattr(selector, "build", model)
    monkeypatch.setattr(
        selector.owner_window,
        "owner_window_attention",
        lambda *a, **kw: jnp.full_like(a[0], 7),
    )
    with jax.disable_jit():
        y, route = selector.select_prefill(*xs, kv_heads=2, ratio=4)
    assert int(route) in (2, 3) and bool(jnp.all(y == 7)) and len(calls) == 1


def test_public_entry_uses_normal_selector_without_parameters(monkeypatch):
    xs = inputs(256, [0, 256])
    seen = []

    def choose(*a, **kw):
        seen.append(kw)
        return a[0], jnp.int32(0)

    monkeypatch.setattr(selector, "select_prefill", choose)
    y, owner = qsa_rpa._prefill_attention(*xs, kv_heads=2, ratio=4)
    assert y is xs[0] and owner is xs[5]
    assert seen[0]["cost_model"] is None
    assert "cost_model" not in inspect.signature(qsa_rpa.qsa_attention).parameters


def test_real_default_registration_schema_cache_and_no_profile(monkeypatch):
    import torch

    from vllm_torchtpu.compilation.tpu_compiler import _tpu_compile_env_factors
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    assert "VLLM_TPU_QSA_COST_PROFILE" not in envs.environment_variables
    assert "VLLM_TPU_QSA_COST_PROFILE" not in _tpu_compile_env_factors()
    qsa_op._build_compact_pooled_qsa_op.cache_clear()
    op = qsa_op.build_compact_pooled_qsa_op()
    assert op is qsa_op.build_compact_pooled_qsa_op()
    name = qsa_op._op_name("compact_pooled_qsa", None, None)
    (schema,) = torch._C._jit_get_schemas_for_operator(name)
    assert len(schema.arguments) == 35 and len(schema.returns) == 5
    assert [a.name for a in schema.arguments] == list(inspect.signature(op).parameters)
    assert [a.name for a in schema.arguments][10:13] == [
        "pool",
        "index_pool",
        "ring_pool",
    ]
    assert [a.name for a in schema.arguments][-2:] == [
        "scheduling_group_namespace",
        "ring_capacity",
    ]
    registered = inspect.getclosurevars(op).nonlocals["op"]
    assert tuple(registered._init_fn.donate_argnums) == (10, 11, 12)
    assert (
        "cost_model"
        not in inspect.signature(qsa_op._build_compact_pooled_qsa_op).parameters
    )


@pytest.mark.parametrize("pattern,expected_sc", [("recent", False), ("uniform", True)])
def test_actual_default_work_builder_selects_reuse_and_dispersed(
    core_calls, monkeypatch, pattern, expected_sc
):
    tokens, heads, dim, groups, slots = 256, 24, 256, 32768, 512
    rng = np.random.default_rng(42)
    blocks = (
        np.broadcast_to(
            np.arange(groups - slots, groups, dtype=np.int32), (tokens, slots)
        ).copy()
        if pattern == "recent"
        else np.stack(
            [np.sort(rng.choice(groups, slots, replace=False)) for _ in range(tokens)]
        ).astype(np.int32)
    )
    q = jnp.zeros((tokens, heads, dim), jnp.bfloat16)
    xs = (
        q,
        q,
        jnp.asarray(blocks),
        jnp.full(tokens, groups * 4, jnp.int32),
        jnp.zeros((tokens, 158), jnp.int32),
        jnp.zeros((1, 1664, 1, 4, dim), jnp.float8_e4m3fn),
        jnp.array([0, tokens], jnp.int32),
    )
    called = []

    def sparse(*a, **kw):
        called.append(1)
        return jnp.full_like(a[0], 7)

    monkeypatch.setattr(selector.owner_window, "owner_window_attention", sparse)
    with jax.disable_jit():
        out, route = selector.select_prefill(*xs, kv_heads=2, ratio=4)
    assert (int(route) in (2, 3)) == expected_sc and bool(called) == expected_sc
    if expected_sc:
        assert bool(jnp.all(out == 7))
