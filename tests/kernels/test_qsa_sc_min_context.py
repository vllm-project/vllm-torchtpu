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
"""Context preference retains the normal cost model and SC certificates."""

import hashlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.test_qsa_prefill_tc_packing import inputs
from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc
from vllm_torchtpu.kernels.sparse_gqa_prefill import selector as sel

pytest_plugins = ["tests.kernels.test_qsa_prefill_tc_packing"]
KNOB = "VLLM_TPU_QSA_SC_MIN_CONTEXT"


@pytest.mark.parametrize(
    "lengths,starts,threshold,expected",
    [
        ([0, 8, 9, 0], [0, 4], None, False),
        ([0, 8, 9, 0], [0, 4], 8, True),
        ([0, 8, 9, 0], [0, 4], 9, True),
        ([0, 8, 9, 0], [0, 4], 10, False),
        ([9999, 8, 9, 9999], [1, 3], 10, False),
        ([9999, 8, 9, 9999], [1, 1, 3, 3], 9, True),
        ([9999, 8, 9, 9999], [1, 1], 1, False),
        ([0, -1, 0], [0, 3], 1, False),
        ([], [0, 0], 1, False),
        ([np.iinfo(np.int32).max], [0, 1], 10**80, False),
        ([np.iinfo(np.int32).max], [0, 1], np.iinfo(np.int32).max, True),
    ],
)
def test_effective_causal_max(lengths, starts, threshold, expected):
    actual = sel._context_prefers_sc(
        jnp.array(lengths, jnp.int32), jnp.array(starts, jnp.int32), threshold
    )
    assert bool(actual) is expected


def workload(pattern, length=131072):
    tokens, groups, slots = 256, length // 4, 512
    rng = np.random.default_rng(1964)
    ids = (
        np.broadcast_to(np.arange(groups - slots, groups), (tokens, slots)).copy()
        if pattern == "recent"
        else np.stack(
            [np.sort(rng.choice(groups, slots, replace=False)) for _ in range(tokens)]
        )
    )
    q = jnp.zeros((tokens, 24, 256), jnp.bfloat16)
    return (
        q,
        q,
        jnp.asarray(ids, jnp.int32),
        jnp.full(tokens, length, jnp.int32),
        jnp.zeros((tokens, 158), jnp.int32),
        jnp.zeros((1, 1664, 1, 4, 256), jnp.float8_e4m3fn),
        jnp.array([0, tokens], jnp.int32),
    )


@pytest.mark.parametrize("pattern", ["recent", "uniform"])
@pytest.mark.parametrize("threshold", [None, 131073, 131072, 131071])
def test_actual_default_cost_or_inclusive_context_preference(
    core_calls, monkeypatch, pattern, threshold
):
    xs = workload(pattern)
    calls = []

    def sparse(*a, **kw):
        calls.append(1)
        return jnp.full_like(a[0], 7)

    monkeypatch.setattr(sel.owner_window, "owner_window_attention", sparse)
    with jax.disable_jit():
        out, route = sel.select_prefill(
            *xs, kv_heads=2, ratio=4, sc_min_context=threshold
        )
    expected_sc = pattern == "uniform" or threshold in (131072, 131071)
    assert (int(route) in (2, 3)) is expected_sc
    assert bool(calls) is expected_sc
    if expected_sc:
        assert bool(jnp.all(out == 7))


@pytest.mark.parametrize(
    "kind", ["ordered", "unordered", "duplicate", "negative", "future", "tail"]
)
def test_certification_refinement_cannot_be_vetoed_by_cost(
    core_calls, monkeypatch, kind
):
    xs = list(inputs(256, [0, 256], columns=10))
    length = 8447 if kind == "tail" else 8448
    ids = jnp.arange(length // 4 - 4, length // 4, dtype=jnp.int32)
    if kind in ("unordered", "duplicate"):
        ids = ids[::-1]
    if kind == "duplicate":
        ids = ids.at[1].set(ids[0])
    if kind == "negative":
        ids = ids.at[0].set(-1)
    if kind == "future":
        ids = ids.at[0].set(length // 4)
    xs[2] = jnp.broadcast_to(ids, (256, 4))
    xs[3] = jnp.full(256, length, jnp.int32)
    xs[4] = jnp.zeros_like(xs[4])
    prepared_calls = []
    original = tc._prepare_prefill_metadata

    def prepare(*a, **kw):
        prepared_calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(tc, "_prepare_prefill_metadata", prepare)
    monkeypatch.setattr(
        sel.owner_window,
        "owner_window_attention",
        lambda *a, **kw: jnp.full_like(a[0], 7),
    )
    cheap_tc = sel.CostModel(0, 0, 0, 10000, 0, 0)
    with jax.disable_jit():
        out, route = sel.select_prefill(
            *xs, kv_heads=2, ratio=4, cost_model=cheap_tc, sc_min_context=length
        )
    safe = kind not in ("duplicate", "negative", "future")
    assert (int(route) in (2, 3)) is safe
    assert len(prepared_calls) == (0 if kind in ("ordered", "tail") else 1)
    if safe:
        assert bool(jnp.all(out == 7))
    else:
        assert int(route) == 4 and len(core_calls) == 1


@pytest.mark.parametrize(
    "case", ["nonunit", "dynamic_scale", "resource", "multi", "clipped"]
)
def test_preference_does_not_widen_supported_schedules(core_calls, monkeypatch, case):
    xs = list(inputs(256, [0, 256], columns=10))
    options = dict(k_scale=1.0)
    if case == "nonunit":
        options["k_scale"] = 0.5
    if case == "dynamic_scale":
        options["k_scale"] = jnp.float32(1)
    if case == "resource":
        monkeypatch.setattr(
            sel.owner_window, "is_supported_by_owner_window", lambda *a, **kw: False
        )
    if case == "multi":
        xs[-1] = jnp.array([0, 128, 256], jnp.int32)
    if case == "clipped":
        xs[-1] = jnp.array([8, 248], jnp.int32)
        xs[3] = xs[3].at[:8].set(0).at[248:].set(0)
    with jax.disable_jit():
        _, route = sel.select_prefill(
            *xs, kv_heads=2, ratio=4, sc_min_context=1, **options
        )
    assert int(route) == 4


@pytest.mark.parametrize("text", ["0", "-1", "abc"])
def test_invalid_context_setting(monkeypatch, text):
    monkeypatch.setenv(KNOB, text)
    with pytest.raises(ValueError):
        envs.environment_variables[KNOB]()


def test_registered_identity_schema_and_fresh_getter_under_env_cache(monkeypatch):
    import torch
    from torch_tpu._internal import pallas

    from vllm_torchtpu.compilation.tpu_compiler import _tpu_compile_env_factors
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    monkeypatch.delenv(KNOB, raising=False)
    qsa_op._build_compact_pooled_qsa_op.cache_clear()
    registrations = []
    original = pallas.jax_op

    def register(name, core, **kw):
        registrations.append((name, core, kw))
        return original(name, core, **kw)

    monkeypatch.setattr(pallas, "jax_op", register)
    monkeypatch.setattr(envs, "__getattr__", envs.__getattr__)
    envs.enable_envs_cache()
    baseline = qsa_op.build_compact_pooled_qsa_op()
    for text in ("8448", "16384", str(10**80)):
        monkeypatch.setenv(KNOB, text)
        configured = qsa_op.build_compact_pooled_qsa_op()
        assert configured is qsa_op.build_compact_pooled_qsa_op()
        assert configured is not baseline
        assert envs.VLLM_TPU_QSA_SC_MIN_CONTEXT is None
        assert _tpu_compile_env_factors()[KNOB] == int(text)
    monkeypatch.delenv(KNOB)
    assert qsa_op.build_compact_pooled_qsa_op() is baseline
    assert _tpu_compile_env_factors()[KNOB] is None
    assert len(registrations) == 4
    for (name, core, kwargs), value in zip(registrations, [None, 8448, 16384, 10**80]):
        expected_kind = "compact_pooled_qsa"
        if value is not None:
            expected_kind += (
                "_sc_min_context_" + hashlib.sha256(str(value).encode()).hexdigest()
            )
        assert name == qsa_op._op_name(expected_kind, None, None)
        assert core.keywords["sc_min_context"] == value
        assert kwargs["donate_argnums"] == (10, 11)
        (schema,) = torch._C._jit_get_schemas_for_operator(name)
        assert len(schema.arguments) == 33 and len(schema.returns) == 4
        assert [a.name for a in schema.arguments][10:12] == ["pool", "index_pool"]
        assert schema.arguments[-1].name == "scheduling_group_namespace"
        assert "sc_min_context" not in inspect.signature(baseline).parameters
