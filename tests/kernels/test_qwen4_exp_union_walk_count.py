# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_UNION_COUNT: the host-side D2 engagement count ("eligible steps").

A step counts when union_walk_prefill was traced for the step's row count (the static dispatch held) and the step's
longest row reaches the min context recorded at trace time. Tracing (make_jaxpr) is enough to record; nothing runs.
"""

import functools

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud

PAGE, RATIO = 3200, 4


@pytest.fixture
def v7x(monkeypatch):
    if jax.default_backend() != "tpu":
        from jax._src import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
    monkeypatch.setattr(ud, "_WALK_TRACED", {})
    monkeypatch.setattr(ud, "_ENGAGEMENT", {"steps": 0, "eligible": 0})
    yield


def _trace(tokens, min_context):
    args = (
        jax.ShapeDtypeStruct((tokens, 24, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 24, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 512), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens, 2), jnp.int32),
        jax.ShapeDtypeStruct((4, PAGE, 1, 4, 256), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((3,), jnp.int32),
    )
    f = functools.partial(ud.union_walk_prefill, kv_heads=2, ratio=RATIO, group=128, kv_tile=1024,
                          union_cap=12288, min_context=min_context, lean=True)
    jax.make_jaxpr(lambda *a: f(*a)[0])(*args)


def test_eligible_steps_follow_trace_and_min_context(v7x, monkeypatch):
    logged = []
    monkeypatch.setattr(ud, "_engagement_log", lambda msg, *a: logged.append(a))
    _trace(1024, 32768)
    assert ud._WALK_TRACED == {1024: 32768}
    steps = [(1024, 40000), (1024, 1000), (2048, 90000), (1024, 32768)]
    got = [ud.count_union_walk_step(t, longest, every=2) for t, longest in steps]
    # traced bucket + long enough; too short; a bucket the walk was never traced for; exactly the threshold
    assert got == [True, False, False, True]
    assert ud._ENGAGEMENT == {"steps": 4, "eligible": 2}
    assert logged == [(1, 2), (2, 4)]  # every 2 steps: (eligible, steps)


def test_min_context_zero_counts_every_traced_step(v7x):
    _trace(1024, 0)
    assert all(ud.count_union_walk_step(1024, longest, every=0) for longest in (1, 500, 100000))
    assert not ud.count_union_walk_step(4096, 100000, every=0)
