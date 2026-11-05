# SPDX-License-Identifier: Apache-2.0
"""CPU support-envelope and fallback dispatch regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import decode_preparation as prep
from vllm_torchtpu.kernels.experimental.push_fused_moe import presort


@pytest.fixture
def tpu_v7(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(prep, "get_tpu_version", lambda: 7)


def inputs(tokens=288, experts=384):
    return [
        jax.ShapeDtypeStruct((tokens * 6,), jnp.int32),
        jax.ShapeDtypeStruct((experts,), jnp.int32),
        jax.ShapeDtypeStruct((tokens, 40, 128), jnp.float8_e4m3fn),
    ]


@pytest.mark.parametrize("tokens", [96, 192, 256, 288, 384])
@pytest.mark.parametrize("experts", [384, 432])
def test_approved_tokens(tpu_v7, tokens, experts):
    assert presort.supports_local_sort(
        *inputs(tokens, experts), e_total=experts, topk=6
    )


@pytest.mark.parametrize(
    "index,shape,dtype",
    [
        (0, (1727,), jnp.int32),
        (0, (288, 6), jnp.int32),
        (0, (1728,), jnp.int16),
        (1, (384,), jnp.int16),
        (1, (383,), jnp.int32),
        (2, (287, 40, 128), jnp.float8_e4m3fn),
        (2, (), jnp.float8_e4m3fn),
    ],
)
def test_rejects_other_contracts(tpu_v7, index, shape, dtype):
    xs = inputs()
    xs[index] = jax.ShapeDtypeStruct(shape, dtype)
    assert not presort.supports_local_sort(*xs, e_total=384, topk=6)


@pytest.mark.parametrize("tokens", [0, 95, 128, 193, 287, 385, 512])
def test_unapproved_tokens_fall_back(tpu_v7, monkeypatch, tokens):
    xs = inputs(tokens)
    assert not presort.supports_local_sort(*xs, e_total=384, topk=6)
    calls = []

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return "native"

    monkeypatch.setattr(presort.host, "local_sort_activations", fallback)
    assert presort.local_sort_activations(*xs, e_total=384, topk=6) == "native"
    assert calls == [(tuple(xs), {"e_total": 384, "topk": 6})]


@pytest.mark.parametrize("backend,version", [("cpu", 7), ("gpu", 7), ("tpu", 6)])
def test_backend_guard(monkeypatch, backend, version):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(prep, "get_tpu_version", lambda: version)
    assert not presort.supports_local_sort(*inputs(), e_total=384, topk=6)


@pytest.mark.parametrize("experts,topk", [(128, 6), (384, 4)])
def test_routing_guard(tpu_v7, experts, topk):
    assert not presort.supports_local_sort(*inputs(), e_total=experts, topk=topk)


@pytest.mark.parametrize(
    "shape,dtype",
    [((288, 40, 128), jnp.bfloat16), ((288, 20, 128), jnp.float8_e5m2)],
)
def test_activation_layout_is_unconstrained(tpu_v7, shape, dtype):
    """Only pairs/counts enter the fused kernel; q keeps the native gather."""
    xs = inputs()
    xs[2] = jax.ShapeDtypeStruct(shape, dtype)
    assert presort.supports_local_sort(*xs, e_total=384, topk=6)


@pytest.mark.parametrize("tokens", [96, 192, 256, 288, 384])
@pytest.mark.parametrize("experts", [384, 432])
@pytest.mark.parametrize("interpret", [True, False])
def test_physical_slot_sort_matches_native(monkeypatch, tokens, experts, interpret):
    if not interpret and jax.default_backend() != "tpu":
        pytest.skip("compiled Pallas requires TPU")
    rng = np.random.default_rng(17)
    slots_per_rank = experts // 16
    active_per_rank = slots_per_rank - (2 if experts == 432 else 0)
    active = np.flatnonzero(np.arange(experts) % slots_per_rank < active_per_rank)
    pairs = np.stack([rng.choice(active, 6, replace=False) for _ in range(tokens)])
    pairs[: tokens * 3 // 4] = active[-6:]
    pairs[-3:] = experts
    pairs = jnp.asarray(pairs.reshape(-1), jnp.int32)
    counts = jnp.bincount(pairs, length=experts)
    q = jnp.arange(tokens, dtype=jnp.int32)[:, None]
    expected = np.argsort(np.asarray(pairs), kind="stable") // 6
    expected[np.arange(tokens * 6) >= int(counts.sum())] = tokens - 1
    native = presort.host.local_sort_activations(
        pairs, counts, q, e_total=experts, topk=6
    )
    calls = []

    def fuse(fn, args, *, name):
        calls.append(name)
        return prep.fuse_function(fn, args, name=name, interpret=interpret)

    monkeypatch.setattr(presort, "fuse_function", fuse)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(prep, "get_tpu_version", lambda: 7)
    actual = presort.local_sort_activations(pairs, counts, q, e_total=experts, topk=6)
    assert calls == ["dsv41_presort_metadata_v2"]
    np.testing.assert_array_equal(actual[0][:, 0], expected)
    for got, want in zip(actual, native):
        np.testing.assert_array_equal(got, want)
