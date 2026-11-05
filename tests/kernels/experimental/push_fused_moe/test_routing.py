# SPDX-License-Identifier: Apache-2.0
"""CPU checks for exact routing dispatch eligibility."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import decode_preparation as prep
from vllm_torchtpu.kernels.experimental.push_fused_moe import routing
from vllm_torchtpu.kernels.experimental.push_fused_moe.host import run_tables


def supported(tokens=288, **changes):
    experts = changes.pop("e_total", 384)
    i32 = jnp.zeros((1,), jnp.int32)
    gathered = SimpleNamespace(
        topk=6, n_tokens=tokens * 16, local_blocks=i32, block_hist=i32
    )
    values = dict(
        fuse_metadata=True,
        t_local=tokens,
        gathered=gathered,
        e_total=experts,
        ep=16,
        tile_m=128,
        block=routing.host.routing_block(tokens, 6),
        shard_stride=routing.host.ragged_stride_bound(tokens * 16, 6, experts, 128),
        runs=(i32,) * 4,
        me=i32,
    )
    for key in ("topk", "n_tokens", "local_blocks", "block_hist"):
        if key in changes:
            setattr(gathered, key, changes.pop(key))
    values.update(changes)
    try:
        routing.prepare_routing(**values)
    except DispatchReached as exc:
        return str(exc) == "fused"
    raise AssertionError("routing did not reach either dispatch path")


class DispatchReached(Exception):
    pass


def stop(path):
    def reached(*args, **kwargs):
        raise DispatchReached(path)

    return reached


@pytest.fixture
def tpu_v7(monkeypatch):
    monkeypatch.setattr(routing, "build_routing_tables_sharded", lambda *a, **k: None)
    monkeypatch.setattr(routing.host, "local_slab_rows", lambda *a, **k: None)
    monkeypatch.setattr(
        routing.host, "shard_transport_tables_in_blocks", stop("fallback")
    )
    monkeypatch.setattr(routing, "fuse_function", stop("fused"))
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(prep, "get_tpu_version", lambda: 7)


@pytest.mark.parametrize("tokens", [96, 192, 256, 288, 384])
@pytest.mark.parametrize("experts", [384, 432])
def test_exact_approved_tokens(tpu_v7, tokens, experts):
    assert supported(tokens, e_total=experts)
    stride = routing.host.ragged_stride_bound(tokens * 16, 6, experts, 128)
    assert not supported(tokens, e_total=experts, shard_stride=stride + 128)
    assert not supported(
        tokens, e_total=experts, block=routing.host.routing_block(tokens, 6) + 128
    )


@pytest.mark.parametrize(
    "tokens", [1, 95, 97, 128, 191, 193, 255, 257, 287, 289, 383, 385, 512]
)
def test_unapproved_tokens(tpu_v7, tokens):
    assert not supported(tokens)


@pytest.mark.parametrize(
    "changes",
    [
        {"fuse_metadata": False},
        {"topk": 4},
        {"e_total": 383},
        {"ep": 8},
        {"tile_m": 256},
        {"n_tokens": 288 * 16 + 1},
    ],
)
def test_contract_guards(tpu_v7, changes):
    assert not supported(**changes)


@pytest.mark.parametrize("field", ["me", "local_blocks", "block_hist", "runs"])
def test_dtype_guards(tpu_v7, field):
    bad = jax.ShapeDtypeStruct((1,), jnp.int16)
    assert not supported(**{field: (bad,) * 4 if field == "runs" else bad})


@pytest.mark.parametrize("backend,version", [("cpu", 7), ("gpu", 7), ("tpu", 6)])
def test_backend_guards(tpu_v7, monkeypatch, backend, version):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(prep, "get_tpu_version", lambda: version)
    assert not supported()


@pytest.mark.parametrize("tokens", [96, 192, 256, 288, 384])
@pytest.mark.parametrize("experts", [384, 432])
@pytest.mark.parametrize("interpret", [True, False])
def test_physical_slot_tables_match_native(monkeypatch, tokens, experts, interpret):
    if not interpret and jax.default_backend() != "tpu":
        pytest.skip("compiled Pallas requires TPU")
    ep, topk, tile = 16, 6, 128
    rng = np.random.default_rng(23)
    slots_per_rank = experts // ep
    active_per_rank = slots_per_rank - (2 if experts == 432 else 0)
    slots = np.arange(experts)
    local_slot = slots % slots_per_rank + slots // slots_per_rank % 2 * 5
    active = np.flatnonzero(local_slot % slots_per_rank < active_per_rank)
    pairs = np.stack(
        [rng.choice(active, topk, replace=False) for _ in range(tokens * ep)]
    ).astype(np.int32)
    pairs[::tokens] = experts
    pairs[-tokens:] = experts
    block = routing.host.routing_block(tokens, topk)
    stride = routing.host.ragged_stride_bound(tokens * ep, topk, experts, tile)
    hist = jnp.asarray(
        np.stack(
            [
                np.bincount(p.reshape(-1), minlength=experts + 1)[:experts]
                for p in pairs.reshape(ep, tokens, topk)
            ]
        ),
        jnp.int32,
    )
    runs = run_tables(hist)
    calls = []

    def fuse(fn, args, *, name):
        calls.append(name)
        return prep.fuse_function(fn, args, name=name, interpret=interpret)

    monkeypatch.setattr(routing, "fuse_function", fuse)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(prep, "get_tpu_version", lambda: 7)
    for rank in range(ep):
        local = jnp.asarray(pairs[rank * tokens : (rank + 1) * tokens]).reshape(
            -1, block
        )
        gathered = routing.host.GatheredPairs(
            jnp.asarray(pairs).reshape(-1, block),
            local,
            routing.host.pair_block_hist(local, experts),
            hist,
            tokens * ep,
            topk,
        )
        kwargs = dict(
            e_total=experts,
            ep=ep,
            t_local=tokens,
            block=block,
            tile_m=tile,
            shard_stride=stride,
        )
        me = jnp.asarray(rank, jnp.int32)
        native = routing.prepare_routing(
            me, gathered, runs, **kwargs, fuse_metadata=False
        )
        actual = routing.prepare_routing(me, gathered, runs, **kwargs)
        for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(native)):
            np.testing.assert_array_equal(got, want)
    assert calls == ["dsv41_routing_position_metadata"] * ep
