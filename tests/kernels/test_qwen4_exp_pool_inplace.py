# SPDX-License-Identifier: Apache-2.0
"""QSA must thread the unified pool through its step without a conditional.

A ``lax.cond``/``case`` that returns the pool makes XLA allocate a second
pool-sized buffer for the conditional's result, so a compiled forward needs
twice the pool in HBM (the CompileTimeHbmOom on every mixed/prefill bucket).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.mesh import AbstractDevice, AbstractMesh

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.request_types import decode_prefix

_OPTIONS = dict(kv_heads=1, ratio=4, k_scale=1.0, v_scale=1.0)


def _fake_rpa(q, gate, blocks, lengths, table, cache, **_):
    """Distinct, input-dependent output; cache threaded through unchanged."""
    out = (
        q * 3 + gate + lengths[:, None, None] + table[:, :1, None] + blocks[:, :1, None]
    )
    return jnp.where((lengths > 0)[:, None, None], out, 0), cache


def _fake_prefill(q, gate, blocks, lengths, table, cache, starts, **_):
    out = q * 5 - gate + lengths[:, None, None] * 7 + starts[1] + table[:, :1, None]
    out = out + blocks[:, :1, None]
    return jnp.where((lengths > 0)[:, None, None], out, 0), cache


def _reference(query, gate, blocks, lengths, table, cache, starts):
    """The routing qsa_attention had before, with its lax.cond on the cache."""
    count = decode_prefix(starts)
    capacity = len(starts) - 1

    def split(_):
        decode_lengths = jnp.where(jnp.arange(capacity) < count, lengths[:capacity], 0)
        decoded, owner = _fake_rpa(
            query[:capacity],
            gate[:capacity],
            blocks[:capacity],
            decode_lengths,
            table[:capacity],
            cache,
        )
        rows = jnp.arange(len(query))
        ids = jnp.minimum(rows + count, len(query) - 1)
        prefill_lengths = jnp.where(rows < starts[-1] - count, lengths[ids], 0)
        prefilled, owner = _fake_prefill(
            query[ids],
            gate[ids],
            blocks[ids],
            prefill_lengths,
            table[ids],
            owner,
            jnp.maximum(starts - count, 0),
        )
        output = jnp.where(
            (rows < count)[:, None, None],
            decoded[jnp.minimum(rows, capacity - 1)],
            prefilled[jnp.maximum(rows - count, 0)],
        )
        return jnp.where((lengths > 0)[:, None, None], output, 0), owner

    def prefill(_):
        return _fake_prefill(query, gate, blocks, lengths, table, cache, starts)

    return jax.lax.cond((count > 0) & (count < starts[-1]), split, prefill, None)


@pytest.mark.parametrize(
    "starts",
    [
        [0, 1, 2, 6, 6],  # decode prefix, then a prefill
        [0, 3, 4, 5, 6],  # prefill first: no decode prefix
        [0, 2, 5, 7, 8],  # prefill only
        [0, 1, 2, 3, 4],  # every request decodes, bucket wider than capacity
        [0, 0, 0, 0, 0],  # empty step
        [0, 1, 1, 1, 1],  # one decode, idle rows
    ],
)
def test_masked_routing_matches_conditional_routing(monkeypatch, starts):
    monkeypatch.setattr(qsa_rpa, "rpa_attention", _fake_rpa)
    monkeypatch.setattr(qsa_rpa, "_prefill_attention", _fake_prefill)
    rng = np.random.default_rng(sum(starts))
    tokens, capacity = 8, 4
    starts = jnp.asarray(starts, jnp.int32)
    query = jnp.asarray(rng.normal(size=(tokens, 2, 16)), jnp.float32)
    gate = jnp.asarray(rng.normal(size=(tokens, 2, 16)), jnp.float32)
    blocks = jnp.asarray(rng.integers(-1, 9, size=(tokens, 3)), jnp.int32)
    valid = jnp.arange(tokens) < starts[-1]
    lengths = jnp.where(valid, jnp.asarray(rng.integers(1, 99, tokens)), 0)
    table = jnp.asarray(rng.integers(0, 50, size=(tokens, 2)), jnp.int32)
    cache = jnp.asarray(rng.normal(size=(5, 4)), jnp.float32)
    assert len(starts) - 1 == capacity

    run = jax.jit(qsa_rpa.qsa_attention, static_argnames=tuple(_OPTIONS))
    output, owner = run(query, gate, blocks, lengths, table, cache, starts, **_OPTIONS)
    want_output, _ = jax.jit(_reference)(
        query, gate, blocks, lengths, table, cache, starts
    )
    np.testing.assert_array_equal(np.asarray(output), np.asarray(want_output))
    np.testing.assert_array_equal(np.asarray(owner), np.asarray(cache))


@pytest.mark.parametrize("max_query_len, want_rpa", [(1, False), (2, True), (4, True)])
def test_mtp_verify_steps_stay_on_rpa(monkeypatch, max_query_len, want_rpa):
    """8 verify tokens over 4 request rows (2 per request) is a decode step
    under MTP: every token must run on RPA as its own row, not on prefill."""
    monkeypatch.setattr(qsa_rpa, "rpa_attention", _fake_rpa)
    monkeypatch.setattr(qsa_rpa, "_prefill_attention", _fake_prefill)
    rng = np.random.default_rng(7)
    tokens = 8
    starts = jnp.asarray([0, 2, 4, 6, 8], jnp.int32)
    query = jnp.asarray(rng.normal(size=(tokens, 2, 16)), jnp.float32)
    gate = jnp.asarray(rng.normal(size=(tokens, 2, 16)), jnp.float32)
    blocks = jnp.asarray(rng.integers(-1, 9, size=(tokens, 3)), jnp.int32)
    lengths = jnp.asarray(rng.integers(1, 99, tokens), jnp.int32)
    table = jnp.asarray(rng.integers(0, 50, size=(tokens, 2)), jnp.int32)
    cache = jnp.asarray(rng.normal(size=(5, 4)), jnp.float32)

    static = tuple(_OPTIONS) + ("max_query_len",)
    run = jax.jit(qsa_rpa.qsa_attention, static_argnames=static)
    output, _ = run(
        query, gate, blocks, lengths, table, cache, starts,
        max_query_len=max_query_len, **_OPTIONS,
    )
    rpa, _ = _fake_rpa(query, gate, blocks, lengths, table, cache)
    assert np.allclose(np.asarray(output), np.asarray(rpa), rtol=1e-5) == want_rpa


def _pool_results_of_control_flow(module, pool_type):
    """Names of case/if ops in `module` with a result typed like the pool."""
    hits = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    name = inner.operation.name
                    if name in ("stablehlo.case", "stablehlo.if"):
                        types = [str(r.type) for r in inner.operation.results]
                        if pool_type in types:
                            hits.append(name)
                    walk(inner.operation)

    walk(module.operation)
    return hits


def _lower_pooled_step(tokens, kv_heads=2, q_heads=24, **extra):
    """Lower the real pooled QSA step for TPU (no device needed)."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import pooled_qsa_step

    requests = 3
    rng = np.random.default_rng(tokens)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.3, shape), jnp.bfloat16)

    pool = jnp.zeros((12, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    args = (
        rand((tokens, q_heads * 256 * 2)),
        rand((tokens, kv_heads * 256)),
        rand((tokens, kv_heads * 256)),
        rand((tokens, 5 * 128)),
        rand((256,)),
        rand((256,)),
        rand((128,)),
        rand((128,)),
        jnp.zeros((tokens,), jnp.int32),
        rand((4096, 128)),
        pool,
        jnp.array([0, 1, 3, tokens - 1], jnp.int32),
        jnp.full((requests,), 100, jnp.int32),
        jnp.zeros((requests * 32,), jnp.int32),
        jnp.ones((requests, 2), jnp.int32),
        jnp.full((requests, 2), 2, jnp.int32),
    )
    options = dict(
        compress_ratio=4,
        token_topk=2048,
        norm_eps=1e-6,
        rotary_dim=64,
        k_scale=0.375,
        v_scale=2.5,
        **extra,
    )
    step = jax.jit(
        pooled_qsa_step, donate_argnums=(10,), static_argnames=tuple(options)
    )
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        return step.trace(*args, **options).lower(lowering_platforms=("tpu",))


@pytest.mark.parametrize("tokens", [8, 16])
@pytest.mark.parametrize("slot", [None, 7])
def test_pooled_step_threads_pool_without_conditional(tokens, slot):
    """A mixed bucket, wider than the request count: no conditional returns
    the pool, including for a packed compressed-key slot."""
    extra = {} if slot is None else {"compressed_slot": slot}
    module = _lower_pooled_step(tokens, **extra).compiler_ir("stablehlo")
    pool_type = "tensor<12x1664x1x4x256xf8E4M3FN>"
    main = next(
        op
        for op in module.body.operations
        if op.operation.name == "func.func"
        and str(op.operation.attributes["sym_name"]) == '"main"'
    )
    # Sanity: the pool is a program result, so the check below sees its type.
    assert pool_type in str(main.operation.attributes["function_type"])
    assert _pool_results_of_control_flow(module, pool_type) == []


def test_unpacked_pooled_step_lowers_identically_with_slot_zero():
    """Packing support must leave the unpacked program bit-identical."""
    assert (
        _lower_pooled_step(16).as_text()
        == _lower_pooled_step(16, compressed_slot=0).as_text()
    )


@pytest.mark.parametrize("tokens", [8, 128])
@pytest.mark.parametrize("main_slot", [0, 1])
def test_paired_main_kv_step_threads_pool_without_conditional(tokens, main_slot):
    """Paired main KV slots (gather/scatter and tiled writers, head-offset
    attention) thread the pool through without a conditional result.

    The lowered geometry has one KV head per device, as at TP8."""
    module = _lower_pooled_step(
        tokens, kv_heads=1, q_heads=3, main_slot=main_slot
    ).compiler_ir("stablehlo")
    pool_type = "tensor<12x1664x1x4x256xf8E4M3FN>"
    assert _pool_results_of_control_flow(module, pool_type) == []


def test_unpaired_pooled_step_lowers_identically_with_default_main_slot():
    """Paired main KV support must leave the unpaired program bit-identical."""
    assert (
        _lower_pooled_step(16).as_text()
        == _lower_pooled_step(16, main_slot=-1).as_text()
    )
