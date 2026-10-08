# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Parity, routing edge cases and admission for the small-batch MoE kernel.

TPU tests compile the kernel; the interpreted tests run with JAX_PLATFORMS=cpu
and check indexing and arithmetic, not TPU resource capacity.
"""

import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.moe_lowtok import kernel
from vllm_torchtpu.layers.core import fused_moe_gmm as wrapper

FP8 = jnp.float8_e4m3fn


def _needs(backend):
    if jax.default_backend() != backend:
        pytest.skip(f"requires {backend}")


@pytest.fixture
def interpret(monkeypatch):
    _needs("cpu")
    monkeypatch.setattr(
        kernel.pl,
        "pallas_call",
        functools.partial(kernel.pl.pallas_call, interpret=True),
    )
    monkeypatch.setattr(
        kernel.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(vmem_capacity_bytes=64 * 1024**2),
    )
    kernel.moe_lowtok.clear_cache()
    yield
    kernel.moe_lowtok.clear_cache()


def _quantize_rows(v):
    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / 448.0
    return (v / scale).astype(FP8), scale


def _reference(x, ids, weights, w13, s13, w2, s2):
    """Plain JAX with the kernel's arithmetic: per-row FP8 activations."""
    e, _, two_inter = w13.shape
    inter = two_inter // 2
    x_q, x_s = _quantize_rows(x.astype(jnp.float32))
    s13 = s13.reshape(e, two_inter)
    s2 = s2.reshape(e, -1)
    out = jnp.zeros((x.shape[0], x.shape[1]), jnp.float32)
    for expert in range(e):
        w = jnp.sum(jnp.where(ids == expert, weights, 0).astype(jnp.float32), axis=1)
        h = jnp.dot(x_q, w13[expert], preferred_element_type=jnp.float32)
        h = h * x_s * s13[expert][None, :]
        a = jax.nn.silu(h[:, :inter]) * h[:, inter:]
        a_q, a_s = _quantize_rows(a)
        o = jnp.dot(a_q, w2[expert], preferred_element_type=jnp.float32)
        out = out + w[:, None] * (o * a_s * s2[expert][None, :])
    return out.astype(x.dtype)


def _router(scores, topk):
    """The served router: top-k, ties to the lowest id, renormalized, bf16."""
    weights, ids = jax.lax.top_k(scores, topk)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    return weights.astype(jnp.bfloat16), ids.astype(jnp.int32)


def _weights(rng, experts, hidden, inter):
    w13 = np.clip(rng.standard_normal((experts, hidden, 2 * inter)) * 40, -448, 448)
    w2 = np.clip(rng.standard_normal((experts, inter, hidden)) * 40, -448, 448)
    s13 = rng.random((experts, 1, 1, 2 * inter)) / (40 * np.sqrt(hidden))
    s2 = rng.random((experts, 1, 1, hidden)) / (40 * np.sqrt(inter))
    return (
        jnp.asarray(w13, FP8),
        jnp.asarray(s13, jnp.float32),
        jnp.asarray(w2, FP8),
        jnp.asarray(s2, jnp.float32),
    )


def _ids_operands(seed, tokens, experts, hidden, inter, topk):
    """About half of the routed pairs land on this shard's experts."""
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((tokens, hidden)), jnp.bfloat16)
    ids = np.stack(
        [rng.choice(2 * experts, size=topk, replace=False) for _ in range(tokens)]
    )
    local = jnp.asarray(np.where(ids < experts, ids, -1), jnp.int32)
    weights = rng.random((tokens, topk))
    weights = jnp.asarray(weights / weights.sum(axis=1, keepdims=True), jnp.bfloat16)
    return (x, local, weights, *_weights(rng, experts, hidden, inter))


def _scores_operands(seed, tokens, experts, hidden, inter, start, n_global=512):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((tokens, hidden)), jnp.bfloat16)
    logits = rng.standard_normal((tokens, n_global)) * 2
    logits[:, start : start + experts] += 5.0
    scores = jax.nn.softmax(jnp.asarray(logits, jnp.float32), axis=-1)
    return x, scores, _weights(rng, experts, hidden, inter)


def _ids_call(x, ids, weights, w13, s13, w2, s2):
    route = kernel.routing_table(ids, weights, w13.shape[0])
    return kernel.moe_lowtok(x, route, w13, s13, w2, s2)


def _scores_call(x, scores, start, weights, topk):
    return kernel.moe_lowtok(x, scores, *weights, jnp.int32(start), topk=topk)


def _scores_want(x, scores, weights, topk, start, experts):
    w13, s13, w2, s2 = weights
    w, ids = _router(scores, topk)
    local = jnp.where((ids >= start) & (ids < start + experts), ids - start, -1)
    return _ids_call(x, local, w, w13, s13, w2, s2)


def _assert_close(got, want, rel_l2=1e-2):
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    assert got.shape == want.shape
    assert np.isfinite(got).all()
    # One activation near an FP8 rounding boundary moves an element by a few
    # percent of the peak, so the per-element bound is loose and L2 is tight.
    scale = np.max(np.abs(want)) + 1e-6
    np.testing.assert_allclose(got / scale, want / scale, atol=6e-2, rtol=0)
    err = np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-9)
    assert err < rel_l2, err


@pytest.mark.parametrize("tokens", [1, 4, 8, 13, 32, 64, 128, 256])
@pytest.mark.parametrize("shape", [(8, 512, 256, 4), (64, 4096, 1024, 10)])
def test_ids_entry_matches_reference(tokens, shape):
    _needs("tpu")
    experts, hidden, inter, topk = shape
    ops = _ids_operands(tokens * 7 + experts, tokens, experts, hidden, inter, topk)
    got = jax.block_until_ready(_ids_call(*ops))
    _assert_close(got, _reference(*ops), 2.5e-2 if tokens < 4 else 1e-2)


def test_no_local_expert_is_zero():
    _needs("tpu")
    x, ids, weights, *w = _ids_operands(3, 4, 8, 512, 256, 4)
    got = _ids_call(x, jnp.full_like(ids, -1), weights, *w)
    assert np.all(np.asarray(got, np.float32) == 0)


@pytest.mark.parametrize("tokens", [1, 4, 8, 13, 32])
@pytest.mark.parametrize("shape", [(8, 512, 256, 4, 16), (64, 4096, 1024, 10, 128)])
def test_scores_entry_matches_router_then_ids_entry(tokens, shape):
    """Same experts and the same renormalized bf16 weights as the router."""
    _needs("tpu")
    experts, hidden, inter, topk, start = shape
    x, scores, weights = _scores_operands(
        tokens * 11 + experts, tokens, experts, hidden, inter, start
    )
    got = _scores_call(x, scores, start, weights, topk)
    _assert_close(got, _scores_want(x, scores, weights, topk, start, experts))


def test_scores_entry_zero_score_rows_select_nothing():
    """A padded row's scores are zeroed; it adds no hit and outputs zero."""
    _needs("tpu")
    experts, hidden, inter, topk, start = 8, 512, 256, 4, 16
    x, scores, weights = _scores_operands(23, 5, experts, hidden, inter, start)
    scores = scores.at[2].set(0.0)
    got = np.asarray(
        _scores_call(x, scores, start, weights, topk),
        np.float32,
    )
    assert np.all(got[2] == 0)
    # A copy: np.asarray of a device array is a read-only view.
    want = np.array(_scores_want(x, scores, weights, topk, start, experts))
    want[2] = 0
    _assert_close(np.delete(got, 2, 0), np.delete(want, 2, 0))


@pytest.mark.parametrize("tokens", [5, 33])
def test_interpreted_ids_entry_matches_reference(interpret, tokens):
    ops = _ids_operands(tokens, tokens, 8, 512, 256, 4)
    _assert_close(_ids_call(*ops), _reference(*ops), 2.5e-2)


@pytest.mark.parametrize("n_global", [512, 160])
def test_interpreted_scores_entry_matches_router_then_reference(interpret, n_global):
    experts, hidden, inter, topk, start = 8, 512, 256, 4, 16
    x, scores, weights = _scores_operands(9, 6, experts, hidden, inter, start, n_global)
    scores = scores.at[2].set(0.0)
    got = np.asarray(
        _scores_call(x, scores, start, weights, topk),
        np.float32,
    )
    assert np.all(got[2] == 0)
    w, ids = _router(scores, topk)
    local = jnp.where((ids >= start) & (ids < start + experts), ids - start, -1)
    local = local.at[2].set(-1)
    _assert_close(got, _reference(x, local, w, *weights), 2.5e-2)


def _shapes(m, h, i, e):
    return (
        jax.ShapeDtypeStruct((m, h), jnp.bfloat16),
        jax.ShapeDtypeStruct((e, h, 2 * i), FP8),
        jax.ShapeDtypeStruct((e, i, h), FP8),
        jax.ShapeDtypeStruct((e, 1, 1, 2 * i), jnp.float32),
        jax.ShapeDtypeStruct((e, 1, 1, h), jnp.float32),
    )


def test_qwen_shape_is_supported():
    assert kernel.can_use_moe_lowtok(
        *_shapes(256, 4096, 1024, 64),
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
    )


@pytest.mark.parametrize(
    "unsupported",
    ["experts", "inter", "activation", "bias", "scale", "bf16", "packed", "padded"],
)
def test_unsupported_layouts_are_refused(unsupported):
    x, w1, w2, s1, s2 = _shapes(16, 512, 256, 8)
    b1, activation, rhs_quant_dtype = None, "silu", None
    if unsupported == "experts":
        x, w1, w2, s1, s2 = _shapes(16, 512, 256, 129)
    elif unsupported == "inter":
        x, w1, w2, s1, s2 = _shapes(16, 512, 128, 8)
    elif unsupported == "activation":
        activation = "gelu"
    elif unsupported == "bias":
        b1 = jax.ShapeDtypeStruct((8, 1, 512), jnp.float32)
    elif unsupported == "scale":
        s1 = jax.ShapeDtypeStruct((8, 4, 1, 512), jnp.float32)
    elif unsupported == "bf16":
        w1 = jax.ShapeDtypeStruct(w1.shape, jnp.bfloat16)
    elif unsupported == "packed":
        rhs_quant_dtype = jnp.int4
    else:
        x = jax.ShapeDtypeStruct((16, 640), jnp.bfloat16)
    assert not kernel.can_use_moe_lowtok(
        x,
        w1,
        w2,
        s1,
        s2,
        b1,
        None,
        activation=activation,
        rhs_quant_dtype=rhs_quant_dtype,
    )


@pytest.mark.parametrize(
    "m, enabled, use_ep, selected",
    [
        (1, True, True, True),
        (128, True, True, True),
        (129, True, True, False),
        (128, False, True, False),
        (128, True, False, False),
    ],
)
def test_token_cap_dispatch(m, enabled, use_ep, selected, monkeypatch):
    """Rows count every token of the call, draft tokens included."""

    class LowtokSelected(RuntimeError):
        pass

    class StandardSelected(RuntimeError):
        pass

    def lowtok_selected(*args, **kwargs):
        raise LowtokSelected

    def standard_selected(*args, **kwargs):
        raise StandardSelected

    monkeypatch.setattr(wrapper, "_MOE_LOWTOK", enabled)
    monkeypatch.setattr(wrapper, "moe_lowtok", lowtok_selected)
    monkeypatch.setattr(wrapper, "prepare_routed_gmm_inputs", standard_selected)
    wrapper.fused_moe_func.clear_cache()
    args = (
        *_shapes(m, 512, 256, 8),
        None,
        None,
        jax.ShapeDtypeStruct((m, 4), jnp.bfloat16),
        jax.ShapeDtypeStruct((m, 4), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    try:
        with pytest.raises(LowtokSelected if selected else StandardSelected):
            jax.eval_shape(
                functools.partial(wrapper.fused_moe_func, topk=4, use_ep=use_ep),
                *args,
            )
    finally:
        wrapper.fused_moe_func.clear_cache()


def _oracle(x, ids, weights, w13, s13, w2, s2, experts_start):
    """Float64 routed MoE over the dequantized FP8 weights."""
    x = np.asarray(x, np.float64)
    w13 = np.asarray(w13, np.float64) * np.asarray(s13, np.float64)[:, 0]
    w2 = np.asarray(w2, np.float64) * np.asarray(s2, np.float64)[:, 0]
    local = np.asarray(ids) - experts_start
    weights = np.asarray(weights, np.float64)
    inter = w2.shape[1]
    out = np.zeros(x.shape)
    for expert in range(w13.shape[0]):
        w = np.sum(np.where(local == expert, weights, 0), axis=1)
        h = x @ w13[expert]
        a = h[:, :inter] / (1 + np.exp(-h[:, :inter])) * h[:, inter:]
        out += w[:, None] * (a @ w2[expert])
    return out


def _rel_l2(got, want):
    got = np.asarray(got, np.float64)
    assert np.isfinite(got).all()
    return np.linalg.norm(got - want) / np.linalg.norm(want)


@pytest.mark.parametrize("m", [16, 128])
def test_fused_moe_func_selects_kernel_and_matches_oracle(monkeypatch, m):
    _needs("tpu")
    rng = np.random.default_rng(m)
    e, h, i, topk, start = 8, 512, 256, 4, 64
    x = jnp.asarray(rng.standard_normal((m, h)), jnp.bfloat16)
    w13, s13, w2, s2 = _weights(rng, e, h, i)
    # Distinct ids over 16 global experts, half of them on this shard.
    ids = np.stack([rng.choice(16, size=topk, replace=False) for _ in range(m)])
    ids = jnp.asarray(ids + start - 4, jnp.int32)
    weights = rng.random((m, topk))
    weights = jnp.asarray(weights / weights.sum(axis=1, keepdims=True), jnp.bfloat16)
    args = (x, w13, w2, s13, s2, None, None, weights, ids)
    kwargs = dict(experts_start=jnp.array(start, jnp.int32), topk=topk, use_ep=True)
    want = _oracle(x, ids, weights, w13, s13, w2, s2, start)
    monkeypatch.setattr(wrapper, "_MOE_LOWTOK", False)
    wrapper.fused_moe_func.clear_cache()
    gmm_err = _rel_l2(wrapper.fused_moe_func(*args, **kwargs), want)
    monkeypatch.setattr(wrapper, "_MOE_LOWTOK", True)
    wrapper.fused_moe_func.clear_cache()
    try:
        assert "moe_lowtok" in wrapper.fused_moe_func.lower(*args, **kwargs).as_text()
        err = _rel_l2(wrapper.fused_moe_func(*args, **kwargs), want)
    finally:
        wrapper.fused_moe_func.clear_cache()
    # An E4M3 operand has rel-L2 error ~2^-3 / sqrt(12 * 7/3) = 0.024; x's enters
    # both silu(h1) and h3, then `a` is requantized: ~sqrt(3) * 0.024 = 0.041,
    # bounded at 1.5x. GMM's 512-column blocks and lowtok's whole rows round
    # differently, so each is compared with the oracle, not with the other.
    assert err < 0.06, (err, gmm_err)
    assert err < gmm_err + 0.005, (err, gmm_err)


@pytest.mark.parametrize("tokens", [4, 16])
@pytest.mark.parametrize("extra_slots", [4, 8])
def test_scores_copies_entry_matches_reference_across_ranks(tokens, extra_slots):
    """With hot-expert copy slots (e.g. 4 or 8 copies/rank = 32 or 64 total copies),
    every hit expert is executed on exactly one of its holders (primary or copy)
    and the sum across all 8 ranks matches the 512-expert reference."""
    _needs("tpu")
    rng = np.random.default_rng(tokens * 100 + extra_slots)
    ranks, owned, hidden, inter, topk = 8, 64, 512, 256, 10
    n_global = ranks * owned
    slots = owned + extra_slots
    x = jnp.asarray(rng.standard_normal((tokens, hidden)), jnp.bfloat16)
    logits = rng.standard_normal((tokens, n_global)).astype(np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    scores = jnp.asarray(exp / exp.sum(axis=1, keepdims=True), jnp.float32)
    w13_all, s13_all, w2_all, s2_all = _weights(rng, n_global, hidden, inter)

    prim = np.array([g // owned for g in range(n_global)], dtype=np.int32)
    copy = np.full(n_global, -1, dtype=np.int32)
    order = []
    cores = [list(range(r * owned, (r + 1) * owned)) for r in range(ranks)]
    for r in range(ranks):
        for j in range(extra_slots):
            g = ((r + 1) % ranks) * owned + j
            copy[g] = r
            cores[r].append(g)
            order.append(g)

    hot, hprim, hcopy = (
        np.asarray(t) for t in kernel.hot_tables(prim, copy, order)
    )
    hot1h = kernel.hot_onehot(hot)
    prim_j = jnp.asarray(prim, jnp.int32)
    copy_j = jnp.asarray(copy, jnp.int32)

    total_out = jnp.zeros((tokens, hidden), jnp.float32)
    for r in range(ranks):
        gid = np.asarray(cores[r], dtype=np.int32)
        tab = jnp.asarray(
            kernel.pack_tables(
                r * owned,
                gid,
                hot,
                hprim,
                hcopy,
                np.asarray(kernel.slot_table(gid, slots)),
            ),
            jnp.int32,
        )
        part = kernel.moe_lowtok(
            x,
            scores,
            w13_all[gid],
            s13_all[gid],
            w2_all[gid],
            s2_all[gid],
            jnp.int32(r * owned),
            tab,
            prim_j,
            copy_j,
            hot1h,
            topk=topk,
            ranks=ranks,
            owned=owned,
            has_copies=True,
        )
        total_out = total_out + part.astype(jnp.float32)

    w, ids = _router(scores, topk)
    want = _reference(x, ids, w, w13_all, s13_all, w2_all, s2_all)
    _assert_close(total_out.astype(jnp.bfloat16), want, 2.5e-2)

