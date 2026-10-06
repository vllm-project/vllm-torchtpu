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


def _quantized_dot(v, w):
    """FP8 activations scaled per row and 512-column block, times w."""
    block = min(v.shape[1], 512)
    out = 0
    for k in range(0, v.shape[1], block):
        b = v[:, k : k + block]
        scale = jnp.max(jnp.abs(b), axis=1, keepdims=True) / 448.0
        q = (b * jnp.where(scale == 0, 0, 1 / scale)).astype(FP8)
        out = out + scale * jnp.dot(
            q, w[k : k + block], preferred_element_type=jnp.float32
        )
    return out


def _reference(x, ids, weights, w13, s13, w2, s2):
    """Plain JAX with the kernel's arithmetic: blockwise FP8 activations."""
    e, _, two_inter = w13.shape
    inter = two_inter // 2
    x = x.astype(jnp.float32)
    s13 = s13.reshape(e, two_inter)
    s2 = s2.reshape(e, -1)
    out = jnp.zeros((x.shape[0], x.shape[1]), jnp.float32)
    for expert in range(e):
        w = jnp.sum(jnp.where(ids == expert, weights, 0).astype(jnp.float32), axis=1)
        h = _quantized_dot(x, w13[expert]) * s13[expert][None, :]
        a = jax.nn.silu(h[:, :inter]) * h[:, inter:]
        o = _quantized_dot(a, w2[expert])
        out = out + w[:, None] * (o * s2[expert][None, :])
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


def test_interpreted_activation_scales_are_blockwise(interpret):
    # A 1e6 outlier in the first 512 columns, on weight rows that are zero.
    # One scale per row would flush the second block's values (~1) below
    # FP8's smallest subnormal and zero the output; 512-column scales keep it.
    x, ids, weights, w13, s13, w2, s2 = _ids_operands(11, 8, 8, 1024, 256, 4)
    x = x.at[:, 0].set(1e6)
    w13 = w13.at[:, :512].set(0)
    ops = (x, ids, weights, w13, s13, w2, s2)
    want = _oracle(x, ids, weights, w13, s13, w2, s2, 0)
    assert np.linalg.norm(want) > 0
    assert _rel_l2(_ids_call(*ops), want) < 0.06


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
    # bounded at 1.5x. GMM and lowtok both scale per 512-column block but
    # round and accumulate differently, so each is compared with the oracle.
    assert err < 0.06, (err, gmm_err)
    assert err < gmm_err + 0.005, (err, gmm_err)
