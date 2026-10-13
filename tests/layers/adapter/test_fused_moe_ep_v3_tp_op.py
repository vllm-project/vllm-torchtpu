# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The fused_moe_ep_v3_tp op's arithmetic on an eight-device CPU mesh
(XLA_FLAGS=--xla_force_host_platform_device_count=8): tensor-parallel attention
over the EP group, every rank holding all tokens.

Two kernels in the op run only on a TPU, so stand-ins take their place: v3's
fused_ep_moe is v3's own reference (kernels/fused_moe/v3/reference.py: its
quantization steps, dense over the experts), and fused_moe_func (the grouped
matmul of the production EP path) is a dense f32 loop over this rank's experts.
Production's router select runs in interpret mode. Everything else is the op:
routing, padding, the token split and restore, the expert-id relabelling into
mesh order, the shared expert, the routed scale and the all-reduces.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.kernels import router_topk
from vllm_torchtpu.kernels.fused_moe.v3 import Config
from vllm_torchtpu.kernels.fused_moe.v3 import reference as v3_reference
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3_tp as tp

pytestmark = [pytest.mark.cpu_test, pytest.mark.multichip]

R, L, D, F, FS, K, N = 8, 2, 256, 128, 32, 4, 64  # ranks, local experts, ...
E = R * L
SCALE = 2.5
FP8 = jnp.float8_e4m3fn


def _requires_eight_devices():
    if len(jax.devices()) != R:
        pytest.skip("requires eight devices; CPU virtual devices are sufficient")


def _dense_fused_moe_func(x, w13, w2, s13, s2, b1, b2, w, ids, start, **kw):
    """fused_moe_func's EP result in f32: this rank's experts only."""
    xs = x.astype(jnp.float32)
    out = jnp.zeros((x.shape[0], w2.shape[2]), jnp.float32)
    for j in range(w13.shape[0]):
        weight = jnp.sum(
            jnp.where(ids == start + j, w.astype(jnp.float32), 0.0), axis=1
        )
        h = xs @ (w13[j].astype(jnp.float32) * s13[j].reshape(1, -1))
        act = jax.nn.silu(h[:, :F]) * h[:, F:]
        out = out + weight[:, None] * (
            act @ (w2[j].astype(jnp.float32) * s2[j].reshape(1, -1))
        )
    return out.astype(x.dtype)


def _reference_fused_ep_moe(x, w1, w2, s1, s2, logits, *, top_k, routing, config, **kw):
    return v3_reference.fused_ep_moe_reference(
        x,
        w1,
        w2,
        s1,
        s2,
        None,
        top_k=top_k,
        renormalize=True,
        weight_format="fp8",
        routing=routing,
        activation_block=config.activation_block,
    )


@pytest.fixture
def stand_ins(monkeypatch):
    monkeypatch.setattr(tp, "fused_moe_func", _dense_fused_moe_func)
    monkeypatch.setattr(tp, "fused_ep_moe", _reference_fused_ep_moe)
    monkeypatch.setattr(
        tp, "router_select", functools.partial(router_topk.select, interpret=True)
    )


def _model(seed=0):
    """One layer in EP-rank order: routed experts [E, ...], the shared expert
    split into R TP slices, router scores and bias."""
    rng = np.random.default_rng(seed)

    def fp8(shape, scale):
        return jnp.asarray(rng.standard_normal(shape) * scale, jnp.float32).astype(FP8)

    return dict(
        x=jnp.asarray(rng.standard_normal((N, D)), jnp.float32).astype(jnp.bfloat16),
        scores=jax.nn.sigmoid(jnp.asarray(rng.standard_normal((N, E)), jnp.float32)),
        bias=jnp.asarray(rng.uniform(-0.05, 0.05, E), jnp.float32),
        w13=fp8((E, D, 2 * F), 1 / np.sqrt(D)),
        w2=fp8((E, F, D), 1 / np.sqrt(F)),
        s13=jnp.asarray(rng.uniform(0.5, 1.5, (E, 1, 1, 2 * F)), jnp.float32),
        s2=jnp.asarray(rng.uniform(0.5, 1.5, (E, 1, 1, D)), jnp.float32),
        sw13=fp8((R, D, 2 * FS), 1 / np.sqrt(D)),
        ss13=jnp.asarray(rng.uniform(0.5, 1.5, (R, 2 * FS)), jnp.float32),
        sw2=fp8((R, FS, D), 1 / np.sqrt(FS * R)),
        ss2=jnp.asarray(rng.uniform(0.5, 1.5, (R, D)), jnp.float32),
    )


def _expected(m, n_valid):
    """The layer in f32 numpy: production routing, every expert dense, the
    shared expert summed over its slices, shared + SCALE * routed."""
    f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    x, scores, bias = f32(m["x"]), f32(m["scores"]), f32(m["bias"])
    ids = np.argsort(-(scores + bias), axis=1, kind="stable")[:, :K]
    w = np.take_along_axis(scores, ids, axis=1)
    w = w / w.sum(axis=1, keepdims=True)
    w[n_valid:] = 0
    routed = np.zeros((N, D), np.float32)
    for e in range(E):
        we = np.where(ids == e, w, 0).sum(axis=1)
        h = x @ (f32(m["w13"][e]) * f32(m["s13"][e]).reshape(1, -1))
        act = h[:, :F] / (1 + np.exp(-h[:, :F])) * h[:, F:]
        routed += we[:, None] * (
            act @ (f32(m["w2"][e]) * f32(m["s2"][e]).reshape(1, -1))
        )
    shared = np.zeros((N, D), np.float32)
    for r in range(R):
        h = x @ (f32(m["sw13"][r]) * f32(m["ss13"][r]))
        act = h[:, :FS] / (1 + np.exp(-h[:, :FS])) * h[:, FS:]
        shared += act @ (f32(m["sw2"][r]) * f32(m["ss2"][r]))
    return shared + SCALE * routed


IDENTITY = tuple(range(R))


def _run(m, *, min_tokens, n_valid=N, rank_order=IDENTITY):
    """The op's function on mesh-stacked operands: mesh index p holds EP rank
    rank_order[p]'s experts and every rank holds every token."""
    mesh = Mesh(np.asarray(jax.devices()), ("d",))
    topo = tp._Topology(
        mesh=mesh, mesh_index=0, rank_order=rank_order, groups=(tuple(range(R)),)
    )
    statics = tp._OpStatics(
        topk=K,
        routed_scale=SCALE,
        activation="silu",
        use_sparse_core=False,
        onehot_threshold=0,
        local_experts=L,
        min_tokens=min_tokens,
        config=Config(),
    )
    fn, _ = tp._build_fn(topo, statics)
    block = np.concatenate([np.arange(r * L, (r + 1) * L) for r in rank_order])
    pad = np.zeros(2 * N, bool)
    pad[n_valid:N] = True
    out = fn(
        jnp.tile(m["x"], (R, 1)),
        jnp.tile(m["scores"], (R, 1)),
        m["bias"],
        jnp.asarray(pad),
        m["w13"][block],
        m["w2"][block],
        m["s13"][block],
        m["s2"][block],
        m["sw13"].reshape(R * D, 2 * FS),
        m["ss13"].reshape(R * 2 * FS),
        m["sw2"].reshape(R * FS, D),
        m["ss2"].reshape(R * D),
        jnp.asarray([[r * L] for r in rank_order], jnp.int32),
        jnp.arange(R, dtype=jnp.int32).reshape(R, 1),
    )
    return np.asarray(out, np.float32).reshape(R, N, D)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize(
    "min_tokens", [N + 1, N], ids=["production_ep_path", "v3_path"]
)
def test_both_paths_match_the_layer_on_every_rank(stand_ins, min_tokens):
    _requires_eight_devices()
    m = _model()
    out = _run(m, min_tokens=min_tokens)
    assert all(np.array_equal(out[r], out[0]) for r in range(R))  # replicated output
    want = _expected(m, N)
    # Both paths round the shared expert's activations to fp8 (production's
    # xla_quantized_matmul) and sum in bf16 (0.025 here); v3 adds its fp8 token,
    # intermediate and result rows.
    assert _rel(out[0], want) < (0.04 if min_tokens > N else 0.06)


def test_v3_path_relabels_router_ids_into_mesh_order(stand_ins):
    _requires_eight_devices()
    m = _model(1)
    order = (2, 0, 3, 1, 6, 4, 7, 5)
    np.testing.assert_array_equal(
        _run(m, min_tokens=N, rank_order=order)[0], _run(m, min_tokens=N)[0]
    )


@pytest.mark.parametrize(
    "min_tokens", [N + 1, N], ids=["production_ep_path", "v3_path"]
)
def test_padded_rows_leave_valid_rows_unchanged(stand_ins, min_tokens):
    _requires_eight_devices()
    m = _model(2)
    full = _run(m, min_tokens=min_tokens)[0]
    padded = _run(m, min_tokens=min_tokens, n_valid=N - 10)[0]
    np.testing.assert_array_equal(padded[: N - 10], full[: N - 10])
    assert np.isfinite(padded).all()
