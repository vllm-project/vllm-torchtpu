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
"""One-hot permute/combine against the SparseCore path at a decode batch."""

import functools
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.megablox.gmm_v2 import get_packing_factor
from vllm_torchtpu.layers.core import fused_moe_gmm
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    moe_gmm,
    prepare_routed_gmm_inputs,
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
)


@pytest.mark.parametrize("num_tokens", [96, 192])
def test_onehot_combine_matches_sparse_core_path(num_tokens):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    hidden, inter, topk, num_experts, local = 8192, 2048, 10, 512, 64
    experts_start = 2 * local
    kx, kl, k1, k2 = jax.random.split(jax.random.key(num_tokens), 4)
    x = (jax.random.normal(kx, (num_tokens, hidden)) / 4).astype(jnp.bfloat16)
    probs = jax.nn.softmax(jax.random.normal(kl, (num_tokens, num_experts)), axis=-1)
    weights, ids = jax.lax.top_k(probs, topk)
    w1, s1 = quantize_to_native_fp4_kmajor(
        jax.random.normal(k1, (local, hidden, 2 * inter)) / 8, 256, pack=False
    )
    w2, s2 = quantize_to_native_fp4_kmajor(
        jax.random.normal(k2, (local, inter, hidden)) / 8, 256, pack=False
    )

    def run(threshold):
        return np.asarray(
            jax.jit(
                lambda *a: fused_moe_func(
                    *a,
                    experts_start=jnp.int32(experts_start),
                    topk=topk,
                    activation="silu",
                    use_ep=True,
                    use_sparse_core=True,
                    onehot_moe_permute_threshold=threshold,
                )
            )(x, w1, w2, s1, s2, None, None, weights.astype(jnp.bfloat16), ids),
            np.float32,
        )

    sparse_core = run(0)
    onehot = run(num_tokens * topk)
    assert np.abs(sparse_core).max() > 0
    ulp = np.spacing(np.abs(sparse_core).astype(jnp.bfloat16).astype(np.float32))
    ulp = np.maximum(ulp, np.float32(2.0**-126))
    worst = float((np.abs(onehot - sparse_core) / ulp).max())
    differ = float((onehot != sparse_core).mean())
    print(
        f"tokens={num_tokens} max diff {worst:.2f} bf16 ulp, "
        f"{differ:.4%} elements differ"
    )
    assert worst <= 2.0


HIDDEN, INTER, TOPK, NUM_EXPERTS, LOCAL = 8192, 2048, 10, 512, 64
DP_AXIS, TP_AXIS = "dp", "tp"


def _require_tpu_devices(count):
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < count:
        pytest.skip(f"needs {count} TPU devices")
    return devices[:count]


def _previous_fused_moe_ep(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    experts_start,
    *,
    topk,
    threshold,
    skip_padded_tokens=False,
):
    """``fused_moe_func`` for EP on SparseCore TPUs without
    ``max_local_routes``: argsort permute, one-hot or SparseCore token
    movement in ``prepare_routed_gmm_inputs`` and ``moe_gmm``."""
    num_tokens, hidden_size = hidden_states.shape
    padded_hidden_size = w1.shape[1] * get_packing_factor(w1.dtype, None)
    topk_weights = topk_weights.astype(hidden_states.dtype)
    if num_tokens * topk <= threshold:
        hidden_states = jnp.where(jnp.isfinite(hidden_states), hidden_states, 0)
        topk_weights = jnp.where(jnp.isfinite(topk_weights), topk_weights, 0)
    local_ids = topk_ids - experts_start
    valid = (local_ids >= 0) & (local_ids < w1.shape[0])
    topk_weights = jnp.where(valid, topk_weights, jnp.zeros_like(topk_weights))
    topk_ids = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))
    x, group_sizes, revert, weights_flat, valid_mask, token_sorted = (
        prepare_routed_gmm_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            local_num_experts=w1.shape[0],
            topk=topk,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=threshold,
            skip_padded_tokens=skip_padded_tokens,
        )
    )
    x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))
    out = moe_gmm(
        x,
        w1,
        w1_scale,
        None,
        w2,
        w2_scale,
        None,
        group_sizes,
        revert,
        weights_flat,
        valid_mask,
        token_sorted,
        activation="silu",
        num_tokens=num_tokens,
        topk=topk,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
    )
    return out[:num_tokens, :hidden_size]


def _current_fused_moe_ep(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    experts_start,
    *,
    topk,
    threshold,
    skip_padded_tokens=False,
):
    return fused_moe_func(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        None,
        None,
        topk_weights,
        topk_ids,
        experts_start=experts_start,
        topk=topk,
        activation="silu",
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
        skip_padded_tokens=skip_padded_tokens,
    )


def _experts(key, num_experts, *, fp4):
    """Served expert shapes: NVFP4 at block 256 (target model) or bf16 (draft)."""
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (num_experts, HIDDEN, 2 * INTER), jnp.bfloat16) / 8
    w2 = jax.random.normal(k2, (num_experts, INTER, HIDDEN), jnp.bfloat16) / 8
    if not fp4:
        return w1, w2, None, None
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    return w1, w2, s1, s2


def _routing(key, num_tokens, routing, *, skew_experts):
    """Router output for ``num_tokens``: ``uniform``; ``skewed`` (most routes
    on ``skew_experts``); ``padded`` (the last quarter are zero padding rows
    that keep their routing); ``zero_weight`` (those rows carry zero weights)."""
    kx, kl = jax.random.split(key)
    x = (jax.random.normal(kx, (num_tokens, HIDDEN)) / 4).astype(jnp.bfloat16)
    logits = jax.random.normal(kl, (num_tokens, NUM_EXPERTS))
    if routing == "skewed":
        logits = logits + jnp.where(jnp.arange(NUM_EXPERTS) < skew_experts, 4.0, 0.0)
    pad = num_tokens // 4 if routing in ("padded", "zero_weight") else 0
    if pad:
        x = x.at[-pad:].set(0)
    weights, ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), TOPK)
    weights = weights.astype(jnp.bfloat16)
    if routing == "zero_weight":
        weights = weights.at[-pad:].set(0)
    return x, weights, ids.astype(jnp.int32)


def _assert_bitwise(got, want):
    got, want = np.asarray(got), np.asarray(want)
    assert np.isfinite(want.astype(np.float32)).all()
    assert np.abs(want.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


@pytest.mark.parametrize(
    "num_tokens,routing,fp4,skip_padded",
    [
        (24, "uniform", True, False),  # 240 routed rows: K=0 step, draft passes
        (96, "uniform", True, False),  # 960: dp2xtp4 K=3 verify
        (384, "uniform", True, False),  # 3840: 8 DP members at K=3, the cap
        (96, "skewed", True, False),  # every route of most tokens is local
        (384, "skewed", True, False),
        (96, "padded", True, False),
        (96, "zero_weight", True, True),  # TPU_MOE_SKIP_PADDED_TOKENS
        (96, "zero_weight", True, False),
        (24, "uniform", False, False),  # bf16 MTP draft experts
        (96, "skewed", False, False),
    ],
)
def test_default_onehot_path_matches_previous(num_tokens, routing, fp4, skip_padded):
    """The default EP path's one-hot permute/combine, one sort over
    (expert, route) keys, is bitwise the argsort + inverse-permutation path."""
    (device,) = _require_tpu_devices(1)
    experts_start = 2 * LOCAL
    w1, w2, s1, s2 = _experts(jax.random.key(1 + fp4), LOCAL, fp4=fp4)
    x, weights, ids = _routing(
        jax.random.key(num_tokens), num_tokens, routing, skew_experts=3 * LOCAL
    )
    kw = dict(topk=TOPK, threshold=num_tokens * TOPK, skip_padded_tokens=skip_padded)
    args = (x, w1, w2, s1, s2, weights, ids, jnp.int32(experts_start))
    with jax.default_device(device):
        want = jax.jit(functools.partial(_previous_fused_moe_ep, **kw))(*args)
        got = jax.jit(functools.partial(_current_fused_moe_ep, **kw))(*args)
    _assert_bitwise(got, want)


@pytest.mark.parametrize(
    "num_tokens,threshold,local_routes",
    [
        (384, 3840, True),  # at the cap
        (400, 3840, False),  # above: SparseCore (its gather tiles 16 tokens)
        (96, 960, True),  # threshold = the step's routed rows
        (96, 959, False),
    ],
)
def test_default_path_onehot_cap_boundaries(num_tokens, threshold, local_routes):
    """At or below the threshold the default path takes the local-routes
    formulation; above it the SparseCore path is unchanged. Both match the
    previous path bitwise."""
    (device,) = _require_tpu_devices(1)
    w1, w2, s1, s2 = _experts(jax.random.key(1), LOCAL, fp4=True)
    x, weights, ids = _routing(
        jax.random.key(num_tokens), num_tokens, "uniform", skew_experts=0
    )
    kw = dict(topk=TOPK, threshold=threshold)
    args = (x, w1, w2, s1, s2, weights, ids, jnp.int32(LOCAL))
    fused_moe_func.clear_cache()
    with (
        jax.default_device(device),
        patch.object(
            fused_moe_gmm,
            "onehot_moe_local_routes",
            wraps=fused_moe_gmm.onehot_moe_local_routes,
        ) as local,
    ):
        want = jax.jit(functools.partial(_previous_fused_moe_ep, **kw))(*args)
        got = jax.jit(functools.partial(_current_fused_moe_ep, **kw))(*args)
    fused_moe_func.clear_cache()
    assert local.called == local_routes
    if local_routes:
        assert local.call_args.kwargs["max_local_routes"] == num_tokens * TOPK
    _assert_bitwise(got, want)


@pytest.mark.multichip
@pytest.mark.parametrize(
    "tokens_per_engine,routing",
    [
        (12, "uniform"),  # 240 routed rows per rank: K=0, 12 requests per engine
        (48, "uniform"),  # 960: K=3 verify step
        (192, "uniform"),  # 3840: the cap
        (48, "skewed"),  # most routes on rank 0
        (48, "padded"),
    ],
)
def test_default_path_dp2xtp4_matches_previous(tokens_per_engine, routing):
    """dp2xtp4 default EP layer (DP all-gather, local experts, DP
    reduce-scatter + TP all-reduce): every rank's partial and the layer output
    are bitwise the previous one-hot path's."""
    devices = _require_tpu_devices(8)
    dp_size, tp_size = 2, 4
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    ranks = P((DP_AXIS, TP_AXIS))
    w1, w2, s1, s2 = jax.jit(
        functools.partial(_experts, num_experts=NUM_EXPERTS, fp4=True),
        out_shardings=NamedSharding(mesh, ranks),
    )(jax.random.key(3))
    engines = [
        _routing(
            jax.random.key(10 * tokens_per_engine + e),
            tokens_per_engine,
            routing,
            skew_experts=LOCAL,
        )
        for e in range(dp_size)
    ]
    x, weights, ids = (jnp.stack([eng[i] for eng in engines]) for i in range(3))
    starts = jnp.arange(dp_size * tp_size, dtype=jnp.int32) * LOCAL
    routes = dp_size * tokens_per_engine * TOPK
    threshold = resolve_onehot_permute_threshold(routes)
    assert threshold >= routes

    def layer(impl):
        def body(x, weights, ids, w1, w2, s1, s2, es):
            gathered = [
                jax.lax.all_gather(a[0], DP_AXIS, tiled=True) for a in (x, weights, ids)
            ]
            partial = impl(
                gathered[0],
                w1,
                w2,
                s1,
                s2,
                gathered[1],
                gathered[2],
                es[0],
                topk=TOPK,
                threshold=threshold,
            )
            out = jax.lax.psum_scatter(
                partial, DP_AXIS, scatter_dimension=0, tiled=True
            )
            return partial[None], jax.lax.psum(out, TP_AXIS)[None]

        engine = P(DP_AXIS)
        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(engine,) * 3 + (ranks,) * 5,
                out_specs=(ranks, engine),
                check_vma=False,
            )
        )(x, weights, ids, w1, w2, s1, s2, starts)

    want_partial, want_out = layer(_previous_fused_moe_ep)
    got_partial, got_out = layer(_current_fused_moe_ep)
    _assert_bitwise(got_partial, want_partial)
    _assert_bitwise(got_out, want_out)
