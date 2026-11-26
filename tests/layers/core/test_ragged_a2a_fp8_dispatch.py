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
"""Capacity exchange that moves fp8 token rows with a float32 scale per token
(``fp8_dispatch``): close to the bf16 exchange (the expert inputs carry fp8
rounding). The route plan kernel (``route_plan_kernel``) leaves either
exchange's result bitwise unchanged, and so does the SparseCore dispatch
(``sc_dispatch``).

A broken SC dispatch protocol hangs rather than fails: its tests time out."""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.layers.core.fused_moe_gmm import (
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import DP_AXIS, TP_AXIS, ragged_a2a_moe

# fp8 rows travel as uint32 words [8, H / 32] of whole tiles: H % 4096 == 0.
HIDDEN = 4096
INTER = 512
# The fp8 row gather and the route plan kernel in a lowered program.
FP8_GATHER = re.compile(r"row_copy_gather-[A-Za-z0-9_\-]*-f8\b")
ROUTE_PLAN = re.compile(r"local_route_plan-")
SC_DISPATCH = re.compile(r"\bsc_dispatch\b(?!_)")
# The SC dispatch's remote copies from the SparseCore into TensorCore memory.
# A global flag (Mosaic's per-kernel flags do not take it); production passes
# it in LIBTPU_INIT_ARGS.
SC_FLAGS = dict(xla_mosaic_unsafe_allow_multicore_remote_dma=True)


def _require_eight_tpu_devices():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < 8:
        pytest.skip("needs 8 TPU devices")
    return devices[:8]


def _inputs(tp_size, num_tokens, num_experts, *, topk, skew):
    num_ranks = 8
    key = jax.random.key(num_ranks * 100 + num_tokens + 7 * skew)
    kx, kl, k1, k2 = jax.random.split(key, 4)
    engines = num_ranks // tp_size
    x = (jax.random.normal(kx, (engines, 1, num_tokens, HIDDEN)) / 4).astype(
        jnp.bfloat16
    )
    x = jnp.broadcast_to(x, (engines, tp_size, num_tokens, HIDDEN)).reshape(-1, HIDDEN)
    logits = jax.random.normal(kl, (engines, 1, num_tokens, num_experts))
    if skew:
        local = num_experts // num_ranks
        owner_dp = (jnp.arange(num_experts) // local) // tp_size
        logits = logits + jnp.where(owner_dp == 0, 4.0, 0.0)
    logits = jnp.broadcast_to(logits, (engines, tp_size, num_tokens, num_experts))
    weights, ids = jax.lax.top_k(jax.nn.softmax(logits.reshape(-1, num_experts)), topk)
    w1 = jax.random.normal(k1, (num_experts, HIDDEN, 2 * INTER)) / 8
    w2 = jax.random.normal(k2, (num_experts, INTER, HIDDEN)) / 8
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    es = jnp.arange(num_ranks, dtype=jnp.int32) * (num_experts // num_ranks)
    return x, w1, w2, s1, s2, weights.astype(jnp.bfloat16), ids.astype(jnp.int32), es


# (dp, tp, tokens per engine, topk, rows x, routes x, padded tokens, skewed
# routing, combine). Every layout takes the capacity exchange with the plan
# kernel; with capacity-sized buffers the skewed one overflows into the
# full-size exchange.
LAYOUTS = [
    (2, 4, 92, 10, 1.2, 2.0, "", False, "owner"),
    (2, 4, 48, 10, 1.0, 2.0, "skip", False, "owner"),  # padded tokens skipped
    (4, 2, 48, 10, 1.15, 1.5, "", False, "owner"),
    (4, 2, 24, 2, 2.0, 0.0, "", False, "owner"),  # top-2
    (8, 1, 24, 10, 1.15, 1.5, "", False, "owner"),
    (4, 2, 48, 10, 1.15, 1.5, "", False, "streamed"),
    (8, 1, 24, 10, 1.15, 1.5, "", False, "streamed"),
    (8, 1, 24, 10, 1.15, 1.5, "", True, "owner"),  # overflows: full size
]


def _setup(layout, whole_group):
    """The stacked operands, the exchange's keywords and the shared-expert
    keyword of one layout."""
    dp_size, tp_size, num_tokens, topk, factor, route_factor, padded, skew, combine = (
        LAYOUTS[layout]
    )
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        tp_size, num_tokens, 64, topk=topk, skew=skew
    )
    if padded:
        weights = weights.reshape(8, num_tokens, topk).at[:, -5:].set(0)
        weights = weights.reshape(-1, topk)
    stacked = tuple(
        a.reshape(dp_size, tp_size, -1, *a.shape[1:])
        for a in (x, w1, w2, s1, s2, weights, ids, es)
    )
    owner = combine == "owner"
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * topk
        ),
        skip_padded_tokens=padded == "skip",
        capacity_factor=factor,
        route_factor=route_factor,
        row_gather=True,
        row_scatter=True,
        streamed_combine=not owner,
        owner_combine=owner,
        owner_tail=owner,
        sorted_plan=True,
        routing_dma=True,
        send_index_rows_minor=True,
        plan_kernel=True,
        whole_group=whole_group,
    )
    kwargs = {}
    if owner:
        rng = np.random.default_rng(3)
        kwargs["shared"] = jnp.asarray(
            rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8,
            jnp.bfloat16,
        )
    return stacked, common, kwargs


@pytest.mark.multichip
@pytest.mark.parametrize("whole_group", [False, True])
@pytest.mark.parametrize("layout", range(len(LAYOUTS)))
def test_fp8_dispatch_is_close_to_bf16(layout, whole_group):
    """Every rank's result within fp8 rounding of the bf16 exchange: TP 1 / 2
    / 4, top-10 and top-2, padded tokens skipped, owner (with the shared
    expert and the owner tail) and streamed combine, and a step that
    overflows into the (bf16) full-size exchange."""
    stacked, common, kwargs = _setup(layout, whole_group)
    run = jax.jit(functools.partial(ragged_a2a_moe, fp8_dispatch=True, **common))
    lowered = run.lower(*stacked, **kwargs).as_text()
    assert FP8_GATHER.search(lowered)
    base = np.asarray(
        jax.jit(functools.partial(ragged_a2a_moe, **common))(*stacked, **kwargs)
    )
    got = np.asarray(run(*stacked, **kwargs)).astype(np.float32)
    base = base.astype(np.float32)
    assert np.abs(base).max() > 0
    assert np.isfinite(got).all()
    # fp8 e4m3 keeps 3 mantissa bits: ~2-3% relative rounding per input element,
    # averaged down by the matmuls.
    err = np.linalg.norm(got - base) / np.linalg.norm(base)
    assert err < 0.05, err


@pytest.mark.multichip
@pytest.mark.parametrize("fp8_dispatch", [False, True])
@pytest.mark.parametrize("whole_group", [False, True])
@pytest.mark.parametrize("layout", range(len(LAYOUTS)))
def test_route_plan_kernel_is_bitwise(layout, whole_group, fp8_dispatch):
    """Every rank's result with the route plan kernel equals the XLA route
    plan's bitwise, with bf16 and fp8 rows, in every layout above."""
    stacked, common, kwargs = _setup(layout, whole_group)
    common["fp8_dispatch"] = fp8_dispatch
    run = jax.jit(functools.partial(ragged_a2a_moe, route_plan_kernel=True, **common))
    assert ROUTE_PLAN.search(run.lower(*stacked, **kwargs).as_text())
    base = jax.jit(functools.partial(ragged_a2a_moe, **common))(*stacked, **kwargs)
    got = run(*stacked, **kwargs)
    assert np.abs(np.asarray(base).astype(np.float32)).max() > 0
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(base).view(np.uint16)
    )


def _sc_layouts():
    """The owner-combine layouts: the SC dispatch takes those with at most 64
    tokens per engine."""
    return [i for i, layout in enumerate(LAYOUTS) if layout[-1] == "owner"]


def _sc_run(common, compiler_options, layers=1):
    """``layers`` exchanges back to back in one jit, each layer's routing the
    previous one's with the expert ids shifted (whole local blocks plus one)
    and its tokens the previous layer's result."""
    num_experts = 64

    def run(x, w1, w2, s1, s2, weights, ids, es, **kwargs):
        outs = []
        for layer in range(layers):
            shifted = (ids + layer * (num_experts // 8 + 1)) % num_experts
            x = ragged_a2a_moe(
                x, w1, w2, s1, s2, weights, shifted, es, **common, **kwargs
            )
            outs.append(x)
        return jnp.stack(outs)

    return jax.jit(run, compiler_options=compiler_options)


def _check_sc_bitwise(layout, *, layers, compiler_options):
    stacked, common, kwargs = _setup(layout, whole_group=True)
    common.update(fp8_dispatch=True, route_plan_kernel=True)
    run = _sc_run(dict(common, sc_dispatch=True), compiler_options, layers)
    lowered = run.lower(*stacked, **kwargs).as_text()
    num_tokens = LAYOUTS[layout][2]
    taken = num_tokens <= 64
    assert bool(SC_DISPATCH.search(lowered)) == taken
    assert bool(FP8_GATHER.search(lowered)) != taken
    base = _sc_run(common, None, layers)(*stacked, **kwargs)
    got = run(*stacked, **kwargs)
    assert np.abs(np.asarray(base).astype(np.float32)).max() > 0
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(base).view(np.uint16)
    )


@pytest.mark.multichip
@pytest.mark.timeout(900)
@pytest.mark.parametrize("layout", _sc_layouts())
def test_sc_dispatch_is_bitwise(layout):
    """Every rank's result with the SC dispatch equals the fp8 exchange's
    bitwise with whole-group sizes and the owner combine: TP 1 / 2 / 4, top-10
    and top-2, padded tokens skipped and routing skewed onto one DP member
    (an engine with more tokens falls back to the all-to-all)."""
    _check_sc_bitwise(layout, layers=1, compiler_options=SC_FLAGS)


@pytest.mark.multichip
@pytest.mark.timeout(900)
@pytest.mark.parametrize("layout", [1, 4, 7])
def test_sc_dispatch_layers_back_to_back(layout):
    """Three layers in one jit, each routing differently, every one bitwise
    the fp8 exchange's: the dispatch's barrier and wave semaphores are reused
    from layer to layer."""
    _check_sc_bitwise(layout, layers=3, compiler_options=SC_FLAGS)
