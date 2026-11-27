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
"""Device tests for TPU-MoonEP dynamic hot-expert migration in ``fused_ep_moe_v2``.

Verifies numerical parity against a dense f32 reference under heavily skewed
routing distributions across:
- softmax routing (Qwen-style),
- sigmoid routing with ``score_bias`` and ``routed_scaling_factor`` (GLM/DS-style),
- non-identity ``mesh_ep_ranks`` permutations,
- masked ``-inf`` padding rows (partially filled serving buckets),
- decode fallback when ``t_local < moonep_min_local_tokens``, and
- ``token_replica_groups`` (TP+EP replicated input slicing).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import (
    AXIS,
    WeightFormat,
    fused_ep_moe_v2,
)

pytestmark = pytest.mark.multichip

EP = 8
CAPACITY = 128
FP8 = jnp.float8_e4m3fn

SHAPE = dict(hidden=512, inter=256, e_total=64, topk=4, tokens_per_rank=256)

# Same FP8 wire transport tolerances as test_fused_ep_moe_v2.py.
FP8_BATCH_L2 = 0.06
FP8_TOKEN_L2 = 0.09


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    """Skip unless eight TPU devices are available (deferred past collection)."""
    devices = jax.local_devices()
    if len(devices) < EP or devices[0].platform != "tpu":
        pytest.skip("TPU-MoonEP device tests require 8 local TPU devices")


def _mesh():
    return Mesh(np.asarray(jax.local_devices()[:EP]), axis_names=(AXIS,))


def _make_fp8_weights(mesh, seed, *, hidden, inter, e_total):
    (axis,) = mesh.axis_names
    g_local = e_total // mesh.shape[axis]

    def local(_):
        me = lax.axis_index(axis)
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.key(seed), me))
        out = []
        for key, shape in (
            (k1, (g_local, hidden, 2 * inter)),
            (k2, (g_local, inter, hidden)),
        ):
            w = jax.random.normal(key, shape, jnp.float32) / 10.0
            amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
            scale = jnp.where(amax == 0, 1.0, amax / float(jnp.finfo(FP8).max))
            out.append(
                (
                    (w / scale).astype(FP8),
                    scale.astype(jnp.float32).reshape(shape[0], shape[2]),
                )
            )
        (w1, w1s), (w2, w2s) = out
        return w1, w2, w1s, w2s

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(axis),),
            out_specs=(P(axis),) * 4,
            check_vma=False,
        )
    )(jnp.zeros((EP,), jnp.float32))


def _make_clustered_skew_gating(
    rng, *, tokens, e_total, topk, hot_rank=0, hot_mult=4.0
):
    """Build gating logits that concentrate ~hot_mult x mean load on ``hot_rank``."""
    g_local = e_total // EP
    logits = rng.normal(0.0, 0.25, size=(tokens, e_total)).astype(np.float32)
    hot_experts = np.arange(hot_rank * g_local, (hot_rank + 1) * g_local)
    hot_frac = min(0.95, hot_mult / EP)
    n_hot = int(tokens * hot_frac)
    hot_rows = rng.choice(tokens, size=n_hot, replace=False)
    # Bias a small subset of experts on hot_rank so top-B triggers cleanly.
    hot_subset = hot_experts[: max(topk, min(4, g_local))]
    for r in hot_rows:
        chosen = rng.choice(hot_subset, size=topk, replace=False)
        logits[r, chosen] += 6.0
    return logits


def _dense_reference(
    mesh,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    gating,
    *,
    topk,
    scoring_fn="softmax",
    score_bias=None,
    routed_scaling_factor=1.0,
):
    """Dense f32 MoE reference supporting softmax/sigmoid, score_bias, and -inf rows."""
    (axis,) = mesh.axis_names
    g_local = w1.shape[0] // mesh.shape[axis]
    inter = w1.shape[2] // 2
    bias_arr = None if score_bias is None else jnp.asarray(score_bias, jnp.float32)

    def local(x_g, gating_g, w1_l, w2_l, s1_l, s2_l):
        g32 = gating_g.astype(jnp.float32)
        valid = jnp.any(jnp.isfinite(g32), axis=-1)
        g_safe = jnp.where(valid[:, None], g32, 0.0)
        if scoring_fn == "softmax":
            scores = jax.nn.softmax(g_safe, axis=-1)
            topk_w, topk_idx = lax.top_k(scores, topk)
        elif scoring_fn == "sigmoid":
            scores = jax.nn.sigmoid(g_safe)
            sel = scores + bias_arr[None, :] if bias_arr is not None else scores
            _, topk_idx = lax.top_k(sel, topk)
            topk_w = jnp.take_along_axis(scores, topk_idx, axis=-1)
        else:
            raise ValueError(f"unsupported scoring_fn: {scoring_fn}")
        topk_w = topk_w / jnp.maximum(topk_w.sum(axis=-1, keepdims=True), 1e-20)
        topk_w = topk_w * jnp.float32(routed_scaling_factor)
        topk_w = jnp.where(valid[:, None], topk_w, 0.0)

        first = lax.axis_index(axis) * g_local
        x32 = x_g.astype(jnp.float32)

        def expert(e, acc):
            w1e = lax.dynamic_index_in_dim(w1_l, e, 0, keepdims=False).astype(
                jnp.float32
            ) * lax.dynamic_index_in_dim(s1_l, e, 0, keepdims=True)
            w2e = lax.dynamic_index_in_dim(w2_l, e, 0, keepdims=False).astype(
                jnp.float32
            ) * lax.dynamic_index_in_dim(s2_l, e, 0, keepdims=True)
            weight = jnp.sum(jnp.where(topk_idx == first + e, topk_w, 0.0), axis=-1)[
                :, None
            ]
            acc1 = x32 @ w1e
            row = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]) @ w2e
            return acc + weight * row

        mine = lax.fori_loop(0, g_local, expert, jnp.zeros_like(x32))
        return lax.psum(mine, axis)

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(), P(), P(axis), P(axis), P(axis), P(axis)),
            out_specs=P(),
            check_vma=False,
        )
    )(x, gating, w1, w2, w1_scale, w2_scale)


def _assert_close_to_ref(actual, want, *, valid_mask=None):
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    assert np.all(np.isfinite(a)), "non-finite values in MoonEP output"
    if valid_mask is not None:
        masked = a[~valid_mask]
        assert masked.size > 0
        assert float(np.max(np.abs(masked))) == 0.0, (
            f"masked rows must be exact zero, got {np.max(np.abs(masked))}"
        )
        a = a[valid_mask]
        w = w[valid_mask]
    batch_l2 = float(np.linalg.norm(a - w) / np.linalg.norm(w))
    per_tok = np.linalg.norm(a - w, axis=-1)
    scale = np.linalg.norm(w, axis=-1)
    worst_tok_l2 = float(np.max(per_tok / np.where(scale == 0, 1.0, scale)))
    assert batch_l2 < FP8_BATCH_L2, f"batch rel_L2={batch_l2:.5f} >= {FP8_BATCH_L2}"
    assert worst_tok_l2 < FP8_TOKEN_L2, (
        f"worst-token rel_L2={worst_tok_l2:.5f} >= {FP8_TOKEN_L2}"
    )


@pytest.fixture(scope="module")
def base_problem():
    mesh = _mesh()
    shard = NamedSharding(mesh, P(AXIS))
    tokens = SHAPE["tokens_per_rank"] * EP
    rng = np.random.default_rng(2026)

    w1, w2, s1, s2 = _make_fp8_weights(
        mesh,
        seed=11,
        hidden=SHAPE["hidden"],
        inter=SHAPE["inter"],
        e_total=SHAPE["e_total"],
    )
    x_np = (rng.normal(0.0, 0.1, size=(tokens, SHAPE["hidden"]))).astype(np.float32)
    x = jax.device_put(jnp.asarray(x_np, jnp.bfloat16), shard)
    g_skew_np = _make_clustered_skew_gating(
        rng,
        tokens=tokens,
        e_total=SHAPE["e_total"],
        topk=SHAPE["topk"],
        hot_rank=0,
        hot_mult=4.0,
    )
    g_skew = jax.device_put(jnp.asarray(g_skew_np, jnp.float32), shard)
    rank = jax.device_put(np.arange(EP, dtype=np.int32).reshape(EP, 1), shard)
    want_softmax = _dense_reference(
        mesh, x, w1, w2, s1, s2, g_skew, topk=SHAPE["topk"], scoring_fn="softmax"
    )
    return dict(
        mesh=mesh,
        shard=shard,
        x=x,
        w1=w1,
        w2=w2,
        s1=s1,
        s2=s2,
        g_skew=g_skew,
        g_skew_np=g_skew_np,
        rank=rank,
        want_softmax=want_softmax,
    )


def test_moonep_softmax_skew_parity(base_problem):
    """Softmax routing under 4x clustered skew matches dense f32 reference."""
    p = base_problem
    out = jax.jit(
        lambda x_, w1_, w2_, s1_, s2_, g_, rk_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            g_,
            rank=rk_,
            topk=SHAPE["topk"],
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            moonep_slots=4,
            moonep_min_mig_frac=0.0,
            moonep_min_local_tokens=128,
        )
    )(p["x"], p["w1"], p["w2"], p["s1"], p["s2"], p["g_skew"], p["rank"])
    _assert_close_to_ref(out, p["want_softmax"])


def test_moonep_sigmoid_bias_and_routed_scale_parity(base_problem):
    """Sigmoid + score_bias + routed_scaling_factor (GLM/DeepSeek-style) parity."""
    p = base_problem
    rng = np.random.default_rng(77)
    sb_np = rng.normal(0.0, 0.3, size=(SHAPE["e_total"],)).astype(np.float32)
    sb = jax.device_put(jnp.asarray(sb_np), NamedSharding(p["mesh"], P()))
    out = jax.jit(
        lambda x_, w1_, w2_, s1_, s2_, g_, rk_, sb_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            g_,
            rank=rk_,
            topk=SHAPE["topk"],
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            scoring_fn="sigmoid",
            score_bias=sb_,
            routed_scaling_factor=2.5,
            moonep_slots=4,
            moonep_min_mig_frac=0.0,
            moonep_min_local_tokens=128,
        )
    )(p["x"], p["w1"], p["w2"], p["s1"], p["s2"], p["g_skew"], p["rank"], sb)
    want = _dense_reference(
        p["mesh"],
        p["x"],
        p["w1"],
        p["w2"],
        p["s1"],
        p["s2"],
        p["g_skew"],
        topk=SHAPE["topk"],
        scoring_fn="sigmoid",
        score_bias=sb_np,
        routed_scaling_factor=2.5,
    )
    _assert_close_to_ref(out, want)


def test_moonep_non_identity_mesh_ep_ranks(base_problem):
    """Non-identity mesh_ep_ranks permutation routes and migrates accurately."""
    p = base_problem
    perm = tuple(reversed(range(EP)))
    g_local = SHAPE["e_total"] // EP
    idx_e = np.concatenate([np.arange(r * g_local, (r + 1) * g_local) for r in perm])
    shard = p["shard"]
    w1_p = jax.device_put(np.asarray(p["w1"])[idx_e], shard)
    w2_p = jax.device_put(np.asarray(p["w2"])[idx_e], shard)
    s1_p = jax.device_put(np.asarray(p["s1"])[idx_e], shard)
    s2_p = jax.device_put(np.asarray(p["s2"])[idx_e], shard)
    out = jax.jit(
        lambda x_, w1_, w2_, s1_, s2_, g_, rk_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            g_,
            rank=rk_,
            topk=SHAPE["topk"],
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            mesh_ep_ranks=perm,
            moonep_slots=4,
            moonep_min_mig_frac=0.0,
            moonep_min_local_tokens=128,
        )
    )(p["x"], w1_p, w2_p, s1_p, s2_p, p["g_skew"], p["rank"])
    _assert_close_to_ref(out, p["want_softmax"])


def test_moonep_masked_padding_rows(base_problem):
    """Partially filled buckets with -inf gate rows produce exact zeros on padding."""
    p = base_problem
    t_local = SHAPE["tokens_per_rank"]
    g_np = p["g_skew_np"].copy()
    valid_mask = np.ones((t_local * EP,), dtype=bool)
    for r in range(EP):
        g_np[r * t_local + 150 : (r + 1) * t_local, :] = -np.inf
        valid_mask[r * t_local + 150 : (r + 1) * t_local] = False
    g_masked = jax.device_put(jnp.asarray(g_np, jnp.float32), p["shard"])
    out = jax.jit(
        lambda x_, w1_, w2_, s1_, s2_, g_, rk_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            g_,
            rank=rk_,
            topk=SHAPE["topk"],
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            moonep_slots=4,
            moonep_min_mig_frac=0.0,
            moonep_min_local_tokens=128,
        )
    )(p["x"], p["w1"], p["w2"], p["s1"], p["s2"], g_masked, p["rank"])
    want = _dense_reference(
        p["mesh"],
        p["x"],
        p["w1"],
        p["w2"],
        p["s1"],
        p["s2"],
        g_masked,
        topk=SHAPE["topk"],
        scoring_fn="softmax",
    )
    _assert_close_to_ref(out, want, valid_mask=valid_mask)


def test_moonep_decode_fallback_below_min_local_tokens(base_problem):
    """Buckets with t_local < moonep_min_local_tokens fall back to stock kernel."""
    p = base_problem
    out = jax.jit(
        lambda x_, w1_, w2_, s1_, s2_, g_, rk_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            g_,
            rank=rk_,
            topk=SHAPE["topk"],
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            moonep_slots=4,
            moonep_min_mig_frac=0.0,
            moonep_min_local_tokens=SHAPE["tokens_per_rank"] * 2,
        )
    )(p["x"], p["w1"], p["w2"], p["s1"], p["s2"], p["g_skew"], p["rank"])
    _assert_close_to_ref(out, p["want_softmax"])


def test_moonep_token_replica_groups_tp_ep(base_problem):
    """TP+EP replicated inputs (token_replica_groups) work with MoonEP active."""
    p = base_problem
    groups = (tuple(range(EP)),)
    # 1024 logical tokens replicated on every TP rank -> t_local = 1024 // 8 = 128.
    rows = 1024
    rng = np.random.default_rng(303)
    x_logical = rng.normal(0.0, 0.1, size=(rows, SHAPE["hidden"])).astype(np.float32)
    g_logical = _make_clustered_skew_gating(
        rng,
        tokens=rows,
        e_total=SHAPE["e_total"],
        topk=SHAPE["topk"],
        hot_rank=0,
        hot_mult=4.0,
    )
    x_rep = jax.device_put(
        jnp.asarray(np.tile(x_logical, (EP, 1)), jnp.bfloat16), p["shard"]
    )
    g_rep = jax.device_put(
        jnp.asarray(np.tile(g_logical, (EP, 1)), jnp.float32), p["shard"]
    )
    out = jax.jit(
        lambda x_, w1_, w2_, s1_, s2_, g_, rk_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            g_,
            rank=rk_,
            topk=SHAPE["topk"],
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            token_replica_groups=groups,
            moonep_slots=4,
            moonep_min_mig_frac=0.0,
            moonep_min_local_tokens=128,
        )
    )(x_rep, p["w1"], p["w2"], p["s1"], p["s2"], g_rep, p["rank"])

    out_np = np.asarray(out, np.float32).reshape(EP, rows, SHAPE["hidden"])
    for r in range(1, EP):
        np.testing.assert_allclose(out_np[r], out_np[0], rtol=0.0, atol=0.0)

    want = _dense_reference(
        p["mesh"],
        jax.device_put(
            jnp.asarray(x_logical, jnp.bfloat16), NamedSharding(p["mesh"], P())
        ),
        p["w1"],
        p["w2"],
        p["s1"],
        p["s2"],
        jax.device_put(
            jnp.asarray(g_logical, jnp.float32), NamedSharding(p["mesh"], P())
        ),
        topk=SHAPE["topk"],
        scoring_fn="softmax",
    )
    _assert_close_to_ref(out_np[0], want)


def test_moonep_balanced_no_migration_parity(base_problem):
    """Balanced routing (any_migrated == 0) skips scale all-gather and matches ref."""
    p = base_problem
    tokens = SHAPE["tokens_per_rank"] * EP
    e_total = SHAPE["e_total"]
    topk = SHAPE["topk"]
    # Construct exact uniform routing: every expert receives the exact same count.
    g_bal_np = np.full((tokens, e_total), -10.0, dtype=np.float32)
    for t in range(tokens):
        chosen = [(t * topk + k) % e_total for k in range(topk)]
        g_bal_np[t, chosen] = 5.0
    g_bal = jax.device_put(jnp.asarray(g_bal_np, jnp.float32), p["shard"])
    out = jax.jit(
        lambda x_, w1_, w2_, s1_, s2_, g_, rk_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            g_,
            rank=rk_,
            topk=topk,
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            moonep_slots=4,
            moonep_min_mig_frac=0.05,
            moonep_min_local_tokens=128,
        )
    )(p["x"], p["w1"], p["w2"], p["s1"], p["s2"], g_bal, p["rank"])
    want = _dense_reference(
        p["mesh"],
        p["x"],
        p["w1"],
        p["w2"],
        p["s1"],
        p["s2"],
        g_bal,
        topk=topk,
        scoring_fn="softmax",
    )
    _assert_close_to_ref(out, want)


def test_moonep_multi_hot_homes_concurrent_rdma(base_problem):
    """Multiple overloaded homes trigger concurrent multi-home Round-0/Round-1 RDMA."""
    p = base_problem
    tokens = SHAPE["tokens_per_rank"] * EP
    e_total = SHAPE["e_total"]
    topk = SHAPE["topk"]
    g_local = e_total // EP
    rng = np.random.default_rng(404)
    logits = rng.normal(0.0, 0.25, size=(tokens, e_total)).astype(np.float32)
    # Deliberately make higher rank (5) heaviest, then rank 3, then rank 0,
    # exercising descending-overload Phase B and multi-home Round-0/Round-1 RDMA.
    hot_specs = ((5, 0.38), (3, 0.26), (0, 0.20))
    perm_rows = rng.permutation(tokens)
    cursor = 0
    for hot_rank, frac in hot_specs:
        n_rows = int(tokens * frac)
        rows = perm_rows[cursor : cursor + n_rows]
        cursor += n_rows
        hot_experts = np.arange(hot_rank * g_local, (hot_rank + 1) * g_local)[:topk]
        logits[rows[:, None], hot_experts[None, :]] += 6.0
    g_multi = jax.device_put(jnp.asarray(logits, jnp.float32), p["shard"])
    out = jax.jit(
        lambda x_, w1_, w2_, s1_, s2_, g_, rk_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            g_,
            rank=rk_,
            topk=topk,
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            moonep_slots=4,
            moonep_min_mig_frac=0.01,
            moonep_min_local_tokens=128,
        )
    )(p["x"], p["w1"], p["w2"], p["s1"], p["s2"], g_multi, p["rank"])
    want = _dense_reference(
        p["mesh"],
        p["x"],
        p["w1"],
        p["w2"],
        p["s1"],
        p["s2"],
        g_multi,
        topk=topk,
        scoring_fn="softmax",
    )
    _assert_close_to_ref(out, want)


def test_moonep_zero_local_expert_rank_asymmetric_mid_trigger(base_problem):
    """A rank with 0 local expert tokens and only migrated experts is deadlock-free."""
    p = base_problem
    tokens = SHAPE["tokens_per_rank"] * EP
    e_total = SHAPE["e_total"]
    topk = SHAPE["topk"]
    g_local = e_total // EP
    rng = np.random.default_rng(505)
    logits = _make_clustered_skew_gating(
        rng,
        tokens=tokens,
        e_total=e_total,
        topk=topk,
        hot_rank=0,
        hot_mult=4.5,
    )
    # Suppress all experts owned by rank 7 so rank 7 has n_local_active == 0
    # (mid_trigger == 0) while receiving migrated hot experts from rank 0.
    logits[:, 7 * g_local : 8 * g_local] = -1e4
    g_zero_local = jax.device_put(jnp.asarray(logits, jnp.float32), p["shard"])
    out = jax.jit(
        lambda x_, w1_, w2_, s1_, s2_, g_, rk_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            g_,
            rank=rk_,
            topk=topk,
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            moonep_slots=4,
            moonep_min_mig_frac=0.0,
            moonep_min_local_tokens=128,
        )
    )(p["x"], p["w1"], p["w2"], p["s1"], p["s2"], g_zero_local, p["rank"])
    want = _dense_reference(
        p["mesh"],
        p["x"],
        p["w1"],
        p["w2"],
        p["s1"],
        p["s2"],
        g_zero_local,
        topk=topk,
        scoring_fn="softmax",
    )
    _assert_close_to_ref(out, want)


def test_moonep_multi_layer_consecutive_no_scratch_pollution(base_problem):
    """Consecutive layers with distinct weights and skews have no scratch pollution."""
    p = base_problem
    tokens = SHAPE["tokens_per_rank"] * EP
    e_total = SHAPE["e_total"]
    topk = SHAPE["topk"]

    w1_b, w2_b, s1_b, s2_b = _make_fp8_weights(
        p["mesh"],
        seed=99,
        hidden=SHAPE["hidden"],
        inter=SHAPE["inter"],
        e_total=e_total,
    )
    rng = np.random.default_rng(606)
    g1 = jax.device_put(
        jnp.asarray(
            _make_clustered_skew_gating(
                rng, tokens=tokens, e_total=e_total, topk=topk, hot_rank=0, hot_mult=4.0
            ),
            jnp.float32,
        ),
        p["shard"],
    )
    g2 = jax.device_put(
        jnp.asarray(
            _make_clustered_skew_gating(
                rng, tokens=tokens, e_total=e_total, topk=topk, hot_rank=3, hot_mult=4.0
            ),
            jnp.float32,
        ),
        p["shard"],
    )
    g3_np = np.full((tokens, e_total), -10.0, dtype=np.float32)
    for t in range(tokens):
        g3_np[t, [(t * topk + k) % e_total for k in range(topk)]] = 5.0
    g3 = jax.device_put(jnp.asarray(g3_np, jnp.float32), p["shard"])
    g4 = jax.device_put(
        jnp.asarray(
            _make_clustered_skew_gating(
                rng, tokens=tokens, e_total=e_total, topk=topk, hot_rank=6, hot_mult=4.0
            ),
            jnp.float32,
        ),
        p["shard"],
    )

    @jax.jit
    def run_four_layers(x_, w1a, w2a, s1a, s2a, w1b, w2b, s1b, s2b, rk):
        kw = dict(
            rank=rk,
            topk=topk,
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            sharded_plan=True,
            moonep_slots=4,
            moonep_min_mig_frac=0.05,
            moonep_min_local_tokens=128,
        )
        y1 = fused_ep_moe_v2(x_, w1a, w2a, s1a, s2a, g1, **kw)
        y2 = fused_ep_moe_v2(x_, w1b, w2b, s1b, s2b, g2, **kw)
        y3 = fused_ep_moe_v2(x_, w1a, w2a, s1a, s2a, g3, **kw)
        y4 = fused_ep_moe_v2(x_, w1b, w2b, s1b, s2b, g4, **kw)
        return y1, y2, y3, y4

    y1, y2, y3, y4 = run_four_layers(
        p["x"], p["w1"], p["w2"], p["s1"], p["s2"], w1_b, w2_b, s1_b, s2_b, p["rank"]
    )
    for y, (w1_, w2_, s1_, s2_, g_) in (
        (y1, (p["w1"], p["w2"], p["s1"], p["s2"], g1)),
        (y2, (w1_b, w2_b, s1_b, s2_b, g2)),
        (y3, (p["w1"], p["w2"], p["s1"], p["s2"], g3)),
        (y4, (w1_b, w2_b, s1_b, s2_b, g4)),
    ):
        want = _dense_reference(
            p["mesh"], p["x"], w1_, w2_, s1_, s2_, g_, topk=topk, scoring_fn="softmax"
        )
        _assert_close_to_ref(y, want)


def test_moonep_non_fp8_falls_back_cleanly(base_problem):
    """Non-FP8 formats (e.g. BF16) fall back to stock fused_ep_moe_v2 cleanly."""
    p = base_problem
    w1_bf16 = (p["w1"].astype(jnp.float32) * p["s1"][:, None, :]).astype(jnp.bfloat16)
    w2_bf16 = (p["w2"].astype(jnp.float32) * p["s2"][:, None, :]).astype(jnp.bfloat16)
    out = jax.jit(
        lambda x_, w1_, w2_, g_, rk_: fused_ep_moe_v2(
            x_,
            w1_,
            w2_,
            None,
            None,
            g_,
            rank=rk_,
            topk=SHAPE["topk"],
            renormalize=True,
            mesh=p["mesh"],
            capacity=CAPACITY,
            weight_format=WeightFormat.BF16,
            sharded_plan=True,
            moonep_slots=4,
            moonep_min_mig_frac=0.0,
            moonep_min_local_tokens=128,
        )
    )(p["x"], w1_bf16, w2_bf16, p["g_skew"], p["rank"])
    _assert_close_to_ref(out, p["want_softmax"])
