"""ragged_gather_reduce_v5 (SC route-count grouped combine) vs v4 and a float64 oracle.

v5 must be bit-identical to v4 (same float32 products, same per-token k-ascending
summation order, the same plsc.pack round-to-nearest-even), and within the bf16
rounding bound of a float64 oracle. Inputs mirror moe_gmm's EP path: x is the
expert-sorted gmm2 output, indices are argsort_revert_indices (token-major),
valid = local expert; rows past the valid prefix are NaN.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v4 import (
    ragged_gather_reduce_v4,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v5 import (
    ragged_gather_reduce_v5,
    supported,
)

pytestmark = pytest.mark.tpu_test


def _inputs(num_tokens, topk, ep, num_experts, hidden, sigma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    p = (
        np.exp(rng.normal(0.0, sigma, size=num_experts))
        if sigma
        else np.ones(num_experts)
    )
    p /= p.sum()
    ids = np.stack(
        [
            rng.choice(num_experts, size=topk, replace=False, p=p)
            for _ in range(num_tokens)
        ]
    )
    per_rank = num_experts // ep
    if sigma:
        loads = np.bincount(ids.reshape(-1), minlength=num_experts)
        hot = int(np.argmax(loads.reshape(ep, per_rank).sum(1)))
        ids = (ids - hot * per_rank) % num_experts
    flat = ids.reshape(-1)
    valid = flat < per_rank
    sorted_idx = np.argsort(np.where(valid, flat, num_experts), kind="stable")
    revert = np.argsort(sorted_idx).astype(np.int32)
    w = rng.normal(size=(num_tokens, topk)).astype(np.float32)
    w = (np.exp(w) / np.exp(w).sum(-1, keepdims=True)).reshape(-1)
    x = rng.normal(size=(num_tokens * topk, hidden)).astype(np.float32)
    n_valid = int(valid.sum())
    x[n_valid:] = np.nan
    return (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(revert),
        jnp.asarray(w),
        jnp.asarray(valid),
    )


def _oracle_f64(x, idx, w, valid, k):
    xs = np.asarray(x.astype(jnp.float32), np.float64)[np.asarray(idx)]
    m = np.asarray(valid)
    xs = np.where(m[:, None], xs * np.asarray(w, np.float64)[:, None], 0.0)
    return xs.reshape(-1, k, xs.shape[-1]).sum(1)


CASES = [
    # (tokens, topk, ep, experts, hidden, sigma)  -- production cell classes first
    pytest.param(4096, 4, 2, 128, 6144, 0.0, id="twin-c8k"),
    pytest.param(4096, 8, 4, 256, 6144, 0.8, id="served-c8k-skew"),
    pytest.param(1024, 8, 4, 256, 6144, 0.0, id="served-c2k"),
    pytest.param(512, 8, 1, 256, 6144, 0.0, id="all-valid"),
    pytest.param(1024, 8, 32, 256, 6144, 0.0, id="mostly-empty"),
    pytest.param(512, 8, 128, 256, 6144, 0.0, id="empty-tiles"),
    pytest.param(2048, 8, 4, 256, 2048, 0.0, id="hidden2048"),
    pytest.param(528, 4, 2, 128, 6144, 0.0, id="t528-16tok-tiles"),
]


@pytest.mark.parametrize("tokens,topk,ep,experts,hidden,sigma", CASES)
@pytest.mark.parametrize(
    "tile_tokens,col_chunk",
    [(None, None), (16, 2048), (64, 768)],
    ids=["auto", "t16c2048", "t64c768"],
)
def test_v5_matches_v4_bitwise(
    tokens, topk, ep, experts, hidden, sigma, tile_tokens, col_chunk
):
    if tile_tokens and tokens % tile_tokens:
        pytest.skip("token count not a multiple of the tile")
    x, idx, w, valid = _inputs(tokens, topk, ep, experts, hidden, sigma)
    assert supported(x, idx, topk, tile_tokens or 16)
    got = ragged_gather_reduce_v5(
        x,
        idx,
        w,
        valid,
        reduce_group_size=topk,
        tile_tokens=tile_tokens,
        col_chunk=col_chunk,
    )
    want = ragged_gather_reduce_v4(x, idx, w, valid, reduce_group_size=topk)
    got = np.asarray(got).view(np.uint16)
    want = np.asarray(want).view(np.uint16)
    assert got.shape == want.shape
    bad = np.argwhere(got != want)
    assert bad.size == 0, f"{len(bad)} mismatches vs v4, first {bad[:4].tolist()}"


@pytest.mark.parametrize("tokens,topk,ep,experts,hidden,sigma", CASES[:3])
def test_v5_within_bf16_bound_of_f64(tokens, topk, ep, experts, hidden, sigma):
    x, idx, w, valid = _inputs(tokens, topk, ep, experts, hidden, sigma, seed=1)
    got = np.asarray(
        ragged_gather_reduce_v5(x, idx, w, valid, reduce_group_size=topk).astype(
            jnp.float32
        ),
        np.float64,
    )
    ref = _oracle_f64(x, idx, w, valid, topk)
    assert np.isfinite(got).all()
    tol = 2.0**-8 * np.abs(ref) + 1e-6
    excess = np.abs(got - ref) - tol
    assert (excess <= 0).all(), f"max excess {excess.max()}"
    empty = ~np.asarray(valid).reshape(tokens, topk).any(-1)
    assert (got[empty] == 0).all()
    assert not np.signbit(got[empty]).any()


@pytest.mark.parametrize(
    "tokens,topk,ep", [(4096, 8, 4), (2048, 4, 2)]
)  # >= 2048: v5 applies
def test_moe_gmm_flag_routes_ep_combine_to_v5_bitwise(monkeypatch, tokens, topk, ep):
    from vllm_torchtpu.layers.core import fused_moe_gmm

    hidden = 6144
    x, idx, w, valid = _inputs(tokens, topk, ep, 64 * ep, hidden)
    routes = tokens * topk
    used = []
    real_v5 = fused_moe_gmm.ragged_gather_reduce_v5

    def spy_v5(*args, **kwargs):
        used.append(kwargs.get("reduce_group_size"))
        return real_v5(*args, **kwargs)

    def run(v5, tiled):
        monkeypatch.setattr(fused_moe_gmm, "_SC_COMBINE_V5", v5)
        monkeypatch.setattr(fused_moe_gmm, "_SC_COMBINE_TILED", tiled)
        monkeypatch.setattr(fused_moe_gmm, "ragged_gather_reduce_v5", spy_v5)
        monkeypatch.setattr(fused_moe_gmm, "get_packing_factor", lambda *a, **k: 1)
        outs = iter([jnp.ones((routes, 4), jnp.bfloat16), x])
        monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", lambda *a, **k: next(outs))
        return fused_moe_gmm.moe_gmm(
            x=jnp.ones((routes, 4), jnp.bfloat16),
            w1=jnp.ones((1, 4, 4), jnp.bfloat16),
            w1_scale=None,
            w1_bias=None,
            w2=jnp.ones((1, 4, hidden), jnp.bfloat16),
            w2_scale=None,
            w2_bias=None,
            group_sizes=jnp.array([int(np.asarray(valid).sum())], jnp.int32),
            argsort_revert_indices=idx,
            topk_weights_flat=w,
            valid_mask_flat=valid,
            token_indices_sorted=idx // topk,
            activation="silu",
            num_tokens=tokens,
            topk=topk,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=0,
        )

    v4_out = np.asarray(run(False, True)).view(np.uint16)
    v2_out = np.asarray(run(False, False)).view(np.uint16)
    assert not used
    on = np.asarray(run(True, True)).view(np.uint16)
    assert used == [topk]
    assert (on == v4_out).all()
    assert (on == v2_out).all()


def test_flag_falls_back_where_v5_does_not_apply(monkeypatch):
    from vllm_torchtpu.layers.core import fused_moe_gmm

    monkeypatch.setattr(fused_moe_gmm, "_SC_COMBINE_V5", True)
    big = jnp.zeros((32768, 6144), jnp.bfloat16)
    assert fused_moe_gmm._use_sc_combine_v5(big, jnp.zeros((32768,), jnp.int32), 8)
    # token count not a multiple of the 16-token tile
    assert not fused_moe_gmm._use_sc_combine_v5(
        big, jnp.zeros((8 * 4100,), jnp.int32), 8
    )
    # below 2048 tokens (no 64-token tile per subcore): v4 / v2 instead
    assert not fused_moe_gmm._use_sc_combine_v5(
        big, jnp.zeros((8 * 1024,), jnp.int32), 8
    )
    assert fused_moe_gmm._use_sc_combine_v5(big, jnp.zeros((8 * 2048,), jnp.int32), 8)
    # topk above 16 routes per token
    assert not fused_moe_gmm._use_sc_combine_v5(
        big, jnp.zeros((32 * 1024,), jnp.int32), 32
    )
    # small source: v2's own cutoff (XLA fallback) applies
    small = jnp.zeros((256, 6144), jnp.bfloat16)
    assert not fused_moe_gmm._use_sc_combine_v5(small, jnp.zeros((256,), jnp.int32), 8)
    monkeypatch.setattr(fused_moe_gmm, "_SC_COMBINE_V5", False)
    assert not fused_moe_gmm._use_sc_combine_v5(big, jnp.zeros((32768,), jnp.int32), 8)
