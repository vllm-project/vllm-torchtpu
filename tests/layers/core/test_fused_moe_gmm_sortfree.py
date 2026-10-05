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
"""The sort-free one-hot MoE routing prep is bit-identical to the sorted one.

Geometry follows GLM-5.2 under EP16: 256 global experts, 16 per rank, top-8,
one-hot permute threshold 2048, so every decode batch takes the one-hot path.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    moe_gmm,
    prepare_routed_gmm_inputs,
)

NUM_EXPERTS = 256
EP = 16
LOCAL_EXPERTS = NUM_EXPERTS // EP
TOPK = 8
THRESHOLD = 2048
OUTPUT_NAMES = (
    "x",
    "group_sizes",
    "argsort_revert_indices",
    "topk_weights_flat",
    "valid_mask",
    "token_indices_sorted",
)


def _route(seed, num_tokens, *, rank, skew, num_padded=0, repeat=1):
    """Global top-8 routing remapped to rank-local ids like fused_moe_func.

    ``skew`` is the probability mass put on this rank's first two experts;
    ``repeat`` makes runs of identical routing (MTP drafts of one request);
    the last ``num_padded`` tokens carry zero weights like padded tokens.
    """
    rng = np.random.default_rng(seed)
    probs = np.full(NUM_EXPERTS, 1.0)
    hot = rank * LOCAL_EXPERTS
    probs[hot : hot + 2] += skew * NUM_EXPERTS
    probs /= probs.sum()
    base = -(-num_tokens // repeat)
    ids = np.stack(
        [rng.choice(NUM_EXPERTS, TOPK, replace=False, p=probs) for _ in range(base)]
    )
    ids = np.repeat(ids, repeat, axis=0)[:num_tokens]
    weights = rng.random((num_tokens, TOPK)).astype(np.float32) + 0.01
    if num_padded:
        weights[-num_padded:] = 0.0
    local = ids - hot
    valid = (local >= 0) & (local < LOCAL_EXPERTS)
    weights = np.where(valid, weights, 0.0)
    local = np.where(valid, local, -1)
    hidden = rng.standard_normal((num_tokens, 256)).astype(np.float32)
    return (
        jnp.asarray(hidden, dtype=jnp.bfloat16),
        jnp.asarray(local, dtype=jnp.int32),
        jnp.asarray(weights, dtype=jnp.bfloat16),
        ids,
    )


@functools.partial(jax.jit, static_argnames=("skip_padded_tokens", "onehot_sortfree"))
def _prep(hidden, ids, weights, *, skip_padded_tokens, onehot_sortfree):
    return prepare_routed_gmm_inputs(
        hidden,
        ids,
        weights,
        local_num_experts=LOCAL_EXPERTS,
        topk=TOPK,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=THRESHOLD,
        skip_padded_tokens=skip_padded_tokens,
        onehot_sortfree=onehot_sortfree,
    )


def _mismatches(want, got):
    bad = []
    for name, w, g in zip(OUTPUT_NAMES, want, got):
        w, g = np.asarray(w), np.asarray(g)
        if w.dtype != g.dtype or w.shape != g.shape:
            bad.append(f"{name}: {w.dtype}{w.shape} vs {g.dtype}{g.shape}")
        elif not np.array_equal(w.view(np.uint8), g.view(np.uint8)):
            bad.append(f"{name}: {int((w != g).sum())} elements differ")
    return bad


CASES = [
    # (num_tokens, rank, skew, num_padded, repeat)
    (4, 0, 0.0, 0, 1),
    (8, 3, 0.0, 2, 1),
    (16, 5, 0.0, 0, 1),  # decode: 128 routed rows
    (16, 7, 0.5, 3, 1),  # skewed onto two experts
    (16, 15, 4.0, 0, 1),  # nearly every row in one expert
    (32, 9, 0.0, 5, 1),
    (128, 2, 0.0, 0, 1),
    (128, 11, 1.0, 17, 1),
    # MTP: 4 requests x (1 + 3 drafts), each request's tokens route alike,
    # and a trailing rejected/padded slot.
    (16, 4, 0.0, 1, 4),
    # MTP drafter step: one token per request.
    (4, 4, 0.0, 0, 1),
    # Under the counting-sort block limit (num_buckets > rows).
    (1, 0, 0.0, 0, 1),
    (2, 6, 0.0, 0, 1),
    # Odd, non-power-of-two token counts.
    (3, 1, 0.0, 0, 1),
    (13, 8, 2.0, 4, 1),
]


@pytest.mark.parametrize("skip_padded_tokens", [False, True])
@pytest.mark.parametrize(("num_tokens", "rank", "skew", "num_padded", "repeat"), CASES)
def test_sortfree_prep_bit_identical(
    num_tokens, rank, skew, num_padded, repeat, skip_padded_tokens
):
    for seed in range(4):
        hidden, ids, weights, _ = _route(
            seed + 100 * num_tokens + rank,
            num_tokens,
            rank=rank,
            skew=skew,
            num_padded=num_padded,
            repeat=repeat,
        )
        want = _prep(
            hidden,
            ids,
            weights,
            skip_padded_tokens=skip_padded_tokens,
            onehot_sortfree=False,
        )
        got = _prep(
            hidden,
            ids,
            weights,
            skip_padded_tokens=skip_padded_tokens,
            onehot_sortfree=True,
        )
        assert _mismatches(want, got) == [], f"seed {seed}"


def test_sortfree_all_rows_one_bucket_and_all_invalid():
    hidden = jnp.ones((16, 256), jnp.bfloat16)
    weights = jnp.ones((16, TOPK), jnp.bfloat16)
    for ids in (
        jnp.full((16, TOPK), 3, jnp.int32),  # every row in expert 3
        jnp.full((16, TOPK), -1, jnp.int32),  # every row non-local
    ):
        want = _prep(
            hidden, ids, weights, skip_padded_tokens=True, onehot_sortfree=False
        )
        got = _prep(hidden, ids, weights, skip_padded_tokens=True, onehot_sortfree=True)
        assert _mismatches(want, got) == []


def test_sortfree_matches_numpy_stable_sort():
    hidden, ids, weights, _ = _route(7, 16, rank=5, skew=0.5, num_padded=2)
    _, group_sizes, revert, _, valid, tokens = _prep(
        hidden, ids, weights, skip_padded_tokens=True, onehot_sortfree=True
    )
    flat = np.asarray(ids).reshape(-1)
    keys = np.where(np.asarray(valid), flat, LOCAL_EXPERTS)
    order = np.argsort(keys, kind="stable")
    np.testing.assert_array_equal(np.asarray(revert), np.argsort(order, kind="stable"))
    np.testing.assert_array_equal(np.asarray(tokens), order // TOPK)
    np.testing.assert_array_equal(
        np.asarray(group_sizes), np.bincount(keys, minlength=LOCAL_EXPERTS + 1)[:-1]
    )


def test_positive_control_wrong_permutation_is_caught():
    """Swapping two rows of one bucket keeps a valid permutation and the same
    group sizes, but breaks stability; the comparison must notice."""
    hidden, ids, weights, _ = _route(3, 16, rank=5, skew=0.5)
    want = _prep(hidden, ids, weights, skip_padded_tokens=False, onehot_sortfree=False)
    revert = np.asarray(want[2]).copy()
    keys = np.where(np.asarray(want[4]), np.asarray(ids).reshape(-1), LOCAL_EXPERTS)
    # Two rows of different tokens in one bucket.
    tokens_of_row = np.arange(revert.size) // TOPK
    bucket = np.bincount(keys).argmax()
    rows = np.flatnonzero(keys == bucket)
    a = rows[0]
    b = next(r for r in rows if tokens_of_row[r] != tokens_of_row[a])
    revert[a], revert[b] = revert[b], revert[a]
    tokens_sorted = np.asarray(want[5]).copy()
    tokens_sorted[revert[a]], tokens_sorted[revert[b]] = (
        tokens_sorted[revert[b]],
        tokens_sorted[revert[a]],
    )
    wrong = (*want[:2], jnp.asarray(revert), *want[3:5], jnp.asarray(tokens_sorted))
    bad = _mismatches(want, wrong)
    assert any(m.startswith("argsort_revert_indices") for m in bad)
    assert any(m.startswith("token_indices_sorted") for m in bad)


def _count_sorts(jaxpr):
    """Sort equations, counting each call site of a nested jaxpr."""
    n = 0
    for eqn in jaxpr.eqns:
        n += eqn.primitive.name == "sort"
        for param in eqn.params.values():
            for sub in param if isinstance(param, (tuple, list)) else (param,):
                if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                    n += _count_sorts(sub.jaxpr)
                elif hasattr(sub, "eqns"):
                    n += _count_sorts(sub)
    return n


def test_sortfree_removes_every_sort_at_decode():
    hidden, ids, weights, _ = _route(0, 16, rank=0, skew=0.0)

    def sorts(sortfree):
        jaxpr = jax.make_jaxpr(
            functools.partial(
                prepare_routed_gmm_inputs,
                local_num_experts=LOCAL_EXPERTS,
                topk=TOPK,
                use_ep=True,
                use_sparse_core=True,
                onehot_moe_permute_threshold=THRESHOLD,
                skip_padded_tokens=True,
                onehot_sortfree=sortfree,
            )
        )(hidden, ids, weights)
        return _count_sorts(jaxpr.jaxpr)

    # argsort, argsort(argsort), and the packed value sort.
    assert sorts(False) == 3
    assert sorts(True) == 0


def test_sortfree_inactive_off_the_onehot_path():
    """Above the row cap, or on the ragged path, the sorted form runs."""
    hidden, ids, weights, _ = _route(1, 256, rank=0, skew=0.0)  # 2048 rows
    jaxpr = jax.make_jaxpr(
        functools.partial(
            prepare_routed_gmm_inputs,
            local_num_experts=LOCAL_EXPERTS,
            topk=TOPK,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=THRESHOLD,
            onehot_sortfree=True,
        )
    )(hidden, ids, weights)
    assert _count_sorts(jaxpr.jaxpr) > 0


def test_sortfree_env_flag(monkeypatch):
    getter = envs.environment_variables["TPU_MOE_ONEHOT_SORTFREE"]
    monkeypatch.delenv("TPU_MOE_ONEHOT_SORTFREE", raising=False)
    assert getter() is False
    monkeypatch.setenv("TPU_MOE_ONEHOT_SORTFREE", "1")
    assert getter() is True


def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("gmm_v2 needs a TPU")


def _weights(seed, hidden_size=1024, intermediate=512):
    k1, k2 = jax.random.split(jax.random.key(seed))
    w1 = (
        jax.random.normal(k1, (LOCAL_EXPERTS, hidden_size, 2 * intermediate)) / 10
    ).astype(jnp.bfloat16)
    w2 = (
        jax.random.normal(k2, (LOCAL_EXPERTS, intermediate, hidden_size)) / 10
    ).astype(jnp.bfloat16)
    return w1, w2


def _moe(hidden, w1, w2, weights, global_ids, rank, *, skip, sortfree):
    return fused_moe_func(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        topk_weights=weights,
        topk_ids=global_ids,
        experts_start=jnp.asarray(rank * LOCAL_EXPERTS, jnp.int32),
        topk=TOPK,
        activation="silu",
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=THRESHOLD,
        skip_padded_tokens=skip,
        onehot_sortfree=sortfree,
    )


@pytest.mark.parametrize(
    ("num_tokens", "rank", "skew", "num_padded", "repeat"),
    [
        (4, 0, 0.0, 0, 1),
        (8, 3, 0.0, 2, 1),
        (16, 5, 0.0, 0, 1),
        (16, 15, 4.0, 3, 1),
        (32, 9, 0.0, 5, 1),
        (128, 11, 1.0, 17, 1),
        (16, 4, 0.0, 1, 4),  # MTP verify batch
        (4, 4, 0.0, 0, 1),  # MTP drafter step
    ],
)
@pytest.mark.parametrize("skip_padded_tokens", [False, True])
def test_sortfree_moe_output_bit_identical(
    num_tokens, rank, skew, num_padded, repeat, skip_padded_tokens
):
    _require_tpu()
    w1, w2 = _weights(num_tokens + rank)
    rng = np.random.default_rng(num_tokens * 31 + rank)
    _, _, _, global_ids = _route(
        num_tokens * 7 + rank,
        num_tokens,
        rank=rank,
        skew=skew,
        num_padded=num_padded,
        repeat=repeat,
    )
    hidden = jnp.asarray(
        rng.standard_normal((num_tokens, w1.shape[1])) / 4, dtype=jnp.bfloat16
    )
    weights = rng.random((num_tokens, TOPK)).astype(np.float32) + 0.01
    if num_padded:
        weights[-num_padded:] = 0.0
    weights = jnp.asarray(weights, dtype=jnp.bfloat16)
    ids = jnp.asarray(global_ids, dtype=jnp.int32)
    want = np.asarray(
        _moe(
            hidden, w1, w2, weights, ids, rank, skip=skip_padded_tokens, sortfree=False
        )
    )
    got = np.asarray(
        _moe(hidden, w1, w2, weights, ids, rank, skip=skip_padded_tokens, sortfree=True)
    )
    assert np.isfinite(want.astype(np.float32)).all()
    np.testing.assert_array_equal(want.view(np.uint16), got.view(np.uint16))


def test_positive_control_wrong_permutation_changes_moe_output():
    """A stability-breaking swap in the revert indices must move the combined
    output, so bit-equality of the outputs above is a real check."""
    _require_tpu()
    num_tokens, rank = 16, 5
    w1, w2 = _weights(0)
    hidden, ids, weights, _ = _route(11, num_tokens, rank=rank, skew=0.5)
    hidden = jnp.asarray(
        np.random.default_rng(0).standard_normal((num_tokens, w1.shape[1])) / 4,
        dtype=jnp.bfloat16,
    )
    x, gs, revert, wflat, valid, tok = _prep(
        hidden, ids, weights, skip_padded_tokens=False, onehot_sortfree=True
    )
    run = functools.partial(
        moe_gmm,
        w1=w1,
        w1_scale=None,
        w1_bias=None,
        w2=w2,
        w2_scale=None,
        w2_bias=None,
        group_sizes=gs,
        topk_weights_flat=wflat,
        valid_mask_flat=valid,
        token_indices_sorted=tok,
        activation="silu",
        num_tokens=num_tokens,
        topk=TOPK,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=THRESHOLD,
    )
    good = np.asarray(run(x, argsort_revert_indices=revert))
    bad_revert = np.asarray(revert).copy()
    keys = np.where(np.asarray(valid), np.asarray(ids).reshape(-1), LOCAL_EXPERTS)
    rows = np.flatnonzero(keys == np.bincount(keys)[:LOCAL_EXPERTS].argmax())
    a = rows[0]
    b = next(r for r in rows if r // TOPK != a // TOPK)
    bad_revert[a], bad_revert[b] = bad_revert[b], bad_revert[a]
    bad = np.asarray(run(x, argsort_revert_indices=jnp.asarray(bad_revert)))
    assert not np.array_equal(good.view(np.uint16), bad.view(np.uint16))
