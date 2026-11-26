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
"""The route plan kernel of the capacity exchange against the XLA ops it
replaces: `route_plan` with the owner and streamed destinations, the fp8
exchange's receive-row scales and the owner combine's `owner_plan`."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.route_plan import (
    local_route_plan,
    local_route_plan_supported,
)
from vllm_torchtpu.layers.core import ragged_a2a_moe as ram

TOPK = 10


def gathered_words(rng, num_members, tp_size, num_tokens, local, *, routing):
    """Packed routing [D, K, S] (bf16 weights in the low half, a fifth of them
    zero, some subnormal or non-finite): ``uniform``, ``skewed`` (most routes
    on the first ranks), ``hot_rank`` (rank 0's experts first in every token),
    ``padded`` (uniform, the last tokens' ids -1 and weights 0) or ``repeated``
    (uniform, some tokens naming one expert three times)."""
    slots = num_members * tp_size * local
    count = num_members * num_tokens
    if routing in ("uniform", "padded", "repeated"):
        ids = np.stack([rng.permutation(slots)[:TOPK] for _ in range(count)])
    elif routing == "skewed":
        p = np.exp(-np.arange(slots) / (slots / 6))
        ids = np.stack(
            [
                rng.choice(slots, TOPK, replace=False, p=p / p.sum())
                for _ in range(count)
            ]
        )
    elif routing == "hot_rank":
        hot = min(TOPK, local)
        ids = np.stack([
            np.r_[rng.permutation(local)[:hot],
                  rng.permutation(np.arange(local, slots))[:TOPK - hot]]
            for _ in range(count)
        ])  # fmt: skip
    else:
        raise ValueError(routing)
    ids = ids.reshape(num_members, num_tokens, TOPK)
    weights = rng.integers(0x3000, 0x3F80, ids.shape)
    weights = np.where(rng.random(ids.shape) < 0.2, 0, weights)
    weights = np.where(rng.random(ids.shape) < 0.03, 0x0011, weights)  # subnormal
    weights = np.where(rng.random(ids.shape) < 0.03, 0x7FC0, weights)  # nan
    if routing == "padded":
        ids[:, -max(1, num_tokens // 6) :] = -1
        weights[:, -max(1, num_tokens // 6) :] = 0
    if routing == "repeated":
        again = rng.random(ids.shape[:2]) < 0.3
        for k in (2, 5):
            ids[..., k] = np.where(again, ids[..., 0], ids[..., k])
    words = (ids.astype(np.int64) << 16) | weights
    return np.swapaxes(words, 1, 2).astype(np.int32)


def reference(words, ep, fits, scale, *, local, tp_size, owner, skip, rows, routes):
    """`route_plan` and the receive-row scale scatter of `_capacity_exchange`,
    with the receive rows the exchange plan gives."""
    num_members = words.shape[0]
    packed = jnp.swapaxes(words, 1, 2)
    ids = packed >> 16
    member_index = functools.partial(
        ram._member_index,
        ep_rank=ep,
        dp_size=num_members,
        tp_size=tp_size,
        local_num_experts=local,
    )
    need_in = (member_index(ids) == ep // tp_size).any(axis=2)
    recv_start = ram._exclusive_cumsum(need_in.sum(axis=1, dtype=jnp.int32), 0)
    in_row = recv_start[:, None] + jnp.cumsum(need_in, axis=1, dtype=jnp.int32) - 1
    if owner:
        dests = functools.partial(
            ram.owner_dests,
            dp_size=num_members,
            tp_size=tp_size,
            local_num_experts=local,
        )
    else:
        dests = functools.partial(
            ram.streamed_dests, member_index=member_index, dp_size=num_members
        )
    weights = None
    if skip:
        weights = ram._unpack_routes(packed)[1]
    plan = ram.route_plan(
        ids,
        in_row,
        jnp.bool_(fits),
        jnp.array([ep * local], jnp.int32),
        dests,
        local_num_experts=local,
        routes=routes,
        rows=rows,
        gathered_weights=weights,
    )
    reach = ((ids >= 0) & (ids // local == ep)).any(axis=2)
    row_scale = (
        jnp.ones((rows + 1,), jnp.float32)
        .at[jnp.where(reach, in_row, rows)]
        .set(scale)[:rows]
    )
    return plan, row_scale


def check(words, ep, fits, scale, interpret, **kw):
    (token_sorted, group_sizes, dest_member, dest_row), row_scale = reference(
        words, ep, fits, scale, **kw
    )
    got = local_route_plan(
        words,
        jnp.array([ep], jnp.int32),
        jnp.array([int(fits)], jnp.int32),
        scale,
        local_num_experts=kw["local"],
        tp_size=kw["tp_size"],
        owner=kw["owner"],
        skip_padded=kw["skip"],
        rows=kw["rows"],
        routes=kw["routes"],
        owner_plan=kw["owner"],
        interpret=interpret,
    )
    n = int(group_sizes.sum())
    np.testing.assert_array_equal(got.group_sizes, group_sizes, "group_sizes")
    np.testing.assert_array_equal(got.token_sorted, token_sorted, "token_sorted")
    # Past the sorted routes the readers take no destination.
    np.testing.assert_array_equal(got.dest_member[:n], dest_member[:n], "member")
    np.testing.assert_array_equal(got.dest_row[:n], dest_row[:n], "dest_row")
    if fits:
        # A step that overflows lands no rows (the caller discards it).
        np.testing.assert_array_equal(
            np.asarray(got.row_scale).view(np.int32),
            np.asarray(row_scale).view(np.int32),
            "row_scale",
        )
    assert got.route_scale is None
    by_route = local_route_plan(
        words,
        jnp.array([ep], jnp.int32),
        jnp.array([int(fits)], jnp.int32),
        scale,
        local_num_experts=kw["local"],
        tp_size=kw["tp_size"],
        owner=kw["owner"],
        skip_padded=kw["skip"],
        rows=kw["rows"],
        routes=kw["routes"],
        interpret=interpret,
        route_scale=True,
    )
    assert by_route.row_scale is None
    for name in ("token_sorted", "group_sizes", "dest_member", "dest_row"):
        np.testing.assert_array_equal(
            getattr(by_route, name), getattr(got, name), f"{name} by route"
        )
    want_route_scale = np.ones((kw["routes"],), np.float32)
    want_route_scale[:n] = np.asarray(row_scale)[np.asarray(token_sorted)[:n]]
    np.testing.assert_array_equal(
        np.asarray(by_route.route_scale).view(np.int32),
        want_route_scale.view(np.int32),
        "route_scale",
    )
    if not kw["owner"]:
        assert got.owner is None
        return
    num_members, _, num_tokens = words.shape
    ids, weights = ram._unpack_routes(jnp.swapaxes(words, 1, 2)[ep // kw["tp_size"]])
    want = ram.owner_plan(
        ids,
        weights,
        jnp.bool_(fits),
        ep,
        skip_padded_tokens=kw["skip"],
        dtype=jnp.bfloat16,
        dp_size=num_members,
        tp_size=kw["tp_size"],
        local_num_experts=kw["local"],
    )
    for name, a, b in zip(want._fields, got.owner, want, strict=True):
        np.testing.assert_array_equal(
            np.asarray(a).view(np.int32), np.asarray(b).view(np.int32), name
        )


# (members, TP size, tokens per member, local experts): the 32-chip decode
# exchange (16 members x TP 4, 48 tokens, 768 experts) and smaller ones.
LAYOUTS = [
    (16, 4, 48, 12),
    (16, 2, 20, 8),
    (2, 4, 92, 8),
    (4, 2, 24, 16),
    (8, 1, 128, 8),
]
ROUTINGS = ["uniform", "skewed", "hot_rank", "padded", "repeated"]


def _run(layout, routing, interpret, ranks):
    num_members, tp_size, num_tokens, local = layout
    rng = np.random.default_rng(num_members * 100 + num_tokens + len(routing))
    words = jnp.asarray(
        gathered_words(rng, num_members, tp_size, num_tokens, local, routing=routing)
    )
    scale = jnp.asarray(rng.random((num_members, num_tokens)), jnp.float32)
    rows = num_members * num_tokens
    for owner in (True, False):
        for skip in (False, True):
            kw = dict(
                local=local,
                tp_size=tp_size,
                owner=owner,
                skip=skip,
                rows=rows,
                routes=rows * TOPK,
            )
            assert local_route_plan_supported(
                num_members=num_members,
                num_tokens=num_tokens,
                topk=TOPK,
                local_num_experts=local,
                tp_size=tp_size,
                rows=rows,
                owner=owner,
                owner_plan=owner,
            )
            for ep in ranks(num_members * tp_size, rng):
                check(words, ep, True, scale, interpret, **kw)
            check(words, 0, False, scale, interpret, **kw)


def _ranks(n, rng):
    return sorted({0, n - 1, int(rng.integers(n))})


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("routing", ROUTINGS)
def test_route_plan_kernel_matches_the_xla_plan(layout, routing):
    """On TPU: the kernel's sorted routes, group sizes and receive-row scales
    equal the XLA ops' bitwise, at three ranks and on a step that overflows:
    owner and streamed destinations, padded routes skipped or not, routing
    skewed onto a few ranks or every route of a token on one, and tokens
    naming an expert more than once (equal sort keys)."""
    if jax.default_backend() != "tpu":
        pytest.skip("needs a TPU")
    _run(layout, routing, False, _ranks)


@pytest.mark.cpu_test
@pytest.mark.parametrize("routing", ["uniform", "repeated"])
def test_route_plan_kernel_matches_the_xla_plan_interpreted(routing):
    """Interpret mode, a small layout."""
    _run((2, 4, 12, 8), routing, True, lambda n, rng: [0, n - 1])
