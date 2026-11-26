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
"""SparseCore dispatch of the fp8 token rows on a DP x TP mesh of 8 devices.

Every receiver's rows [0, sum(group_sizes)) must be bitwise the rows the
capacity exchange delivers, in `local_route_plan` order: route j's row is the
source rows (`copy_send_rows_fp8`'s words) of the token at its receive row
``token_sorted[j]``. A TensorCore kernel waits each wave's rows (and the
sentinel rows) on its semaphore, so every semaphore ends balanced, and then
copies the receive buffer out.

A broken protocol hangs rather than fails: run with a timeout.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.megablox.route_plan import local_route_plan
from vllm_torchtpu.kernels.megablox.row_copy import (
    copy_send_rows_fp8,
    fp8_source_rows,
)
from vllm_torchtpu.kernels.megablox.sc_dispatch import (
    TABLE_HEADER,
    sc_dispatch,
    sc_drain,
    sc_send_plan,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import DP_AXIS, TP_AXIS, _fp8_scales

pytestmark = pytest.mark.multichip

N_DEV = 8
TOPK = 10
LOCAL = 12
TOKENS = 48
COLLECTIVE_ID = 7
_FLAGS = dict(
    xla_mosaic_unsafe_allow_multicore_remote_dma=True,
    xla_msa_enable=False,
    xla_tpu_vmem_scavenging_mode="NONE",
)


def _mesh(dp):
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < N_DEV:
        pytest.skip(f"needs {N_DEV} TPU devices")
    if pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("needs SparseCore")
    return Mesh(
        np.asarray(devices[:N_DEV]).reshape(dp, N_DEV // dp), (DP_AXIS, TP_AXIS)
    )


def _routing(rng, dp, tp, kind):
    """Packed routing [D, K, S]: bf16 weights in the low half (a fifth zero,
    some subnormal or NaN). ``skewed``: most routes on the first ranks;
    ``padded``: the last tokens' ids -1 with weight 0; ``repeated``: some
    tokens name one expert three times."""
    slots = dp * tp * LOCAL
    count = dp * TOKENS
    if kind == "skewed":
        p = np.exp(-np.arange(slots) / (slots / 6))
        ids = np.stack(
            [
                rng.choice(slots, TOPK, replace=False, p=p / p.sum())
                for _ in range(count)
            ]
        )
    else:
        ids = np.stack([rng.permutation(slots)[:TOPK] for _ in range(count)])
    ids = ids.reshape(dp, TOKENS, TOPK)
    weights = rng.integers(0x3000, 0x3F80, ids.shape)
    weights = np.where(rng.random(ids.shape) < 0.2, 0, weights)
    weights = np.where(rng.random(ids.shape) < 0.03, 0x0011, weights)
    weights = np.where(rng.random(ids.shape) < 0.03, 0x7FC0, weights)
    if kind == "padded":
        ids[:, -TOKENS // 6 :] = -1
        weights[:, -TOKENS // 6 :] = 0
    if kind == "repeated":
        again = rng.random(ids.shape[:2]) < 0.3
        for k in (2, 5):
            ids[..., k] = np.where(again, ids[..., 0], ids[..., k])
    words = (ids.astype(np.int64) << 16) | weights
    return np.swapaxes(words, 1, 2).astype(np.int32)


def _tokens(rng, dp, hidden):
    rows = dp * TOKENS
    x = rng.standard_normal((rows, hidden)) * rng.uniform(0.01, 30, (rows, 1))
    x[3, 7] = np.inf
    x[5] = 0
    return x


def _tc_receive(counts, recv, sems):
    """Waits counts[e] rows on sems[e] for every semaphore, then copies the
    receive buffer out."""
    tc_mesh = pltpu.create_tensorcore_mesh("tc", num_cores=1)
    sem_ref = jax.new_ref(sems, memory_space=pltpu.SEMAPHORE @ tc_mesh)
    num_sems = sems.shape[0]

    def body(cnt_hbm, buf_ref, sem_r, out_ref, cnt_s):
        pltpu.sync_copy(cnt_hbm, cnt_s)
        for e in range(num_sems):

            def wait(b, e=e):
                rows = buf_ref.at[pl.ds(0, b)]
                pltpu.make_async_copy(rows, rows, sem_r.at[e]).wait()

            n = cnt_s[e]

            @pl.loop(0, n // 64)
            def _(i):
                wait(64)

            b = 32
            while b:
                pl.when((n & b) != 0)(functools.partial(wait, b))
                b //= 2
        pltpu.sync_copy(buf_ref, out_ref)

    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct(recv.shape, recv.dtype),
        mesh=tc_mesh,
        scratch_types=[pltpu.SMEM(counts.shape, jnp.int32)],
        name="tc_wave_receive",
    )(counts, recv, sem_ref)


def _skew(x, me, i, dp):
    """Delays this member's dispatch by 0..(dp - 1) * 4 small matmuls,
    varying by call."""
    steps = lax.rem(me * 5 + i * 3, dp) * 4
    y = jnp.full((1024, 1024), 0.5, jnp.bfloat16)
    y = lax.fori_loop(
        0, steps, lambda _, y: jnp.dot(y, y, preferred_element_type=jnp.bfloat16), y
    )
    return lax.optimization_barrier((x, y))[0]


def _run(mesh, x, words, fits, *, skew, flags=_FLAGS):
    """Runs one dispatch per call (leading axis of ``x`` [calls, D * S, H],
    ``words`` [calls, D, K, S] and ``fits`` [calls, 1]) back to back in one
    jit. Returns per device [N_DEV, calls, ...]: received buffers, source
    rows, send rows, plan group sizes and sentinels, route-plan token_sorted
    and group sizes, and the drain's table copy."""
    dp, tp = mesh.devices.shape
    recv_rows = dp * TOKENS * TOPK

    def per_device(x, words, fits):
        me = lax.axis_index(DP_AXIS)
        ep = (me * tp + lax.axis_index(TP_AXIS))[None]

        def step(_, inputs):
            i, x, words, fits = inputs
            if skew:
                x = _skew(x, me, i, dp)
            plan = sc_send_plan(
                words,
                ep,
                fits,
                local_num_experts=LOCAL,
                tp_size=tp,
                skip_padded=True,
                recv_rows=recv_rows,
            )
            _, inv = _fp8_scales(x)
            src = fp8_source_rows(x, inv)
            sent = copy_send_rows_fp8(
                x,
                inv,
                jnp.arange(TOKENS, dtype=jnp.int32),
                jnp.full((1,), TOKENS, jnp.int32),
                TOKENS,
            )
            recv, sems, send_sem = sc_dispatch(
                src,
                plan.table,
                axis=DP_AXIS,
                num_members=dp,
                num_waves=LOCAL,
                recv_rows=recv_rows,
                collective_id=COLLECTIVE_ID,
            )
            drained = sc_drain(send_sem, src, plan.table, axis=DP_AXIS)
            counts = jnp.zeros((TABLE_HEADER,), jnp.int32)
            counts = counts.at[:LOCAL].set(plan.group_sizes)
            counts = counts.at[LOCAL].set(plan.sentinels[0])
            out = _tc_receive(counts, recv, sems)
            rp = local_route_plan(
                words,
                ep,
                fits,
                local_num_experts=LOCAL,
                tp_size=tp,
                owner=True,
                skip_padded=True,
                rows=dp * TOKENS,
                routes=recv_rows,
            )
            return None, (
                out,
                src,
                sent,
                plan.group_sizes,
                plan.sentinels,
                rp.token_sorted,
                rp.group_sizes,
                drained,
            )

        calls = jnp.arange(x.shape[0], dtype=jnp.int32)
        outs = lax.scan(step, None, (calls, x, words, fits))[1]
        return tuple(o[None] for o in outs)

    f = jax.jit(
        jax.shard_map(
            per_device,
            mesh=mesh,
            in_specs=(P(None, DP_AXIS), P(), P()),
            out_specs=P((DP_AXIS, TP_AXIS)),
            check_vma=False,
        ),
        compiler_options=flags,
    )
    outs = f(
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(words, jnp.int32),
        jnp.asarray(fits, jnp.int32),
    )
    return [np.asarray(o) for o in jax.block_until_ready(outs)]


def _expected(words, ep, fits):
    """Receiver ``ep``'s computed routes as (source member, token) in route
    order, and its group sizes."""
    ids = words >> 16
    exponent = words & 0x7F80
    local = ids - ep * LOCAL
    computed = (
        (local >= 0)
        & (local < LOCAL)
        & (exponent != 0)
        & (exponent != 0x7F80)
        & bool(fits)
    )
    s, k, t = np.nonzero(computed)
    e = local[s, k, t]
    order = np.lexsort((k, t, s, e))
    return s[order], t[order], np.bincount(e, minlength=LOCAL), computed


def _in_row(words, ep):
    """The capacity exchange's receive row of each (member, token) reaching
    ``ep``."""
    local = (words >> 16) - ep * LOCAL
    reach = ((local >= 0) & (local < LOCAL)).any(axis=1)
    return (np.cumsum(reach.reshape(-1)) - 1).reshape(reach.shape)


def _check(mesh, words, fits, outs):
    dp, tp = mesh.devices.shape
    recv, src, sent, gs, sentinels, token_sorted, rp_gs, drained = outs
    np.testing.assert_array_equal(src, sent, err_msg="source rows")
    for dev in range(N_DEV):
        me, t = divmod(dev, tp)
        ep = me * tp + t
        for c in range(words.shape[0]):
            s_ord, t_ord, sizes, computed = _expected(words[c], ep, fits[c, 0])
            where = f"device {dev}, call {c}"
            np.testing.assert_array_equal(gs[dev, c], sizes, err_msg=where)
            np.testing.assert_array_equal(rp_gs[dev, c], sizes, err_msg=where)
            n = len(s_ord)
            in_row = _in_row(words[c], ep)
            np.testing.assert_array_equal(
                token_sorted[dev, c, :n], in_row[s_ord, t_ord], err_msg=where
            )
            # Member s's source rows, as its device on this TP rank built them.
            rows = src[s_ord * tp + t, c, t_ord]
            np.testing.assert_array_equal(recv[dev, c, :n], rows, err_msg=where)
            hits = np.zeros((dp,), bool)
            for r in range(dp):
                hits[r] = _expected(words[c], r * tp + t, fits[c, 0])[3][me].any()
            senders = np.array(
                [_expected(words[c], ep, fits[c, 0])[3][s].any() for s in range(dp)]
            )
            assert sentinels[dev, c, 0] == np.sum(~senders), where
            sends = sum(
                _expected(words[c], r * tp + t, fits[c, 0])[3][me].sum()
                for r in range(dp)
            )
            assert drained[dev, c, 0] == sends + np.sum(~hits), where


@pytest.mark.parametrize("dp", [4, 8])
@pytest.mark.parametrize("kind", ["uniform", "skewed", "padded", "repeated"])
def test_sc_dispatch_lands_rows_in_route_order(dp, kind):
    """One dispatch: each receiver gets every computed route's row at its
    route-plan position; waves and sentinels balance."""
    mesh = _mesh(dp)
    tp = N_DEV // dp
    rng = np.random.default_rng(dp * 7 + len(kind))
    words = _routing(rng, dp, tp, kind)[None]
    fits = np.ones((1, 1), np.int32)
    x = _tokens(rng, dp, 8192)[None]
    _check(mesh, words, fits, _run(mesh, x, words, fits, skew=False))


@pytest.mark.parametrize("dp", [4, 8])
def test_sc_dispatch_back_to_back_skewed(dp):
    """Six dispatches in one jit with routing and tokens varying by call and
    members entering each call at different times (one call overflows: only
    sentinels travel). The barrier and the wave semaphores are reused by
    every call."""
    mesh = _mesh(dp)
    tp = N_DEV // dp
    rng = np.random.default_rng(dp)
    kinds = ["skewed", "uniform", "padded", "skewed", "repeated", "uniform"]
    words = np.stack([_routing(rng, dp, tp, k) for k in kinds])
    fits = np.ones((len(kinds), 1), np.int32)
    fits[3] = 0
    x = np.stack([_tokens(rng, dp, 4096) for _ in kinds])
    _check(mesh, words, fits, _run(mesh, x, words, fits, skew=True))


def test_sc_dispatch_without_memory_flags():
    """The dispatch needs only the multicore remote DMA flag: its buffers are
    in HBM."""
    mesh = _mesh(4)
    rng = np.random.default_rng(3)
    words = _routing(rng, 4, 2, "uniform")[None]
    fits = np.ones((1, 1), np.int32)
    x = _tokens(rng, 4, 8192)[None]
    flags = dict(xla_mosaic_unsafe_allow_multicore_remote_dma=True)
    _check(mesh, words, fits, _run(mesh, x, words, fits, skew=False, flags=flags))
