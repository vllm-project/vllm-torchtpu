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
"""shared_in_op: the fused EP layer adds the shared expert's TP-partial output.

Reference = what the runner computes with the option off: the routed layer's
output, scaled, plus the all-reduce of the shared partial over the TP replica
group. The all-reduce path (rows below shared_rs_min_rows) must match it bit
for bit; the reduce-scatter path changes the summation order and is held to a
tolerance, plus every replica holding identical rows. Eight local TPU devices,
permuted mesh order.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat, fused_ep_moe_v2

pytestmark = pytest.mark.multichip

HIDDEN = 512  # TP4 groups: 4 cores x 128 lanes divide it, so the small all-reduce kernel applies there
_TP8 = ((0, 1, 4, 5, 6, 7, 2, 3),)
_TP4_PERMUTED = ((7, 1, 4, 2), (6, 0, 5, 3))


@pytest.fixture(scope="module")
def mesh():
    devices = jax.local_devices()
    if len(devices) < 8 or devices[0].platform != "tpu":
        pytest.skip("requires eight local TPU devices")
    order = (3, 0, 6, 1, 7, 4, 2, 5)
    return Mesh(np.asarray([devices[i] for i in order]), (AXIS,))


def _inputs(mesh, groups, rows, hidden=HIDDEN):
    ep = mesh.shape[AXIS]
    owner = [0] * ep
    for i, group in enumerate(groups):
        for d in group:
            owner[d] = i
    rank = jax.device_put(
        np.arange(ep, dtype=np.int32).reshape(ep, 1), NamedSharding(mesh, P(AXIS))
    )

    def init(rank_l):
        me = rank_l[0, 0]
        gid = jnp.asarray(owner, dtype=jnp.int32)[me]
        tk = jax.random.split(jax.random.fold_in(jax.random.key(123), gid), 2)
        wk = jax.random.split(jax.random.fold_in(jax.random.key(456), me), 3)
        x = (jax.random.normal(tk[0], (rows, hidden)) * 0.2).astype(jnp.bfloat16)
        gate = jax.random.normal(tk[1], (rows, 2 * ep))
        # Each member of a group holds a different partial: the TP shards of a
        # row-parallel down projection.
        shared = (jax.random.normal(wk[2], (rows, hidden)) * 0.3).astype(jnp.bfloat16)

        def weights(key, width):
            raw = jax.random.normal(key, (2, hidden, width)) / jnp.sqrt(float(hidden))
            maximum = float(jnp.finfo(jnp.float8_e4m3fn).max)
            scale = jnp.maximum(jnp.max(jnp.abs(raw), axis=1, keepdims=True) / maximum, 1e-12)
            return (raw / scale).astype(jnp.float8_e4m3fn), scale[:, 0, :]

        w1, s1 = weights(wk[0], 2 * hidden)  # inter == hidden, so w2 [2, hidden, hidden] matches
        w2, s2 = weights(wk[1], hidden)
        return x, w1, w2, s1, s2, gate, shared

    out = jax.jit(
        jax.shard_map(init, mesh=mesh, in_specs=P(AXIS), out_specs=(P(AXIS),) * 7,
                      check_vma=False)
    )(rank)
    return rank, out[:6], out[6]


def _layer(mesh, rank, inputs, groups, shared=None, **kw):
    def run(rank_arg, shared_arg, *args):
        return fused_ep_moe_v2(
            *args,
            topk=4,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            weight_format=WeightFormat.FP8,
            rank=rank_arg,
            sharded_plan=True,
            token_replica_groups=groups,
            order_after=shared_arg,
            **kw,
        )

    return jax.block_until_ready(jax.jit(run)(rank, shared, *inputs))


def _reference(mesh, routed, shared, groups, scale):
    index_groups = [list(g) for g in groups]

    def body(r, s):
        r = r * jnp.asarray(scale, r.dtype) if scale != 1.0 else r
        return lax.psum(s, AXIS, axis_index_groups=index_groups) + r

    return jax.jit(
        jax.shard_map(body, mesh=mesh, in_specs=(P(AXIS), P(AXIS)), out_specs=P(AXIS),
                      check_vma=False)
    )(routed, shared)


def _exact(mesh, routed, shared, groups, scale):
    """f32 sum of the partials plus the scaled routed rows (rounded once at the
    end only when compared)."""
    index_groups = [list(g) for g in groups]

    def body(r, s):
        r32 = r.astype(jnp.float32) * jnp.float32(scale)
        return lax.psum(s.astype(jnp.float32), AXIS, axis_index_groups=index_groups) + r32

    return jax.jit(
        jax.shard_map(body, mesh=mesh, in_specs=(P(AXIS), P(AXIS)), out_specs=P(AXIS),
                      check_vma=False)
    )(routed, shared)


def _by_rank(arr, rows):
    return {s.index[0].start // rows: np.asarray(s.data).astype(np.float32)
            for s in arr.addressable_shards}


@pytest.mark.parametrize("groups", [_TP8, _TP4_PERMUTED], ids=["tp8", "tp4"])
@pytest.mark.parametrize("rows", [64, 256])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_single_host_shared_in_op(mesh, groups, rows, scale):
    rank, inputs, shared = _inputs(mesh, groups, rows)
    routed = _layer(mesh, rank, inputs, groups)
    want = _by_rank(_reference(mesh, routed, shared, groups, scale), rows)

    # All-reduce path: bit-exact with the runner's arithmetic.
    got_ar = _by_rank(
        _layer(mesh, rank, inputs, groups, shared, shared_in_op=True,
               shared_out_scale=scale, shared_rs_min_rows=rows + 1),
        rows,
    )
    for r in want:
        np.testing.assert_array_equal(got_ar[r], want[r], err_msg=f"rank {r}")

    # Reduce-scatter path: the summation order changes, so it is held to the
    # all-reduce path's own error against an exact (f32) sum, not to its bits.
    got_rs = _by_rank(
        _layer(mesh, rank, inputs, groups, shared, shared_in_op=True,
               shared_out_scale=scale, shared_rs_min_rows=1),
        rows,
    )
    exact = _by_rank(_exact(mesh, routed, shared, groups, scale), rows)
    for r in want:
        assert np.isfinite(got_rs[r]).all()
        norm = max(np.linalg.norm(exact[r]), 1e-20)
        err_rs = float(np.linalg.norm(got_rs[r] - exact[r]) / norm)
        err_ar = float(np.linalg.norm(want[r] - exact[r]) / norm)
        print(f"rank {r}: rel-L2 vs exact  rs {err_rs:.2e}  ar {err_ar:.2e}")
        assert err_rs < max(2.5 * err_ar, 2.0**-9), (r, err_rs, err_ar)
    for g in groups:
        for m in g[1:]:
            np.testing.assert_array_equal(got_rs[m], got_rs[g[0]])

    # Small all-reduce kernel path (TPU_SMALL_ALL_REDUCE at decode buckets):
    # f32 sum in group order, rounded once -- tolerance, identical per group.
    got_sar = _by_rank(
        _layer(mesh, rank, inputs, groups, shared, shared_in_op=True,
               shared_out_scale=scale, shared_rs_min_rows=rows + 1,
               shared_small_ar_max_rows=rows),
        rows,
    )
    for r in want:
        norm = max(np.linalg.norm(exact[r]), 1e-20)
        err_sar = float(np.linalg.norm(got_sar[r] - exact[r]) / norm)
        err_ar = float(np.linalg.norm(want[r] - exact[r]) / norm)
        assert err_sar < max(2.5 * err_ar, 2.0**-9), (r, err_sar, err_ar)
    for g in groups:
        for m in g[1:]:
            np.testing.assert_array_equal(got_sar[m], got_sar[g[0]])

    # Fault injection: dropping the shared term must be caught by the oracle.
    assert any(
        not np.allclose(_by_rank(_reference(mesh, routed, shared * 0, groups, scale), rows)[r],
                        want[r], rtol=1e-2, atol=1e-2)
        for r in want
    )


_DECODE_GLUE = dict(decode_combine=True, pallas_restore=True)


@pytest.mark.parametrize(
    "groups, rows",
    [(_TP8, 8), (_TP8, 16), (_TP8, 32), (_TP4_PERMUTED, 8), (_TP4_PERMUTED, 16)],
    ids=["tp8-r8", "tp8-r16", "tp8-r32", "tp4-r8", "tp4-r16"],
)
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_decode_fold_restore_into_shared_all_reduce(mesh, groups, rows, scale):
    # 1024 = 8 cores x 128 lanes: the small all-reduce applies at TP8 too.
    """shared_fold_restore: decode buckets with the Pallas restore and the
    small all-reduce. Routed rows join the shared partial before the one
    all-reduce, so this is held to the unfolded small-AR path's own error vs an
    exact f32 sum (not its bits), and every replica must hold identical rows."""
    rank, inputs, shared = _inputs(mesh, groups, rows, hidden=1024)
    routed = _layer(mesh, rank, inputs, groups, **_DECODE_GLUE)
    exact = _by_rank(_exact(mesh, routed, shared, groups, scale), rows)
    common = dict(shared_in_op=True, shared_out_scale=scale,
                  shared_rs_min_rows=rows + 1, shared_small_ar_max_rows=rows,
                  **_DECODE_GLUE)
    base = _by_rank(_layer(mesh, rank, inputs, groups, shared, **common), rows)
    fold = _by_rank(
        _layer(mesh, rank, inputs, groups, shared, shared_fold_restore=True, **common),
        rows,
    )
    same = all(np.array_equal(fold[r], base[r]) for r in base)
    for r in base:
        assert np.isfinite(fold[r]).all()
        norm = max(np.linalg.norm(exact[r]), 1e-20)
        err_fold = float(np.linalg.norm(fold[r] - exact[r]) / norm)
        err_base = float(np.linalg.norm(base[r] - exact[r]) / norm)
        print(f"rank {r}: rel-L2 vs exact fold {err_fold:.2e} base {err_base:.2e} "
              f"maxabs fold-base {np.abs(fold[r] - base[r]).max():.3e}")
        assert err_fold < max(2.5 * err_base, 2.0**-9), (r, err_fold, err_base)
    print(f"bit-identical to the unfolded path: {same}")
    # The fold must have run: one fewer bf16 rounding changes some bits.
    assert not same, "fold path did not run (bits identical to the unfolded path)"
    for g in groups:
        for m in g[1:]:
            np.testing.assert_array_equal(fold[m], fold[g[0]])
    # Fault injection: the fold without the routed rows must be caught.
    zero_routed = _by_rank(_exact(mesh, routed * 0, shared, groups, scale), rows)
    assert any(
        not np.allclose(zero_routed[r], exact[r], rtol=1e-2, atol=1e-2) for r in exact
    )
