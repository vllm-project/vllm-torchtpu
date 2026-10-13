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
"""The MTP drafter's sized DCP merge (`TPU_MTP_DCP_MERGE_SIZED`).

`sized_merge_local` runs inside a `shard_map` over the DCP axis; here the axis
is a `jax.vmap` axis, which gives the same collectives' semantics on one host.

  1. Both branches (packed all-to-all at small buckets, lse all-gather +
     reduce-scatter above) land within one bf16 ulp of the f64 merge, including
     `-inf` rows, a rank that owns nothing and NaN partials of weight 0.
  2. The all-to-all branch is bit for bit a host emulation of the
     reduce-scatter's per-element arithmetic summed in rank order: the pack
     moves every bit and the weights are the reduce-scatter's.
  3. Positive controls: rotating the packed lse rows by one rank, or summing
     unweighted, fails check 1.
  4. Routing: the drafter takes the op only with the flag set and an op built
     for its group size; the backbone never does.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

AXIS = "dcp"


def _partials(rng, world, tokens, local=4, vdim=128):
    heads = world * local
    o = rng.standard_normal((world, tokens, heads, vdim)).astype(np.float32)
    o = np.array(jnp.asarray(o).astype(jnp.bfloat16).astype(jnp.float32))  # bf16 values
    lse = (rng.standard_normal((world, tokens, heads)) * 4 + 3).astype(np.float32)
    lse[:, 0] += 200.0
    lse[:, 1] -= 200.0
    lse[1:, 2] = -np.inf  # rank 0 owns token 2 alone
    lse[:, 3] = -np.inf  # nobody owns token 3
    lse[world - 1] = -np.inf  # the last rank owns nothing
    o[world - 1, 5] = np.nan  # ...and left NaN where it ran on a dummy key
    return o, lse


def _run(o, lse, cap, fn=cp_mla.sized_merge_local):
    world = o.shape[0]

    def body(a, b):
        return fn(a, b, world=world, a2a_max_tokens=cap, axis_name=AXIS)

    out = jax.vmap(body, axis_name=AXIS)(jnp.asarray(o, jnp.bfloat16), jnp.asarray(lse))
    # [world, T, local, V] -> the whole [T, H, V]
    out = np.asarray(out.astype(jnp.float32))
    return np.concatenate(list(out), axis=1)


def _reference(o, lse):
    out = np.nan_to_num(o.astype(np.float64), nan=0.0)
    lse = lse.astype(np.float64)
    fin = np.isfinite(lse)
    shift = np.where(fin, lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    w = np.where(fin, np.exp(lse - shift[None]), 0.0)
    den = w.sum(axis=0)
    w = w / np.where(den > 0, den, 1.0)
    return (out * w[..., None]).sum(axis=0)


def _within_1ulp(got, ref):
    return bool(
        np.isfinite(got).all()
        and (np.abs(got - ref) <= 2.0**-8 * np.abs(ref) + 1e-6).all()
    )


def _host_rank_order(o, lse):
    """The reduce-scatter's per-element f32 arithmetic (via the same jnp ops),
    summed over ranks in order 0..world-1 on the host."""
    lse = jnp.asarray(lse)
    weighted = np.asarray(
        jax.vmap(lambda a, b: cp_mla._rs_weighted(a, b, lse))(
            jnp.asarray(o, jnp.bfloat16), lse
        )
    )
    acc = weighted[0]
    for r in range(1, weighted.shape[0]):
        acc = (acc + weighted[r]).astype(np.float32)
    return np.asarray(jnp.asarray(acc).astype(jnp.bfloat16).astype(jnp.float32))


@pytest.mark.parametrize("world", [2, 8, 16])
@pytest.mark.parametrize("tokens,cap", [(16, 64), (16, 8)])
def test_both_branches_within_one_ulp_of_f64(world, tokens, cap):
    o, lse = _partials(np.random.default_rng(world * 100 + cap), world, tokens)
    got = _run(o, lse, cap)
    assert _within_1ulp(got, _reference(o, lse))


@pytest.mark.parametrize("world", [8, 16])
def test_all_to_all_branch_is_rank_order_sum_of_rs_terms(world):
    o, lse = _partials(np.random.default_rng(7), world, 16)
    got = _run(o, lse, 64)
    want = _host_rank_order(o, lse)
    np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32))


def test_branches_differ_only_in_summation_order():
    """Reported, loosely bounded: the share of bf16 outputs the two branches
    round differently (the reduce-scatter here is the vmap psum_scatter)."""
    o, lse = _partials(np.random.default_rng(1012), 16, 16)
    a2a, rs = _run(o, lse, 64), _run(o, lse, 8)
    differ = float((a2a != rs).mean())
    print(f"a2a vs reduce-scatter branch: {differ:.4%} of outputs differ")
    assert differ < 0.01


def test_positive_control_rotated_lse_rows_are_caught(monkeypatch):
    o, lse = _partials(np.random.default_rng(3), 8, 16)
    real = cp_mla._unpack_words

    def rotated(recv, tokens, local, dtype):
        o_recv, l_recv = real(recv, tokens, local, dtype)
        return o_recv, jnp.roll(l_recv, 1, axis=0)

    monkeypatch.setattr(cp_mla, "_unpack_words", rotated)
    assert not _within_1ulp(_run(o, lse, 64), _reference(o, lse))


def test_positive_control_unweighted_sum_is_caught(monkeypatch):
    o, lse = _partials(np.random.default_rng(4), 8, 16)
    monkeypatch.setattr(
        cp_mla,
        "_rs_weighted",
        lambda a, own, lses: jnp.nan_to_num(a.astype(jnp.float32)) / a.shape[0],
    )
    assert not _within_1ulp(_run(o, lse, 64), _reference(o, lse))


class _Group:
    world_size = 4
    rank_in_group = 0


@pytest.mark.parametrize(
    "flag,is_backbone,built,expect",
    [
        (True, False, True, True),
        (True, True, True, False),  # the backbone keeps its own routing
        (False, False, True, False),
        (True, False, False, False),  # no op for this group size
    ],
)
def test_routing(flag, is_backbone, built, expect, monkeypatch):
    calls = []
    monkeypatch.setattr(cp_mla.envs, "TPU_MTP_DCP_MERGE_SIZED", flag, raising=False)
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A", False, raising=False)
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: _Group())
    monkeypatch.setattr(
        cp_mla,
        "_SIZED_MERGE_OPS",
        {4: lambda o, lse: calls.append(1) or o[:, :1]} if built else {},
    )
    monkeypatch.setattr(cp_mla, "cp_lse_ag_out_rs", lambda o, lse, g: o[:, :1])
    o = torch.zeros(16, 4, 8, dtype=torch.bfloat16)
    cp_mla.merge_lse_partials_scatter_heads(
        o, torch.zeros(16, 4), is_backbone=is_backbone
    )
    assert bool(calls) == expect
