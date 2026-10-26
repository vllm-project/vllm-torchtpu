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
"""shared_defer_ar (TPU_DSV4_MHC_MOE_AR_SEAM): up to the cap the op returns a
partial whose small all-reduce over the replica group -- the one the next
layer's mHC seam op runs, same form / group order -- is the non-deferred
op's output BIT FOR BIT: on the fold path (the pre-all-reduce block handed
over untouched) and on every other path (output on slot 0, zeros elsewhere).
Above the cap the op returns the reduced rows unchanged."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar
from vllm_torchtpu.kernels.fused_moe.v2 import AXIS

from .test_fused_ep_moe_v2_shared_in_op import (  # noqa: F401
    _inputs,
    _layer,
    mesh,
)

pytestmark = pytest.mark.multichip

_TP8 = ((0, 1, 4, 5, 6, 7, 2, 3),)
_TP4_PAIRS = ((0, 1, 4, 5), (2, 3, 6, 7))
_TP4_PERMUTED = ((7, 1, 4, 2), (6, 0, 5, 3))
_DECODE_GLUE = dict(decode_combine=True, pallas_restore=True)
CAP = 32


def _tables(mesh, groups):
    ep = mesh.shape[AXIS]
    rank = np.zeros((ep, 1), np.int32)
    peers = np.zeros((ep, len(groups[0])), np.int32)
    for g in groups:
        for pos, core in enumerate(g):
            rank[core], peers[core] = pos, g
    spec = NamedSharding(mesh, P(AXIS))
    return jax.device_put(rank, spec), jax.device_put(peers.reshape(-1), spec)


def _seam_sum(mesh, partial, groups, drop_own=False):
    """The seam op's all-reduce: small_all_reduce over the group, group_form."""
    n = len(groups[0])
    form, chunks, skip = ar.group_form(tuple(tuple(g) for g in groups))
    rank, peers = _tables(mesh, groups)

    def body(x, r, p):
        if drop_own:  # positive control: a partial missing this core's rows
            x = jnp.zeros_like(x)
        return ar.small_all_reduce(x, num_cores=n, axis=AXIS, rank=r, peers=p, form=form,
                                   chunks=chunks, skip_device_barrier=skip)

    return jax.block_until_ready(jax.jit(jax.shard_map(
        body, mesh=mesh, in_specs=(P(AXIS),) * 3, out_specs=P(AXIS), check_vma=False,
    ))(partial, rank, peers))


def _bits(a):
    return np.asarray(a).view(np.uint16)


@pytest.mark.parametrize(
    "groups,pairs,hidden",
    [(_TP8, 0, 1024), (_TP4_PAIRS, 2, 1024), (_TP4_PAIRS, 0, 512), (_TP4_PERMUTED, 0, 512)],
    ids=["tp8", "tp4pairs", "tp4", "tp4perm"],
)
@pytest.mark.parametrize("fold", [1, 2])
@pytest.mark.parametrize("rows", [4, 8, 16, 24, 32, 64])
def test_defer_partial_sums_to_output(mesh, monkeypatch, groups, pairs, hidden, fold, rows):
    monkeypatch.setattr(envs, "TPU_SMALL_ALL_REDUCE_PAIRS", pairs, raising=False)
    rank, inputs, shared = _inputs(mesh, groups, rows, hidden)
    common = dict(shared_in_op=True, shared_out_scale=2.5, shared_rs_min_rows=1024,
                  shared_small_ar_max_rows=CAP, shared_fold_restore=fold, **_DECODE_GLUE)
    base = _layer(mesh, rank, inputs, groups, shared, **common)
    part = _layer(mesh, rank, inputs, groups, shared, shared_defer_ar=CAP, **common)
    assert part.shape == base.shape
    if rows > CAP:
        assert np.array_equal(_bits(part), _bits(base)), "above the cap: reduced rows"
        return
    summed = _seam_sum(mesh, part, groups)
    diff = int((_bits(summed) != _bits(base)).sum())
    assert diff == 0, (diff, base.size)
    # The partial is not already the output (the all-reduce really is deferred).
    assert not np.array_equal(_bits(part), _bits(base))
    # The check can fail: a partial missing one core's contribution.
    bad = _seam_sum(mesh, part, groups, drop_own=True)
    assert not np.array_equal(_bits(bad), _bits(base))
