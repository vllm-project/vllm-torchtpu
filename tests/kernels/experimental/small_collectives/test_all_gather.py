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
"""The fused small all-gather pair kernel: CPU padding/placement checks and
8-core TPU shard_map verification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_gather as ag,
)

from .oracle import AXIS, N, blocks


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows", [1, 4, 8, 16, 20, 24])
@pytest.mark.parametrize("rank_operand", [False, True])
@pytest.mark.parametrize("placement", ["identity", "static_order", "my_rank"])
def test_all_gather_pair_padding_and_rank_placement(
    monkeypatch, rows, rank_operand, placement
):
    with jax.default_device(jax.devices("cpu")[0]):
        n, hidden, num_experts = 8, 128, 512
        x = (
            jnp.arange(rows * hidden, dtype=jnp.float32)
            .reshape(rows, hidden)
            .astype(jnp.bfloat16)
        )
        logits = jnp.arange(rows * num_experts, dtype=jnp.float32).reshape(
            rows, num_experts
        )
        rank_of_core = (0, 1, 6, 7, 2, 3, 4, 5)
        mesh_core = 2
        own_rank = (
            rank_of_core[mesh_core]
            if placement != "identity"
            else mesh_core
        )
        rank = jnp.array([mesh_core], jnp.int32) if rank_operand else None
        my_rank = (
            jnp.array([own_rank], jnp.int32) if placement == "my_rank" else None
        )
        static_order = rank_of_core if placement == "static_order" else None
        calls = []

        def pallas_call(
            kernel, *, out_shape, scratch_shapes, compiler_params, **kwargs
        ):
            padded_rows = out_shape[0].shape[1]
            assert padded_rows % ag.SUBLANES == 0
            assert out_shape[0].shape == (n, padded_rows, hidden)
            assert out_shape[1].shape == (n, padded_rows, num_experts)
            assert scratch_shapes[0].shape == (n, padded_rows, hidden)
            assert scratch_shapes[1].shape == (n, padded_rows, num_experts)
            assert compiler_params.collective_id == ag.COLLECTIVE_ID

            def run(*args):
                x_in, logits_in = args[-2], args[-1]
                np.testing.assert_array_equal(x_in[:rows], x)
                np.testing.assert_array_equal(x_in[rows:], 0)
                np.testing.assert_array_equal(logits_in[:rows], logits)
                np.testing.assert_array_equal(logits_in[rows:], 0)
                if rows % ag.SUBLANES == 0:
                    assert x_in is x
                    assert logits_in is logits
                if rank_operand:
                    np.testing.assert_array_equal(args[0], rank)
                if placement == "my_rank":
                    np.testing.assert_array_equal(args[-3], my_rank)
                calls.append(True)
                out_x = jnp.zeros((n, padded_rows, hidden), dtype=x.dtype)
                out_l = jnp.zeros(
                    (n, padded_rows, num_experts), dtype=logits.dtype
                )
                return (
                    out_x.at[own_rank].set(x_in),
                    out_l.at[own_rank].set(logits_in),
                )

            return run

        monkeypatch.setattr(ag.pl, "pallas_call", pallas_call)
        got_x, got_l = ag.small_all_gather_pair(
            x,
            logits,
            num_cores=n,
            rank=rank,
            my_rank=my_rank,
            rank_of_core=static_order,
        )
        assert len(calls) == 1
        assert got_x.shape == (n * rows, hidden)
        assert got_l.shape == (n * rows, num_experts)
        np.testing.assert_array_equal(
            got_x.reshape(n, rows, hidden)[own_rank], x
        )
        np.testing.assert_array_equal(
            got_l.reshape(n, rows, num_experts)[own_rank], logits
        )


@pytest.mark.cpu_test
def test_all_gather_pair_validation_and_single_core():
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.ones((3, 128), dtype=jnp.bfloat16)
        logits = jnp.ones((3, 512), dtype=jnp.float32)
        got_x, got_l = ag.small_all_gather_pair(x, logits, num_cores=1)
        assert got_x is x
        assert got_l is logits
        with pytest.raises(ValueError, match="row mismatch"):
            ag.small_all_gather_pair(
                x, jnp.ones((4, 512), dtype=jnp.float32), num_cores=8
            )
        with pytest.raises(ValueError, match="not a multiple"):
            ag.small_all_gather_pair(
                jnp.ones((4, 100), dtype=jnp.bfloat16),
                jnp.ones((4, 512), dtype=jnp.float32),
                num_cores=8,
            )
        with pytest.raises(ValueError, match="not a multiple"):
            ag.small_all_gather_pair(
                jnp.ones((4, 128), dtype=jnp.bfloat16),
                jnp.ones((4, 500), dtype=jnp.float32),
                num_cores=8,
            )
        with pytest.raises(ValueError, match="at most one"):
            ag.small_all_gather_pair(
                x,
                logits,
                num_cores=8,
                my_rank=jnp.array([0], jnp.int32),
                rank_of_core=tuple(range(8)),
            )
        with pytest.raises(ValueError, match="permutation"):
            ag.small_all_gather_pair(
                x, logits, num_cores=8, rank_of_core=(0, 1, 2)
            )


def _run_all_gather_sharded(
    mesh,
    per_core_x,
    per_core_logits,
    ranks,
    *,
    rank_operand=True,
    order=None,
    my_ranks=None,
):
    n, rows, hidden = per_core_x.shape
    num_experts = per_core_logits.shape[2]
    spec = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(jnp.asarray(per_core_x).reshape(n * rows, hidden), spec)
    lg = jax.device_put(
        jnp.asarray(per_core_logits).reshape(n * rows, num_experts), spec
    )
    r = jax.device_put(jnp.asarray(ranks, dtype=jnp.int32), spec)
    mr = (
        jax.device_put(jnp.asarray(my_ranks, dtype=jnp.int32), spec)
        if my_ranks is not None
        else r
    )

    def body(x_s, lg_s, r_s, mr_s):
        return ag.small_all_gather_pair(
            x_s,
            lg_s,
            num_cores=N,
            axis=AXIS,
            rank=r_s if rank_operand else None,
            my_rank=mr_s if my_ranks is not None else None,
            rank_of_core=order,
        )

    fn = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS), P(AXIS), P(AXIS)),
            out_specs=(P(AXIS), P(AXIS)),
            check_vma=False,
        )
    )
    out_x, out_lg = fn(x, lg, r, mr)
    return (
        np.asarray(out_x).reshape(n, n * rows, hidden),
        np.asarray(out_lg).reshape(n, n * rows, num_experts),
    )


@pytest.mark.multichip
@pytest.mark.parametrize("rows", [1, 3, 8, 12, 24, 64])
@pytest.mark.parametrize("logits_dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("rank_operand", [False, True])
@pytest.mark.parametrize("mode", ["identity", "static_order", "my_rank"])
def test_each_core_receives_all_blocks_in_rank_order(
    mesh, rows, logits_dtype, rank_operand, mode
):
    per_core_x = blocks((N, rows, 4096), seed=200 + rows)
    rng = np.random.default_rng(300 + rows)
    per_core_lg = rng.standard_normal((N, rows, 512), dtype=np.float32).astype(
        logits_dtype
    )
    perm = (0, 1, 6, 7, 2, 3, 4, 5)
    order = perm if mode == "static_order" else None
    my_ranks = perm if mode == "my_rank" else None
    got_x, got_lg = _run_all_gather_sharded(
        mesh,
        per_core_x,
        per_core_lg,
        range(N),
        rank_operand=rank_operand,
        order=order,
        my_ranks=my_ranks,
    )
    rank_of_core = perm if mode != "identity" else tuple(range(N))
    core_of_rank = [rank_of_core.index(r) for r in range(N)]
    want_x = np.concatenate([per_core_x[core_of_rank[r]] for r in range(N)])
    want_lg = np.concatenate([per_core_lg[core_of_rank[r]] for r in range(N)])
    for core in range(N):
        np.testing.assert_array_equal(got_x[core], want_x)
        np.testing.assert_array_equal(got_lg[core], want_lg)
