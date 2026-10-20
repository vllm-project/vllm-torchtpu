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
"""SPMD token-replica layout contract, including noncontiguous TP groups.

Runs on an eight-device CPU mesh with
XLA_FLAGS=--xla_force_host_platform_device_count=8; the fused FP8/FP4 TPU comparison is
in test_fused_ep_moe_v2_tp_tokens.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import (
    TokenReplicaLayout,
    replica_wire_rows,
)


@pytest.mark.parametrize("groups", [None, tuple((i,) for i in range(8))])
@pytest.mark.parametrize("rows", [1, 7, 17, 257])
def test_singleton_preserves_unaligned_inputs_without_traced_operations(groups, rows):
    layout = TokenReplicaLayout.create(
        ep=8, input_rows=rows, groups=groups, row_alignment=8
    )
    values = jnp.arange(rows * 4, dtype=jnp.float32).reshape(rows, 4)

    def identity_path(x, rank):
        return layout.restore(layout.split(x, rank), "unbound_axis")

    # No mesh is bound: a collective here would fail. The graph must also
    # preserve the exact input shape without padding, slicing or barriers.
    graph = jax.make_jaxpr(identity_path)(values, jnp.int32(0))
    assert not graph.jaxpr.eqns
    np.testing.assert_array_equal(identity_path(values, jnp.int32(0)), values)
    assert layout.local_rows == rows


@pytest.mark.multichip
@pytest.mark.parametrize(
    "groups",
    [
        None,
        ((0, 1), (2, 3), (4, 5), (6, 7)),
        ((6, 1), (4, 3), (0, 7), (2, 5)),
        ((7, 1, 4, 2), (6, 0, 5, 3)),
    ],
)
@pytest.mark.parametrize("rows", [1, 7, 24, 256])
@pytest.mark.parametrize("alignment", [1, 8])
def test_unique_rows_and_original_output_on_every_tp_rank(groups, rows, alignment):
    if len(jax.devices()) != 8:
        pytest.skip("requires eight devices; CPU virtual devices are sufficient")
    layout = TokenReplicaLayout.create(
        ep=8, input_rows=rows, groups=groups, row_alignment=alignment
    )
    mesh = Mesh(np.asarray(jax.devices()), ("d",))
    group_ids = {rank: i for i, group in enumerate(layout.groups) for rank in group}
    original = np.concatenate(
        [
            (
                np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
                + 100 * group_ids[r]
            )
            / 100
            for r in range(8)
        ]
    )
    x = jax.device_put(original, NamedSharding(mesh, P("d")))
    rank = jax.device_put(
        np.arange(8, dtype=np.int32).reshape(8, 1), NamedSharding(mesh, P("d"))
    )

    def local(x_l, rank_l):
        piece = layout.split(x_l, rank_l[0, 0])
        out = jnp.sin(piece) + piece * piece
        return layout.restore(out, "d"), piece

    fn = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("d"), P("d")),
            out_specs=(P("d"), P("d")),
            check_vma=False,
        )
    )
    restored, pieces = fn(x, rank)
    np.testing.assert_allclose(
        restored, np.sin(original) + original * original, rtol=1e-6, atol=1e-6
    )
    per_rank = {s.device.id: np.asarray(s.data) for s in pieces.addressable_shards}
    for group in layout.groups:
        recovered = np.concatenate([per_rank[r] for r in group])[:rows]
        np.testing.assert_array_equal(
            recovered, original[group[0] * rows : (group[0] + 1) * rows]
        )
    assert (
        layout.local_rows * len(layout.groups[0]) - rows
        < len(layout.groups[0]) * alignment
    )


@pytest.mark.parametrize(
    "groups", [(), ((0, 1),), ((0, 0), (2, 3)), ((0,), (1, 2, 3)), ((0, 1), (2, 4))]
)
def test_invalid_replica_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        TokenReplicaLayout.create(ep=4, input_rows=16, groups=groups)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("alignment", [1, 2, 8, 16])
def test_wire_rows_bound_the_layout_and_match_it_from_one_block(width, alignment):
    """Never fewer rows than a member computes; the same from one block up."""
    groups = (
        None
        if width == 1
        else tuple(tuple(range(g * width, (g + 1) * width)) for g in range(2))
    )
    for rows in range(1, 4097):
        local = TokenReplicaLayout.create(
            ep=2 * width, input_rows=rows, groups=groups, row_alignment=alignment
        ).local_rows
        wire = replica_wire_rows(rows, width, alignment)
        assert wire >= local
        if width == 1 or -(-rows // width) > alignment // 2:
            assert wire == local


def test_wire_rows_leave_a_symbolic_row_count_unguarded():
    """One graph serves every token count, as vLLM's MTP drafter requires.

    The route op's fake sizes its outputs with replica_wire_rows over a
    symbolic row count; any comparison on it would bound the traced graph.
    """
    import torch
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()

    @torch.compile(backend=counter, dynamic=True, fullgraph=True)
    def fake(x):
        return x.new_empty((replica_wire_rows(x.shape[0], 16, 8),))

    for rows in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 100, 7):
        assert fake(torch.empty(rows)).shape[0] == replica_wire_rows(rows, 16, 8)
    assert counter.frame_count == 1
