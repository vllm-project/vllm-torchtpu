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
"""Which calls the small collectives take, and what prebuild records.

The kernels themselves are checked against XLA's collectives in
tests/kernels/test_small_all_reduce.py; here the ops are stand-ins.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("jax")
op = pytest.importorskip("vllm_torchtpu.layers.adapter.small_all_reduce_op")

from vllm_torchtpu.compilation import shape_variants  # noqa: E402

N = 8
HIDDEN = N * 128
CAP = 64
MESH_RANKS = tuple(range(N))
RANK_OF_CORE = (0, 1, 6, 7, 2, 3, 4, 5)


@torch.library.custom_op("test_small_collectives::all_reduce", mutates_args=())
def _kernel_all_reduce(x: torch.Tensor, mesh_index: torch.Tensor) -> torch.Tensor:
    return x * 2


@_kernel_all_reduce.register_fake
def _(x, mesh_index):
    return torch.empty_like(x)


@torch.library.custom_op("test_small_collectives::reduce_scatter", mutates_args=())
def _kernel_reduce_scatter(
    x: torch.Tensor,
    mesh_index: torch.Tensor,
    rank_in_mesh: torch.Tensor,
    core_of_rank: torch.Tensor,
) -> torch.Tensor:
    return x[: x.shape[0] // N] * 2


@_kernel_reduce_scatter.register_fake
def _(x, mesh_index, rank_in_mesh, core_of_rank):
    return x.new_empty((x.shape[0] // N, x.shape[1]))


def _fallback(x):
    return x + 1


def _fallback_reduce_scatter(x):
    return x[: x.shape[0] // N] + 1


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setenv("TPU_SMALL_ALL_REDUCE_MAX_ROWS", str(CAP))
    monkeypatch.setattr(op, "_ALL_REDUCE_OP", _kernel_all_reduce)
    monkeypatch.setattr(op, "_REDUCE_SCATTER_OP", _kernel_reduce_scatter)
    monkeypatch.setattr(op, "_NUM_CORES", N)
    monkeypatch.setattr(op, "_MESH_RANKS", MESH_RANKS)
    monkeypatch.setattr(op, "_MESH_INDEX", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_RANK_IN_MESH", torch.tensor([7], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORE_OF_RANK", torch.arange(N, dtype=torch.int32))


def _block(rows, dtype=torch.bfloat16, hidden=HIDDEN):
    return torch.ones(rows, hidden, dtype=dtype)


def test_unarmed_calls_keep_vllm_collectives():
    if op._ALL_REDUCE_OP is not None:
        pytest.skip("armed in this process")
    x = _block(8)
    assert torch.equal(op.all_reduce(x, MESH_RANKS, _fallback), _fallback(x))
    assert torch.equal(
        op.reduce_scatter(_block(N * 8), MESH_RANKS, _fallback_reduce_scatter),
        _fallback_reduce_scatter(_block(N * 8)),
    )


@pytest.mark.usefixtures("armed")
@pytest.mark.parametrize("rows,kernel", [(1, True), (CAP, True), (CAP + 1, False)])
def test_all_reduce_takes_the_kernel_up_to_the_cap(rows, kernel):
    x = _block(rows)
    want = x * 2 if kernel else _fallback(x)
    assert torch.equal(op.all_reduce(x, MESH_RANKS, _fallback), want)


@pytest.mark.usefixtures("armed")
@pytest.mark.parametrize(
    "rows_per_rank,kernel", [(1, True), (CAP, True), (CAP + 1, False)]
)
def test_reduce_scatter_cap_counts_rows_per_rank(rows_per_rank, kernel):
    x = _block(N * rows_per_rank)
    got = op.reduce_scatter(x, MESH_RANKS, _fallback_reduce_scatter)
    want = x[:rows_per_rank] * 2 if kernel else _fallback_reduce_scatter(x)
    assert torch.equal(got, want)


@pytest.mark.usefixtures("armed")
@pytest.mark.parametrize(
    "x",
    [
        _block(8, dtype=torch.float32),
        _block(8, hidden=HIDDEN + 128),  # does not split over 8 cores
        torch.ones(2, 4, HIDDEN, dtype=torch.bfloat16),
    ],
    ids=["float32", "width", "3d"],
)
def test_all_reduce_refuses_blocks_the_kernel_cannot_take(x):
    assert torch.equal(op.all_reduce(x, MESH_RANKS, _fallback), _fallback(x))


@pytest.mark.usefixtures("armed")
def test_groups_other_than_the_mesh_keep_vllm_collectives():
    x, gathered = _block(8), _block(N * 8)
    # a sum does not depend on the rank order, so any order of the mesh's ranks
    assert torch.equal(op.all_reduce(x, MESH_RANKS[::-1], _fallback), x * 2)
    assert torch.equal(op.all_reduce(x, (0, 1, 2, 3), _fallback), _fallback(x))
    # the reduce-scatter's blocks are in group-rank order: it must be the mesh's
    for ranks in (MESH_RANKS[::-1], (0, 1, 2, 3)):
        assert torch.equal(
            op.reduce_scatter(gathered, ranks, _fallback_reduce_scatter),
            _fallback_reduce_scatter(gathered),
        )


def _compile(fn, x, warmup):
    """The graph Dynamo captures for fn with the token dimension dynamic, and
    its output, inside or outside the backbone warmup."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend, dynamic=True, fullgraph=True)
    if warmup:
        with shape_variants.warmup():
            out = compiled(x)
    else:
        out = compiled(x)
    return graphs[0], out


def _lifted_int_inputs(graph):
    return sorted(
        tuple(node.meta["example_value"].shape)
        for node in graph.graph.nodes
        if node.op == "placeholder"
        and isinstance(node.meta.get("example_value"), torch.Tensor)
        and node.meta["example_value"].dtype == torch.int32
    )


@pytest.mark.usefixtures("armed")
def test_both_sides_of_the_row_cap_lift_the_same_buffers():
    """The warmup hands a bucket's executable between the traces on each side
    of the cap only when their inputs agree."""

    def combine(x):
        return op.reduce_scatter(x, MESH_RANKS, _fallback_reduce_scatter)

    def all_reduce(x):
        return op.all_reduce(x, MESH_RANKS, _fallback)

    for fn, fallback, small, large, buffers in (
        (combine, _fallback_reduce_scatter, N * 8, N * (CAP + 8), [(1,), (1,), (N,)]),
        (all_reduce, _fallback, 8, CAP + 8, [(1,)]),
    ):
        small_graph, small_out = _compile(fn, _block(small), warmup=True)
        large_graph, large_out = _compile(fn, _block(large), warmup=True)
        assert _lifted_int_inputs(small_graph) == buffers
        assert _lifted_int_inputs(large_graph) == buffers
        # the read that keeps the buffers live adds zero
        assert torch.equal(large_out, fallback(_block(large)))


@pytest.mark.usefixtures("armed")
def test_compiled_outside_the_warmup_keeps_vllm_collectives():
    """A module compiled outside the backbone warmup must not branch on the
    row count, so it never takes the kernel."""
    x = _block(8)
    graph, out = _compile(lambda t: op.all_reduce(t, MESH_RANKS, _fallback), x, False)
    assert torch.equal(out, _fallback(x))
    assert not any(
        "test_small_collectives" in str(node.target) for node in graph.graph.nodes
    )


def test_prebuild_records_this_rank_and_the_mesh_core_of_every_rank(monkeypatch):
    import numpy as np

    monkeypatch.setenv("TPU_SMALL_ALL_REDUCE_MAX_ROWS", str(CAP))
    monkeypatch.setenv("TPU_SMALL_ALL_REDUCE_FORM", "TwoShot")
    monkeypatch.setenv("TPU_SMALL_REDUCE_SCATTER_HIERARCHICAL", "1")
    for name in (
        "_ALL_REDUCE_OP",
        "_REDUCE_SCATTER_OP",
        "_MESH_INDEX",
        "_RANK_IN_MESH",
        "_CORE_OF_RANK",
    ):
        monkeypatch.setattr(op, name, None)
    dual_core_devs = np.asarray(
        [SimpleNamespace(id=i, device_kind="TPU7x") for i in range(N)], dtype=object
    )
    mesh = SimpleNamespace(shape={op.EP_AXIS_NAME: N}, devices=dual_core_devs)
    built = {}
    monkeypatch.setattr(op, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(
        op, "get_ep_group", lambda: SimpleNamespace(ranks=MESH_RANKS, rank_in_group=6)
    )
    monkeypatch.setattr(op, "ep_rank_order", lambda: RANK_OF_CORE)
    monkeypatch.setattr(op, "ep_mesh_index", lambda: RANK_OF_CORE.index(6))
    monkeypatch.setattr(
        op, "_build_all_reduce", lambda m, form: built.setdefault("form", form)
    )
    monkeypatch.setattr(
        op,
        "_build_reduce_scatter",
        lambda m, *, hierarchical=False: built.setdefault("hierarchical", hierarchical),
    )

    assert op.prebuild(torch.device("cpu"))
    assert built["form"] == "twoshot"
    assert built["hierarchical"] is True
    assert op._MESH_INDEX.tolist() == [2]
    assert op._RANK_IN_MESH.tolist() == [6]
    # rank r is served by mesh core core_of_rank[r]
    assert [RANK_OF_CORE[c] for c in op._CORE_OF_RANK.tolist()] == list(MESH_RANKS)
    assert op._MESH_RANKS == MESH_RANKS and op._NUM_CORES == N
    # idempotent: the second MoE layer finds it built
    monkeypatch.setattr(op, "build_ep_mesh", lambda: pytest.fail("rebuilt"))
    assert op.prebuild(torch.device("cpu"))


def test_prebuild_does_nothing_when_disabled(monkeypatch):
    monkeypatch.delenv("TPU_SMALL_ALL_REDUCE_MAX_ROWS", raising=False)
    monkeypatch.setattr(op, "_ALL_REDUCE_OP", None)
    monkeypatch.setattr(op, "build_ep_mesh", lambda: pytest.fail("mesh resolved"))
    assert not op.prebuild(torch.device("cpu"))


def test_builders_pass_sharded_output_partition_spec_to_jax_op(monkeypatch):
    """Verify output_partition_specs is passed so jax_op shards outputs."""
    recorded = []

    def fake_jax_op(name, fn, **kwargs):
        recorded.append((name, kwargs))
        return SimpleNamespace(register_fake=lambda f: f)

    monkeypatch.setattr(op.pallas, "jax_op", fake_jax_op)
    mesh = SimpleNamespace(shape={op.EP_AXIS_NAME: N})
    op._build_all_reduce(mesh, None)
    op._build_reduce_scatter(mesh)

    expected = op.PartitionSpec(op.EP_AXIS_NAME)
    assert len(recorded) == 2
    assert recorded[0][0] == op.ALL_REDUCE_OP_NAME
    assert recorded[0][1]["input_partition_specs"] == (expected, expected)
    assert recorded[0][1]["output_partition_specs"] == expected
    assert recorded[1][0] == op.REDUCE_SCATTER_OP_NAME
    assert recorded[1][1]["input_partition_specs"] == (expected,) * 4
    assert recorded[1][1]["output_partition_specs"] == expected


def test_build_reduce_scatter_gates_hierarchical_on_dual_core_mesh(monkeypatch):
    """_build_reduce_scatter enables the two-stage kernel only when both the
    knob is set and the mesh pairs dual-core chip mates."""
    import numpy as np

    seen_hierarchical = []

    def fake_shard_map(body_fn, **kwargs):
        def runner(x, mi, mr, cr):
            return body_fn(x, mi, mr, cr)

        return runner

    monkeypatch.setattr(op.jax, "shard_map", fake_shard_map)
    monkeypatch.setattr(
        op.sgs,
        "small_reduce_scatter",
        lambda block, **kw: seen_hierarchical.append(kw["hierarchical"]) or block,
    )

    captured_fns = []
    monkeypatch.setattr(
        op.pallas,
        "jax_op",
        lambda name, fn, **kw: captured_fns.append(fn)
        or SimpleNamespace(register_fake=lambda f: f),
    )

    dual_core_devs = np.asarray(
        [SimpleNamespace(id=i, device_kind="TPU7x") for i in range(N)], dtype=object
    )
    single_core_devs = np.asarray(
        [SimpleNamespace(id=i, device_kind="TPU v6e") for i in range(N)], dtype=object
    )

    op._build_reduce_scatter(
        SimpleNamespace(shape={op.EP_AXIS_NAME: N}, devices=dual_core_devs),
        hierarchical=True,
    )
    op._build_reduce_scatter(
        SimpleNamespace(shape={op.EP_AXIS_NAME: N}, devices=dual_core_devs),
        hierarchical=False,
    )
    op._build_reduce_scatter(
        SimpleNamespace(shape={op.EP_AXIS_NAME: N}, devices=single_core_devs),
        hierarchical=True,
    )
    for fn in captured_fns:
        fn("x", "mi", "mr", "cr")

    assert seen_hierarchical == [True, False, False]


def test_prebuild_warns_and_disables_hierarchical_on_ineligible_mesh(
    monkeypatch, caplog
):
    """prebuild warns once and passes hierarchical=False when the env knob is
    enabled on a single-core-per-chip mesh."""
    import logging

    import numpy as np

    monkeypatch.setenv("TPU_SMALL_ALL_REDUCE_MAX_ROWS", str(CAP))
    monkeypatch.setenv("TPU_SMALL_REDUCE_SCATTER_HIERARCHICAL", "1")
    monkeypatch.setattr(op, "_ALL_REDUCE_OP", None)
    monkeypatch.setattr(op, "_REDUCE_SCATTER_OP", None)
    v6e_devs = np.asarray(
        [SimpleNamespace(id=i, device_kind="TPU v6e") for i in range(N)],
        dtype=object,
    )
    monkeypatch.setattr(
        op,
        "build_ep_mesh",
        lambda: SimpleNamespace(shape={op.EP_AXIS_NAME: N}, devices=v6e_devs),
    )
    monkeypatch.setattr(
        op,
        "get_ep_group",
        lambda: SimpleNamespace(ranks=list(MESH_RANKS), rank_in_group=0),
    )
    monkeypatch.setattr(op, "ep_rank_order", lambda: RANK_OF_CORE)
    monkeypatch.setattr(op, "ep_mesh_index", lambda: 0)

    seen_hierarchical = []
    monkeypatch.setattr(op, "_build_all_reduce", lambda mesh, form: _kernel_all_reduce)
    monkeypatch.setattr(
        op,
        "_build_reduce_scatter",
        lambda mesh, hierarchical=False: seen_hierarchical.append(hierarchical)
        or _kernel_reduce_scatter,
    )

    with caplog.at_level(logging.INFO):
        assert op.prebuild(torch.device("cpu"))

    assert seen_hierarchical == [False]
    assert any(
        "falling back to one-stage reduce-scatter" in rec.getMessage()
        for rec in caplog.records
    )
    assert any(
        "hierarchical_reduce_scatter=False" in rec.getMessage()
        for rec in caplog.records
    )
