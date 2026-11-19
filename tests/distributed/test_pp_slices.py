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
"""A pipeline over several TPU slices: its layout, and the stand-ins for
vLLM's pipeline group and inner data-parallel world on its workers."""

import threading
from types import SimpleNamespace

import pytest
from vllm.config import ParallelConfig
from vllm.distributed import parallel_state
from vllm.distributed.device_communicators import shm_broadcast

from tests.distributed.store_test_utils import TIMEOUT, FakeStore
from vllm_torchtpu.distributed import mesh_utils, pp_slices
from vllm_torchtpu.distributed.pp_slices import PipelineSlices


def _parallel_config(pp, nnodes, world_size):
    return SimpleNamespace(
        pipeline_parallel_size=pp, nnodes=nnodes, world_size=world_size
    )


def test_one_slice_has_no_layout(monkeypatch):
    monkeypatch.delenv("TPU_PP_SLICES", raising=False)
    assert pp_slices.pipeline_slices(_parallel_config(4, 2, 4)) is None
    monkeypatch.setenv("TPU_PP_SLICES", "1")
    assert pp_slices.pipeline_slices(_parallel_config(4, 2, 4)) is None
    assert pp_slices.pipeline_slices(_parallel_config(1, 2, 16)) is None


def test_the_layout_cuts_stages_hosts_and_ranks_slice_major(monkeypatch):
    monkeypatch.setenv("TPU_PP_SLICES", "2")
    # Four stages of eight workers on four hosts of eight: two hosts and two
    # stages a slice.
    layout = pp_slices.pipeline_slices(_parallel_config(4, 4, 32))
    assert layout == PipelineSlices(ranks=(16, 16), stages=4, nnodes=4)
    assert (
        layout.stages_of(1),
        layout.world_size_of(1),
        layout.hosts_of(1),
        layout.stage_width,
    ) == (2, 16, 2, 8)
    assert [layout.slice_of_rank(r) for r in (0, 15, 16, 31)] == [0, 0, 1, 1]
    assert [layout.slice_of_node(n) for n in range(4)] == [0, 0, 1, 1]
    assert [layout.slice_start(r) for r in (0, 15, 16, 31)] == [0, 0, 16, 16]


@pytest.mark.parametrize(
    ("slices", "pp", "nnodes", "message"),
    [
        ("3", 4, 3, r"TPU_PP_SLICES=3 must divide --pipeline-parallel-size \(4\)"),
        ("2", 4, 3, r"TPU_PP_SLICES=2 must divide --nnodes \(3\)"),
        ("2", 4, 1, r"TPU_PP_SLICES=2 must divide --nnodes \(1\)"),
        # an engine without a pipeline
        ("2", 1, 2, r"TPU_PP_SLICES=2 must divide --pipeline-parallel-size \(1\)"),
        ("0", 4, 2, r"TPU_PP_SLICES must be at least 1, got 0\."),
        ("-1", 4, 2, r"TPU_PP_SLICES must be at least 1, got -1\."),
    ],
)
def test_slices_of_unequal_stages_or_hosts_are_refused(
    monkeypatch, slices, pp, nnodes, message
):
    monkeypatch.setenv("TPU_PP_SLICES", slices)
    with pytest.raises(ValueError, match=message):
        pp_slices.pipeline_slices(_parallel_config(pp, nnodes, pp))


@pytest.mark.parametrize(
    ("ranks", "stages", "nnodes", "per_slice", "node_slices"),
    [
        # stages (4, 2) of one device on six hosts of one rank
        ((4, 2), 6, 6, [(4, 4, 0, 0, 0), (2, 2, 4, 4, 4)], [0, 0, 0, 0, 1, 1]),
        # the same stages on three hosts of two ranks
        ((4, 2), 6, 3, [(4, 2, 0, 0, 0), (2, 1, 4, 4, 2)], [0, 0, 1]),
        # stages (2, 1) of four devices, a stage a host
        ((8, 4), 3, 3, [(2, 2, 0, 0, 0), (1, 1, 8, 2, 2)], [0, 0, 1]),
        # the same stages, each spanning two hosts of two ranks
        ((8, 4), 3, 6, [(2, 4, 0, 0, 0), (1, 2, 8, 2, 4)], [0, 0, 0, 0, 1, 1]),
    ],
    ids=["stage_a_host", "two_stages_a_host", "stage_a_host_of_4", "stage_on_2_hosts"],
)
def test_slices_of_different_sizes_follow_from_their_ranks(
    ranks, stages, nnodes, per_slice, node_slices
):
    layout = PipelineSlices(ranks=ranks, stages=stages, nnodes=nnodes)
    # (stages, hosts, first rank, first stage, first node) of each slice
    assert [
        (
            layout.stages_of(s),
            layout.hosts_of(s),
            layout.first_rank_of(s),
            layout.first_stage_of(s),
            layout.first_node_of(s),
        )
        for s in range(layout.slices)
    ] == per_slice
    assert [layout.slice_of_node(n) for n in range(nnodes)] == node_slices
    # The engine ranks of a slice are the ranks on its hosts.
    per_host = layout.devices_per_host
    for rank in range(layout.world_size):
        assert layout.slice_of_rank(rank) == node_slices[rank // per_host]
        assert layout.slice_start(rank) == layout.first_rank_of(
            layout.slice_of_rank(rank)
        )


@pytest.mark.parametrize(
    ("text", "pp", "nnodes", "world_size", "ranks"),
    [
        ("4,2", 6, 6, 6, (4, 2)),
        ("4,2", 6, 3, 6, (4, 2)),
        ("2,1", 3, 3, 12, (8, 4)),
        ("1,2,1", 4, 4, 8, (2, 4, 2)),
        # a trailing comma ends the list; spaces around an entry are fine
        ("4, 2,", 6, 6, 6, (4, 2)),
    ],
)
def test_a_list_gives_the_stages_of_each_slice(
    monkeypatch, text, pp, nnodes, world_size, ranks
):
    monkeypatch.setenv("TPU_PP_SLICES", text)
    layout = pp_slices.pipeline_slices(_parallel_config(pp, nnodes, world_size))
    assert layout == PipelineSlices(ranks=ranks, stages=pp, nnodes=nnodes)


@pytest.mark.parametrize(
    ("number", "listed", "pp", "nnodes", "world_size"),
    [("2", "2,2", 4, 4, 32), ("3", "2,2,2", 6, 3, 6), ("2", "8,8", 16, 2, 16)],
)
def test_a_number_and_a_list_of_equal_slices_agree(
    monkeypatch, number, listed, pp, nnodes, world_size
):
    config = _parallel_config(pp, nnodes, world_size)
    monkeypatch.setenv("TPU_PP_SLICES", number)
    equal = pp_slices.pipeline_slices(config)
    monkeypatch.setenv("TPU_PP_SLICES", listed)
    assert pp_slices.pipeline_slices(config) == equal


@pytest.mark.parametrize(
    ("text", "pp", "nnodes", "message"),
    [
        ("4,", 6, 6, r"TPU_PP_SLICES=4, lists one slice.*unset TPU_PP_SLICES"),
        ("4,0", 4, 4, r"TPU_PP_SLICES=4,0: every slice holds at least 1 stage"),
        ("4,-1", 3, 3, r"every slice holds at least 1 stage"),
        ("3,2", 6, 6, r"lists 5 stages.*--pipeline-parallel-size \(6\)"),
        # an engine without a pipeline
        ("1,1", 1, 2, r"lists 2 stages.*--pipeline-parallel-size \(1\)"),
        # stages of one rank on two hosts of three
        ("4,2", 6, 2, r"slice 0 holds 4 engine ranks.*every host runs 3 workers"),
        ("two", 4, 2, r"TPU_PP_SLICES='two' is neither a number"),
        ("4;2", 6, 6, r"TPU_PP_SLICES='4;2' is neither a number"),
        ("4,,2", 6, 6, r"is neither a number"),
        ("4.0,2", 6, 6, r"is neither a number"),
        (",", 6, 6, r"is neither a number"),
    ],
)
def test_a_list_that_does_not_cut_the_pipeline_is_refused(
    monkeypatch, text, pp, nnodes, message
):
    monkeypatch.setenv("TPU_PP_SLICES", text)
    with pytest.raises(ValueError, match=message):
        pp_slices.pipeline_slices(_parallel_config(pp, nnodes, pp))


class _InSlice(SimpleNamespace):
    """vLLM's pipeline group of a worker: its column over its slice's
    stages, in ranks of the slice's torch world."""

    def destroy(self):
        self.destroyed = True


def _in_slice(layout, engine_rank):
    local = engine_rank - layout.slice_start(engine_rank)
    width = layout.stage_width
    stages = layout.stages_of(layout.slice_of_rank(engine_rank))
    return _InSlice(
        rank_in_group=local // width,
        world_size=stages,
        ranks=[stage * width + local % width for stage in range(stages)],
        destroyed=False,
    )


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("slices", [2, 4])
def test_the_group_reports_the_place_in_the_whole_pipeline(slices, width):
    stages = 8
    layout = PipelineSlices(
        ranks=(stages * width // slices,) * slices, stages=stages, nnodes=slices
    )
    column = width - 1
    # first, middle and last stages of every slice
    for stage in range(stages):
        rank = stage * width + column
        group = pp_slices.SlicedPipelineGroup(
            _in_slice(layout, rank), layout, rank, FakeStore()
        )
        assert group.rank == rank
        assert (group.rank_in_group, group.world_size) == (stage, stages)
        assert group.ranks == [s * width + column for s in range(stages)]
        assert (group.first_rank, group.last_rank) == (
            column,
            (stages - 1) * width + column,
        )
        assert group.is_first_rank == (stage == 0)
        assert group.is_last_rank == (stage == stages - 1)


def test_the_group_reports_the_place_of_a_rank_on_a_smaller_slice():
    # stages (2, 1) of four ranks: engine rank 9 is column 1 of the one stage
    # of slice 1, the last of the pipeline
    layout = PipelineSlices(ranks=(8, 4), stages=3, nnodes=3)
    in_slice = _in_slice(layout, 9)
    assert (in_slice.rank_in_group, in_slice.world_size) == (0, 1)
    group = pp_slices.SlicedPipelineGroup(in_slice, layout, 9, FakeStore())
    assert (group.rank_in_group, group.world_size) == (2, 3)
    assert group.ranks == [1, 5, 9]
    assert group.is_last_rank and not group.is_first_rank


def test_the_group_has_no_torch_collectives_and_destroys_vllms_group():
    layout = PipelineSlices(ranks=(2, 2), stages=4, nnodes=2)
    in_slice = _in_slice(layout, 2)
    group = pp_slices.SlicedPipelineGroup(in_slice, layout, 2, FakeStore())
    for name in ("broadcast_tensor_dict", "send_tensor_dict", "cpu_group", "next_rank"):
        with pytest.raises(AttributeError):
            getattr(group, name)
    group.destroy()
    assert in_slice.destroyed


@pytest.mark.parametrize("src", [0, 3])
def test_a_broadcast_reaches_every_stage_of_its_column_alone(src):
    """Each column broadcasts through the store on its own: the stage at
    ``src`` passes the object and every other stage of that column, on either
    slice, gets it, call after call in order."""
    layout = PipelineSlices(ranks=(4, 4), stages=4, nnodes=2)
    store = FakeStore()
    groups = {
        rank: pp_slices.SlicedPipelineGroup(
            _in_slice(layout, rank), layout, rank, store
        )
        for rank in range(layout.world_size)
    }
    got: dict[int, list] = {rank: [] for rank in groups}

    def run(rank):
        group = groups[rank]
        column = rank % layout.stage_width
        for call in range(2):
            mine = (column, call) if group.rank_in_group == src else None
            got[rank].append(group.broadcast_object(mine, src=src))

    threads = [threading.Thread(target=run, args=(rank,)) for rank in groups]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
        assert not thread.is_alive()
    for rank in groups:
        column = rank % layout.stage_width
        assert got[rank] == [(column, 0), (column, 1)]


def test_a_broadcast_from_a_stage_outside_the_pipeline_is_refused():
    layout = PipelineSlices(ranks=(2, 2), stages=4, nnodes=2)
    group = pp_slices.SlicedPipelineGroup(_in_slice(layout, 0), layout, 0, FakeStore())
    with pytest.raises(ValueError, match="stage 4"):
        group.broadcast_object("id", src=4)


@pytest.mark.parametrize(
    ("ranks", "stages", "nnodes"),
    [
        ((), 4, 2),
        ((2, 0), 4, 2),
        ((2, 2), 0, 2),
        ((2, 2), 4, 0),
        # four ranks in three stages, or on three hosts
        ((2, 2), 3, 2),
        ((2, 2), 4, 3),
        # stages of two ranks: a slice of three ranks splits one
        ((3, 1), 2, 2),
        # hosts of two ranks: a slice of one rank splits one
        ((3, 1), 4, 2),
    ],
    ids=[
        "no_slices",
        "no_ranks",
        "no_stages",
        "no_hosts",
        "stage_width",
        "host_width",
        "whole_stages",
        "whole_hosts",
    ],
)
def test_a_layout_of_partial_stages_or_hosts_is_refused(ranks, stages, nnodes):
    with pytest.raises(ValueError, match="is not a pipeline cut into slices"):
        PipelineSlices(ranks=ranks, stages=stages, nnodes=nnodes)


def test_a_stage_beyond_the_slice_is_refused():
    layout = PipelineSlices(ranks=(2, 2), stages=4, nnodes=2)
    wrong = _in_slice(layout, 3)
    wrong.rank_in_group = 5
    with pytest.raises(RuntimeError, match="engine rank 3 is stage 5 of slice 1"):
        pp_slices.SlicedPipelineGroup(wrong, layout, 3, FakeStore())


def test_a_stage_that_disagrees_with_the_engine_ranks_is_refused():
    layout = PipelineSlices(ranks=(2, 2), stages=4, nnodes=2)
    wrong = _in_slice(layout, 3)
    wrong.rank_in_group = 0
    with pytest.raises(RuntimeError, match="engine rank 3 is stage 0 of slice 1"):
        pp_slices.SlicedPipelineGroup(wrong, layout, 3, FakeStore())


def test_code_inside_the_torch_world_keeps_the_slice_pipeline(monkeypatch):
    layout = PipelineSlices(ranks=(4, 4), stages=4, nnodes=2)
    in_slice = _in_slice(layout, 7)
    monkeypatch.setattr(parallel_state, "_PP", in_slice)
    # Without the stand-in, vLLM's group is the pipeline inside the world.
    assert pp_slices.in_slice_pipeline_group() is in_slice
    store = FakeStore()
    group = pp_slices.install_sliced_pipeline_group(layout, 7, store)
    assert parallel_state.get_pp_group() is group and group.store is store
    assert (group.rank_in_group, group.world_size) == (3, 4)
    assert pp_slices.in_slice_pipeline_group() is in_slice
    # the meshes of PCP and EP are built inside the slice's torch world
    assert mesh_utils.get_pipeline_stage() == (1, 2)


@pytest.mark.parametrize(
    ("node_rank", "nnodes", "master_addr"),
    [(3, 4, "10.0.0.3"), (1, 2, "10.0.0.1")],
    ids=["two_hosts_a_slice", "one_host_a_slice"],
)
def test_the_slice_view_describes_the_slice_alone(node_rank, nnodes, master_addr):
    """PP16 of one device a stage over two slices."""
    parallel_config = ParallelConfig(
        pipeline_parallel_size=16,
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr="10.0.0.1",
        distributed_executor_backend="mp",
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config, model_config=object()
    )
    layout = PipelineSlices(ranks=(8, 8), stages=16, nnodes=nnodes)
    view = pp_slices.slice_view(vllm_config, layout, "10.0.0.3")
    sliced = view.parallel_config
    assert (
        sliced is not parallel_config and view.model_config is vllm_config.model_config
    )
    assert (sliced.pipeline_parallel_size, sliced.world_size) == (8, 8)
    assert (sliced.nnodes, sliced.node_rank) == (nnodes // 2, node_rank % (nnodes // 2))
    # vLLM derives the torch world from these when the slice spans hosts.
    assert sliced.master_addr == master_addr
    assert sliced.world_size_across_dp == 8
    assert sliced.local_world_size == parallel_config.local_world_size
    # the engine's own config is untouched
    assert (parallel_config.pipeline_parallel_size, parallel_config.world_size) == (
        16,
        16,
    )
    assert (parallel_config.nnodes, parallel_config.node_rank) == (nnodes, node_rank)


@pytest.mark.parametrize(
    ("node_rank", "view"),
    [(1, (4, 4, 2, 1, "10.0.0.2")), (2, (2, 2, 1, 0, "10.0.0.1"))],
    ids=["two_hosts_a_slice", "one_host_a_slice"],
)
def test_the_slice_view_of_slices_of_different_sizes(node_rank, view):
    """PP6 of one device a stage on three hosts of two, as stages (4, 2):
    slice 0 on hosts 0 and 1, slice 1 on host 2."""
    parallel_config = ParallelConfig(
        pipeline_parallel_size=6,
        nnodes=3,
        node_rank=node_rank,
        master_addr="10.0.0.1",
        distributed_executor_backend="mp",
    )
    vllm_config = SimpleNamespace(parallel_config=parallel_config)
    layout = PipelineSlices(ranks=(4, 2), stages=6, nnodes=3)
    sliced = pp_slices.slice_view(vllm_config, layout, "10.0.0.2").parallel_config
    assert (
        sliced.pipeline_parallel_size,
        sliced.world_size,
        sliced.nnodes,
        sliced.node_rank,
        sliced.master_addr,
    ) == view
    assert sliced.local_world_size == parallel_config.local_world_size == 2


class _Queue:
    """MessageQueue, recording how each was made; a handle names its queue."""

    made = []

    def __init__(self, n_reader, n_local_reader, max_chunk_bytes, max_chunks):
        self.handle = ("reply", len(_Queue.made), n_local_reader)
        self.ready = False
        _Queue.made.append(self)

    @staticmethod
    def create_from_handle(handle, rank):
        queue = _Queue.__new__(_Queue)
        queue.handle = handle
        queue.reader_rank = rank
        queue.ready = False
        return queue

    def export_handle(self):
        return self.handle

    def wait_until_ready(self):
        self.ready = True


@pytest.fixture
def queues(monkeypatch):
    _Queue.made = []
    monkeypatch.setattr(shm_broadcast, "MessageQueue", _Queue)
    return _Queue


def _on_every_rank(world_size, call):
    """Runs ``call(rank)`` for every engine rank at once, as the workers do;
    returns the results in rank order."""
    results, errors = [None] * world_size, []

    def run(rank):
        try:
            results[rank] = call(rank)
        except BaseException as exc:  # reported by the test thread
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(rank,), daemon=True)
        for rank in range(world_size)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert not errors, errors
    return results


@pytest.mark.parametrize("blocking", [False, True])
def test_every_worker_joins_the_engines_broadcast_queue(queues, blocking):
    store = FakeStore()
    handle = ("broadcast", "from the executor")
    world = 4

    def join(rank):
        engine = pp_slices.EngineQueues(store, rank, world, local_world_size=2)
        # Only the workers of node 0 get the executor's handle.
        own = handle if rank < 2 else None
        return engine.create_mq_broadcaster(
            external_writer_handle=own, blocking=blocking
        )

    joined = _on_every_rank(world, join)
    assert [(q.handle, q.reader_rank) for q in joined] == [
        (handle, rank) for rank in range(world)
    ]
    assert all(q.ready == blocking for q in joined)
    assert pp_slices._get_object(store, "tpu_pp_slices_queue_broadcast_0") == handle


def test_the_writer_rank_needs_the_executors_handle(queues):
    engine = pp_slices.EngineQueues(FakeStore(), 0, 2, local_world_size=1)
    with pytest.raises(NotImplementedError, match="needs the queue's handle"):
        engine.create_mq_broadcaster(external_writer_handle=None)


def test_the_reader_gets_every_reply_queue_in_rank_order(queues):
    store = FakeStore()
    world = 6

    def reply(rank):
        engine = pp_slices.EngineQueues(store, rank, world, local_world_size=2)
        return engine.create_single_reader_mq_broadcasters(reader_rank_in_group=0)

    replies = _on_every_rank(world, reply)
    handles = {rank: queue.export_handle() for rank, (queue, _) in enumerate(replies)}
    assert replies[0][1] == [handles[rank] for rank in range(world)]
    assert all(others == [] for _, others in replies[1:])
    # The queue is read through shared memory on the reader's host only:
    # ranks 0 and 1 are on node 0.
    assert [handles[rank][2] for rank in range(world)] == [1, 1, 0, 0, 0, 0]
    assert not any(queue.ready for queue, _ in replies)


def test_engine_queues_take_the_place_of_the_inner_world_group(monkeypatch):
    monkeypatch.setattr(parallel_state, "_INNER_DP_WORLD", None)
    store = FakeStore()
    queues = pp_slices.install_engine_queues(store, 3, 8, local_world_size=4)
    assert parallel_state._INNER_DP_WORLD is queues
    assert (queues.store, queues.rank) == (store, 3)
    assert (queues.world_size, queues.local_world_size) == (8, 4)
