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
"""Pipelined prefill across DP engines: the plan and the shared-memory hand-off."""

import itertools
import threading
from multiprocessing import shared_memory

import pytest
import torch

from vllm_torchtpu.distributed.dp_pipeline import plan as plan_mod
from vllm_torchtpu.distributed.dp_pipeline import transport
from vllm_torchtpu.distributed.dp_pipeline.plan import PipelinePlan

_server_ids = itertools.count(50000)


def test_parse_splits_and_stage_bounds(monkeypatch):
    assert plan_mod.parse_splits("2:30; 3:20,41;4:14,29,44") == {
        2: (30,),
        3: (20, 41),
        4: (14, 29, 44),
    }
    for bad in ("2:30,40", "3:41,20", "1:", "x:3", "4:0,10,20"):
        with pytest.raises(ValueError):
            plan_mod.parse_splits(bad)
    monkeypatch.setattr(
        plan_mod.envs, "TPU_DP_PIPELINE_SPLITS", "2:30;3:20,41;4:14,29,44"
    )
    assert plan_mod.stage_bounds(4, 60) == (0, 14, 29, 44, 60)
    with pytest.raises(ValueError, match="no 5-stage"):
        plan_mod.stage_bounds(5, 60)
    with pytest.raises(ValueError, match="does not fit"):
        plan_mod.stage_bounds(4, 40)


def test_segments_tile_every_split(monkeypatch):
    monkeypatch.setattr(
        plan_mod.envs, "TPU_DP_PIPELINE_SPLITS", "2:30;3:20,41;4:14,29,44"
    )
    monkeypatch.setattr(plan_mod.envs, "TPU_DP_PIPELINE_STAGES", 2)
    assert plan_mod.segment_bounds(60) == (0, 30, 60)
    monkeypatch.setattr(plan_mod.envs, "TPU_DP_PIPELINE_STAGES", 4)
    bounds = plan_mod.segment_bounds(60)
    assert bounds == (0, 14, 20, 29, 30, 41, 44, 60)
    for num_stages in (2, 3, 4):
        assert set(plan_mod.stage_bounds(num_stages, 60)) <= set(bounds)


def test_plan_round_trip_and_tokens():
    plan = PipelinePlan(
        plan_mod.make_pipeline_id(7, 3), (2, 0, 1), (0, 20, 41, 60), 49152
    )
    assert PipelinePlan.from_json(plan.to_json()) == plan
    assert plan.server_id == 7 and plan.num_stages == 3
    assert plan.stage_of(0) == 1 and plan.layers(1) == (20, 41)
    tokens = {plan.chunk_token(p) for p in (0, 16384, 32768)}
    assert len(tokens) == 3 and 0 not in tokens
    other = PipelinePlan(
        plan_mod.make_pipeline_id(7, 4), (2, 0, 1), (0, 20, 41, 60), 49152
    )
    assert other.chunk_token(0) not in tokens
    for ranks, bounds in (
        ((1,), (0, 60)),
        ((1, 1), (0, 30, 60)),
        ((0, 1), (0, 60)),
        ((0, 1), (0, 40, 30)),
    ):
        with pytest.raises(ValueError):
            PipelinePlan(1, ranks, bounds, 100)
    sp = type("SP", (), {"extra_args": {plan_mod.PLAN_KEY: plan.to_json()}})()
    plain = type("SP", (), {"extra_args": None})()
    assert plan_mod.plan_of(sp) == plan and plan_mod.is_pipelined(sp)
    assert plan_mod.plan_of(plain) is None and not plan_mod.is_pipelined(plain)
    assert plan_mod.make_pipeline_id(1, (1 << 20) - 1) != 1 << 20  # never wraps to 0
    with pytest.raises(ValueError, match="server id"):
        plan_mod.make_pipeline_id(1 << 16, 1)
    assert plan_mod.server_id_of(4_194_303) < 1 << 16


@pytest.fixture
def ring_factory():
    made = []

    def make(tp_size=2, max_rows=8, width=4, dtype=torch.bfloat16):
        server = next(_server_ids)
        rings = [
            transport.EdgeRing(
                server,
                0,
                1,
                transport.HIDDEN,
                max_rows,
                (width,),
                dtype,
                tp_size,
                materialize=lambda t: None,
            )
            for _ in range(1 + tp_size)  # the writer's map, one per reader
        ]
        made.append(transport.ring_name(server, 0, 1, transport.HIDDEN))
        return rings

    yield make
    for name in made:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()


def test_hand_off_reaches_every_receiving_rank(ring_factory):
    sender, *receivers = ring_factory(tp_size=2)
    chunks = [torch.randn(6, 4).to(torch.bfloat16) for _ in range(5)]
    got = [[None] * len(chunks) for _ in receivers]

    def receive(rank):
        for i in range(len(chunks)):
            got[rank][i] = receivers[rank].take(
                100 + i, 6, torch.device("cpu"), rank, 10.0
            )

    threads = [threading.Thread(target=receive, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for i, chunk in enumerate(chunks):
        # Two slots: the third put waits until both ranks took the first chunk.
        sender.put(100 + i, chunk, 10.0)
    for t in threads:
        t.join(10.0)
        assert not t.is_alive()
    for rank in range(2):
        for i, chunk in enumerate(chunks):
            assert torch.equal(got[rank][i], chunk)
            # The received tensor is a copy: the slot it came from is reused.
            assert got[rank][i].data_ptr() != chunk.data_ptr()


def test_a_slot_stays_busy_until_every_rank_acknowledges(ring_factory):
    sender, rank0, rank1 = ring_factory(tp_size=2)
    for token in (1, 2):
        sender.put(token, torch.ones(2, 4, dtype=torch.bfloat16), 1.0)
    rank0.take(1, 2, torch.device("cpu"), 0, 1.0)
    with pytest.raises(TimeoutError, match="free slot"):
        sender.put(3, torch.ones(2, 4, dtype=torch.bfloat16), 0.05)
    rank1.take(1, 2, torch.device("cpu"), 1, 1.0)
    sender.put(3, torch.ones(2, 4, dtype=torch.bfloat16), 1.0)


def test_take_checks_the_row_count_and_times_out(ring_factory):
    sender, rank0, _ = ring_factory(tp_size=2)
    with pytest.raises(TimeoutError, match="chunk 9"):
        rank0.take(9, 2, torch.device("cpu"), 0, 0.05)
    sender.put(9, torch.zeros(3, 4, dtype=torch.bfloat16), 1.0)
    with pytest.raises(RuntimeError, match="3 rows"):
        rank0.take(9, 2, torch.device("cpu"), 0, 1.0)
    with pytest.raises(ValueError, match="does not fit"):
        sender.put(10, torch.zeros(9, 4, dtype=torch.bfloat16), 1.0)


def test_a_smaller_left_over_ring_is_refused():
    server = next(_server_ids)
    small = transport.EdgeRing(
        server, 0, 1, "h", 2, (4,), torch.bfloat16, 2, materialize=lambda t: None
    )
    try:
        with pytest.raises(RuntimeError, match="remove it"):
            transport.EdgeRing(
                server,
                0,
                1,
                "h",
                64,
                (4,),
                torch.bfloat16,
                2,
                materialize=lambda t: None,
            )
    finally:
        small.close()
        shared_memory.SharedMemory(name=transport.ring_name(server, 0, 1, "h")).unlink()


def test_unlink_rings_removes_every_ring_of_the_server():
    server = next(_server_ids)
    made = [
        (0, 1, transport.HIDDEN),
        (0, 2, transport.hidden_channel(1)),
        (2, 1, transport.kv_channel(1)),
    ]
    rings = [
        transport.EdgeRing(
            server, src, dst, ch, 2, (4,), torch.bfloat16, 1, lambda t: None
        )
        for src, dst, ch in made
    ]
    other = next(_server_ids)
    kept = transport.EdgeRing(
        other, 0, 1, "h", 2, (4,), torch.bfloat16, 1, lambda t: None
    )
    transport.unlink_rings(server, 3, tp_size=2)
    for src, dst, ch in made:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=transport.ring_name(server, src, dst, ch))
    # Rings stay usable by the engines that map them.
    rings[0].put(5, torch.ones(2, 4, dtype=torch.bfloat16), 1.0)
    for ring in [*rings, kept]:
        ring.close()
    shared_memory.SharedMemory(name=transport.ring_name(other, 0, 1, "h")).unlink()


def test_a_hand_off_of_several_tensors_arrives_in_pieces(ring_factory):
    sender, receiver = ring_factory(tp_size=1)
    parts = [torch.full((2, 4), float(i)).to(torch.bfloat16) for i in range(3)]
    sender.put(11, parts, 1.0)
    pieces = receiver.take(11, 6, torch.device("cpu"), 0, 1.0, split=2)
    assert all(torch.equal(a, b) for a, b in zip(pieces, parts))
    sender.put(12, torch.ones(5, 4, dtype=torch.bfloat16), 1.0)
    # Rows as sent.
    assert receiver.take(12, None, torch.device("cpu"), 0, 1.0).shape == (5, 4)


def test_max_stages_follows_the_prompt_length(monkeypatch):
    monkeypatch.setattr(plan_mod.envs, "TPU_DP_PIPELINE_STAGES", 4)
    monkeypatch.setattr(
        plan_mod.envs, "TPU_DP_PIPELINE_STAGE_TOKENS", "3:40960;4:73728"
    )
    assert [plan_mod.max_stages(n) for n in (33000, 40960, 73727, 87148)] == [
        2,
        3,
        3,
        4,
    ]
    monkeypatch.setattr(plan_mod.envs, "TPU_DP_PIPELINE_STAGES", 3)
    assert plan_mod.max_stages(87148) == 3
    for bad in ("2:100", "3:1,2", "3:x", "3:80000;4:40000", "4", "3:1;3:2", "3:-5"):
        with pytest.raises(ValueError):
            plan_mod.parse_stage_tokens(bad)
    assert plan_mod.parse_stage_tokens(" 3:10 ; 4:20 ") == {3: 10, 4: 20}


def test_pipelined_tokens_keep_to_the_full_chunks_bucket():
    sizes = [16, 256, 1024, 2048, 4096, 8192, 16384]
    chunk = 16384
    # A rest that pads to a smaller bucket is the owner's ordinary step.
    assert plan_mod.pipelined_tokens(100206, chunk, sizes) == 98304
    assert plan_mod.pipelined_tokens(87148, chunk, sizes) == 81920
    # A rest that pads to the full chunk's bucket is pipelined too.
    assert plan_mod.pipelined_tokens(159430, chunk, sizes) == 159430
    assert plan_mod.pipelined_tokens(49152, chunk, sizes) == 49152
    with pytest.raises(ValueError, match="some tokens"):
        PipelinePlan(1, (0, 1), (0, 30, 60), 0)


def test_a_ring_still_being_sized_by_its_creator_is_waited_for():
    import _posixshmem  # the creator's first step, before it sizes the ring
    import os

    server = next(_server_ids)
    name = transport.ring_name(server, 0, 1, "h")
    fd = _posixshmem.shm_open("/" + name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    sizer = threading.Timer(0.05, lambda: os.ftruncate(fd, 4096 + 2 * 2 * 4 * 2))
    sizer.start()
    try:
        ring = transport.EdgeRing(
            server, 0, 1, "h", 2, (4,), torch.bfloat16, 1, lambda t: None
        )
        ring.put(3, torch.ones(2, 4, dtype=torch.bfloat16), 1.0)
        ring.close()
    finally:
        sizer.join()
        os.close(fd)
        shared_memory.SharedMemory(name=name).unlink()
