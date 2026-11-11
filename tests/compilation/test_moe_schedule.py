# SPDX-License-Identifier: Apache-2.0
"""Resource ordering for split-SparseCore MoE communication."""

import gc
import json
import threading

import pytest

from vllm_torchtpu.compilation import moe_schedule
from vllm_torchtpu.compilation.moe_schedule import Module, rewrite_moe_schedule


@pytest.fixture
def gc_enabled():
    was_enabled = gc.isenabled()
    gc.enable()
    yield
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


@pytest.mark.cpu_test
def test_compile_thread_rewrite_runs_without_automatic_gc(gc_enabled, monkeypatch):
    seen = []
    monkeypatch.setattr(
        moe_schedule, "rewrite_moe_schedule", lambda blob: seen.append(gc.isenabled())
    )
    moe_schedule.rewrite_moe_schedule_without_gc(b"")
    assert seen == [False]
    assert gc.isenabled()

    gc.disable()
    moe_schedule.rewrite_moe_schedule_without_gc(b"")
    assert not gc.isenabled()


@pytest.mark.cpu_test
def test_compile_thread_rewrite_restores_gc_on_error(gc_enabled, monkeypatch):
    def fail(blob):
        raise ValueError("bad module")

    monkeypatch.setattr(moe_schedule, "rewrite_moe_schedule", fail)
    with pytest.raises(ValueError):
        moe_schedule.rewrite_moe_schedule_without_gc(b"")
    assert gc.isenabled()


@pytest.mark.cpu_test
def test_overlapping_compile_threads_keep_gc_off_until_last_exits(
    gc_enabled, monkeypatch
):
    entered = threading.Barrier(2)
    first_done = threading.Event()
    states = {}

    def rewrite(blob):
        entered.wait()
        if blob == b"second":
            assert first_done.wait(timeout=10)
            # The first thread has left the wrapper; GC must stay deferred.
            states["after_first_exit"] = gc.isenabled()

    def first():
        moe_schedule.rewrite_moe_schedule_without_gc(b"first")
        first_done.set()

    monkeypatch.setattr(moe_schedule, "rewrite_moe_schedule", rewrite)
    threads = [
        threading.Thread(target=first),
        threading.Thread(
            target=moe_schedule.rewrite_moe_schedule_without_gc, args=(b"second",)
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert states == {"after_first_exit": False}
    assert gc.isenabled()


@pytest.mark.cpu_test
@pytest.mark.parametrize("gather_first", [False, True])
def test_independent_multicore_gather_finishes_before_split_moe(gather_first):
    module = Module(entry_computation_id=1)
    comp = module.computations.add(id=1)

    def node(id, opcode, operands=(), marker="", cores=()):
        instruction = comp.instructions.add(id=id, name=str(id), opcode=opcode)
        instruction.operand_ids.extend(operands)
        instruction.metadata.op_name = marker
        if cores:
            instruction.backend_config = json.dumps(
                {"sparse_core_config": {"core_ids": list(cores)}}
            ).encode()

    def gather():
        node(10, "async-start", [1], "read_cache_values/gather", ("0", "1"))
        node(11, "async-done", [10])

    node(1, "parameter")
    if gather_first:
        gather()
    node(2, "async-start", [1], "topology_three_phase_ag", ("1",))
    node(3, "async-done", [2])
    node(4, "custom-call", [1], "moe_early_ag_route")
    node(5, "async-start", [4], "all_gather", ("0",))
    node(6, "async-done", [5], "all_gather")
    node(7, "async-start", [6], "moe_preprocess_sc_scatter", ("0",))
    node(8, "async-done", [7])
    operands = [1] * 17
    operands[10], operands[11] = 8, 3
    node(9, "custom-call", operands, "fused_ep_moe_v2_fp4w_test")
    if not gather_first:
        gather()
    node(12, "async-start", [9], "next_layer/gather", ("0", "1"))
    node(13, "async-done", [12])
    node(14, "tuple", [9, 11, 13])
    comp.root_id = 14

    rewritten = rewrite_moe_schedule(module.SerializeToString())
    assert rewritten is not None
    instructions = Module.FromString(rewritten).computations[0].instructions
    by_id = {n.id: n for n in instructions}
    assert 11 in by_id[2].control_predecessor_ids
    assert 13 not in by_id[2].control_predecessor_ids
    assert json.loads(by_id[2].backend_config)["sparse_core_config"]["core_ids"] == [
        "1"
    ]
    assert json.loads(by_id[5].backend_config)["sparse_core_config"]["core_ids"] == [
        "0"
    ]
    positions = {n.id: i for i, n in enumerate(instructions)}
    for n in instructions:
        for dependency in (*n.operand_ids, *n.control_predecessor_ids):
            assert positions[dependency] < positions[n.id]


@pytest.mark.cpu_test
def test_metadata_independent_work_fills_the_metadata_wait():
    module = Module(entry_computation_id=1)
    comp = module.computations.add(id=1)

    def node(id, opcode, operands=(), marker="", cores=()):
        instruction = comp.instructions.add(id=id, name=str(id), opcode=opcode)
        instruction.operand_ids.extend(operands)
        instruction.metadata.op_name = marker
        if cores:
            instruction.backend_config = json.dumps(
                {"sparse_core_config": {"core_ids": list(cores)}}
            ).encode()

    node(1, "parameter")
    node(20, "custom-call", [1] * 17, "fused_ep_moe_v2_fp4w_previous")
    node(21, "fusion", [20], "hidden")
    node(2, "async-start", [21], "topology_three_phase_ag", ("1",))
    node(3, "async-done", [2])
    node(4, "custom-call", [21], "moe_early_ag_route")
    node(5, "async-start", [4], "all_gather", ("0",))
    node(6, "async-done", [5], "all_gather")
    node(7, "async-start", [6], "moe_preprocess_sc_scatter", ("0",))
    node(8, "async-done", [7])
    # Shared expert gate/up and its quantization never read the metadata.
    node(30, "fusion", [21], "shared_gate_up")
    node(31, "fusion", [30], "shared_quant")
    node(32, "fusion", [6], "plan_after_metadata")
    # Inputs of an unrelated submission, and work after its completion.
    node(34, "fusion", [21], "swa_pages")
    node(35, "async-start", [34], "write_swa_owned_pages", ("0",))
    node(36, "async-done", [35])
    node(37, "fusion", [36], "after_swa_write")
    node(40, "fusion", [1], "position_only")
    operands = [1] * 17
    operands[10], operands[11] = 8, 3
    node(9, "custom-call", operands, "fused_ep_moe_v2_fp4w_test")
    node(33, "fusion", [31, 9], "shared_down")
    node(14, "tuple", [33, 32, 37, 40])
    comp.root_id = 14

    rewritten = rewrite_moe_schedule(module.SerializeToString())
    assert rewritten is not None
    instructions = Module.FromString(rewritten).computations[0].instructions
    by_id = {n.id: n for n in instructions}
    for filler in (30, 31):
        assert 5 in by_id[filler].control_predecessor_ids
        assert filler in by_id[6].control_predecessor_ids
    for kept in (4, 21, 32, 33, 34, 37, 40):
        assert 5 not in by_id[kept].control_predecessor_ids
        assert kept not in by_id[6].control_predecessor_ids
    positions = {n.id: i for i, n in enumerate(instructions)}
    assert positions[5] < positions[30] < positions[31] < positions[6]
    for n in instructions:
        for dependency in (*n.operand_ids, *n.control_predecessor_ids):
            assert positions[dependency] < positions[n.id]
