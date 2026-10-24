# SPDX-License-Identifier: Apache-2.0
"""Resource ordering for split-SparseCore MoE communication."""

import json

import pytest

from vllm_torchtpu.compilation.moe_schedule import Module, rewrite_moe_schedule


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
