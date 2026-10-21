# SPDX-License-Identifier: Apache-2.0
"""Check ordering, layer isolation, and cycle avoidance without a TPU."""

import json

import pytest

from vllm_torchtpu.compilation.moe_schedule import Module, rewrite_moe_schedule

pytestmark = pytest.mark.cpu_test


def _layer(comp, offset=0, scale_depends_on_metadata=False):
    def add(index, opcode, operands=(), marker=""):
        node = comp.instructions.add(
            id=offset + index, name=f"n{offset + index}", opcode=opcode
        )
        node.operand_ids.extend(offset + n for n in operands)
        node.metadata.op_name = marker
        return node

    add(0, "parameter")
    add(1, "parameter")
    add(2, "parameter")
    add(3, "add", (0,), "moe_early_ag_route")
    add(4, "async-start", (3,), "moe_metadata_ag")
    add(5, "async-done", (4,), "moe_metadata_ag")
    add(6, "async-start", (0,), "moe_activation_ag")
    add(7, "async-done", (6,), "moe_activation_ag")
    add(8, "copy", (5,) if scale_depends_on_metadata else (1,))
    add(9, "copy", (2,))
    operands = [0] * 17
    operands[10:12] = [5, 7]
    operands[15:17] = [8, 9]
    add(10, "custom-call", operands, "fused_ep_moe_v2_fp4w_silu/pallas_call")


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("scheduled", [False, True])
def test_submission_and_copies_ordered_within_each_layer(layers, scheduled):
    module = Module(name="test")
    comp = module.computations.add(id=1)
    for layer in range(layers):
        _layer(comp, 100 * layer)
    if scheduled:
        sequence = module.schedule.sequences.add(key=1).value
        sequence.instruction_ids.extend(n.id for n in comp.instructions)
    result = rewrite_moe_schedule(module.SerializeToString())
    assert result is not None
    rewritten = Module.FromString(result)
    nodes = {n.id: n for n in rewritten.computations[0].instructions}
    positions = {n.id: i for i, n in enumerate(rewritten.computations[0].instructions)}
    for layer in range(layers):
        n = 100 * layer
        assert positions[n + 6] < positions[n + 3] < positions[n + 4]
        assert positions[n + 4] < positions[n + 8] < positions[n + 5]
        assert positions[n + 4] < positions[n + 9] < positions[n + 5]
        assert set(nodes[n + 3].control_predecessor_ids) == {n + 6}
    if scheduled:
        assert list(rewritten.schedule.sequences[0].value.instruction_ids) == list(
            positions
        )
    assert rewrite_moe_schedule(result) is None


def test_scale_copy_that_consumes_metadata_is_not_moved_before_it():
    module = Module(name="test")
    comp = module.computations.add(id=1)
    _layer(comp, scale_depends_on_metadata=True)
    result = rewrite_moe_schedule(module.SerializeToString())
    nodes = {n.id: n for n in Module.FromString(result).computations[0].instructions}
    assert 8 not in nodes[5].control_predecessor_ids
    assert 9 in nodes[5].control_predecessor_ids


def test_unrelated_module_is_not_rewritten():
    module = Module(name="test")
    comp = module.computations.add(id=1)
    comp.instructions.add(id=0, name="parameter", opcode="parameter")
    assert rewrite_moe_schedule(module.SerializeToString()) is None


def test_native_layer_does_not_reuse_previous_layers_activation_submission():
    module = Module(name="two_layers")
    comp = module.computations.add(id=1)
    _layer(comp)
    _layer(comp, 100)
    nodes = {n.id: n for n in comp.instructions}
    nodes[100].opcode = "bitcast"
    nodes[100].operand_ids.append(10)
    nodes[106].metadata.op_name = "native_all_gather"
    nodes[107].metadata.op_name = "native_all_gather"
    result = rewrite_moe_schedule(module.SerializeToString())
    updated = Module.FromString(result)
    for node in updated.computations[0].instructions:
        if node.id >= 100:
            assert not node.control_predecessor_ids


def test_torchtpu_kernel_names_survive_missing_outer_named_scopes():
    """TorchTPU can retain primitive names while dropping compute_on scopes."""
    module = Module(name="torchtpu")
    comp = module.computations.add(id=1)
    _layer(comp)
    for node in comp.instructions:
        if node.id in (4, 5):
            node.metadata.op_name = "all_gather"
        elif node.id in (6, 7):
            node.metadata.op_name = "topology_three_phase_ag/mpmd_map"
    rewritten = rewrite_moe_schedule(module.SerializeToString())
    assert rewritten is not None
    nodes = {n.id: n for n in Module.FromString(rewritten).computations[0].instructions}
    assert 6 in nodes[3].control_predecessor_ids
    assert 6 in nodes[4].control_predecessor_ids
    assert {8, 9} <= set(nodes[5].control_predecessor_ids)


def test_explicit_core_pinning_preserves_other_backend_settings():
    module = Module(name="placement")
    comp = module.computations.add(id=1)
    _layer(comp)
    for node in comp.instructions:
        if node.id in (4, 6):
            node.backend_config = json.dumps(
                {
                    "sparse_core_config": {
                        "core_ids": ["0"],
                        "offload": "OFFLOAD_KERNEL",
                    },
                    "scoped_memory_configs": [{"size": "33554432"}],
                }
            ).encode()
    result = rewrite_moe_schedule(module.SerializeToString())
    nodes = {n.id: n for n in Module.FromString(result).computations[0].instructions}
    for index, core in ((4, "0"), (6, "1")):
        config = json.loads(nodes[index].backend_config)
        assert config["sparse_core_config"]["core_ids"] == [core]
        assert config["sparse_core_config"]["core_id_mutability"] is False
        assert config["sparse_core_config"]["offload"] == "OFFLOAD_KERNEL"
        assert config["scoped_memory_configs"] == [{"size": "33554432"}]
    assert rewrite_moe_schedule(result) is None
