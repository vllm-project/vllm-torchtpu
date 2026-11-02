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
"""Host-side checks for Dynamic EPLB initial expert placement."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs, expert_placement

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(expert_placement, "_LAYER_MODULES", {})
    monkeypatch.setattr(expert_placement, "_GROUP", None)
    yield


@pytest.fixture
def dynamic_enabled(monkeypatch):
    monkeypatch.setitem(vars(envs), "TPU_MOE_EPLB_DYNAMIC", True)


def test_default_ranks_places_contiguous_block_plus_next_rank_first_expert():
    ranks = expert_placement.default_ranks(8, 2, extra_slots=1)
    assert ranks == ((0, 1, 2, 3, 4), (4, 5, 6, 7, 0))
    ranks_ep8 = expert_placement.default_ranks(384, 8, extra_slots=1)
    assert len(ranks_ep8) == 8
    assert all(len(r) == 49 for r in ranks_ep8)
    assert ranks_ep8[0][:3] == (0, 1, 2) and ranks_ep8[0][-2:] == (47, 48)
    assert ranks_ep8[7][:3] == (336, 337, 338) and ranks_ep8[7][-2:] == (383, 0)


def test_group_slot_table_and_expert_map_invert_initial_ranks():
    g = expert_placement.Group(
        experts=8,
        shards=2,
        slots=4,
        initial_ranks=((7, 0, 5, 2), (1, 6, 3, 4)),
        staging=0,
    )
    assert g.slots == 4 and g.physical == 8
    assert g.slot_table() == [[1], [4], [3], [6], [7], [2], [5], [0]]
    assert g.expert_map(1) == [-1, 0, -1, 2, 3, -1, 1, -1]
    assert g.group("target") is g and g.group("draft") is None


def test_staging_slots_and_replica_cap_shape_the_slot_table():
    """A dynamic layout keeps `staging` spare slots per rank after the active
    ones: physical slots grow, the lists and the table's slot ids stay the
    active width, and the expert map leaves the spares unassigned."""
    g = expert_placement.Group(
        experts=8,
        shards=2,
        slots=6,
        initial_ranks=((7, 0, 5, 2), (1, 6, 3, 4)),
        staging=2,
        replica_cap=2,
    )
    assert (g.active, g.staging, g.slots, g.physical) == (4, 2, 6, 12)
    rows = g.slot_table()
    assert rows[7] == [0, -1] and rows[1] == [6, -1]  # rank 1's slots start at 6
    assert all(len(r) == 2 for r in rows)
    assert g.expert_map(1) == [-1, 0, -1, 2, 3, -1, 1, -1]


def test_layer_index_reads_module_prefixes_and_checkpoint_names():
    assert expert_placement.layer_index("model.layers.12.ffn.experts") == 12
    assert expert_placement.layer_index("layers.3.ffn.experts.17.w1.weight") == 3
    assert expert_placement.layer_index("mtp.0.ffn.experts.1.w1.weight") is None
    assert expert_placement.layer_index("model.sublayers.3.x") is None


@pytest.fixture(autouse=True)
def _no_vllm_config(monkeypatch):
    """Outside a served model the target group applies."""
    import vllm.config

    def _raise():
        raise AssertionError("no current vllm config in this test")

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", _raise)


def _fake_layer(name, rank, experts=8, shards=2):
    manager = SimpleNamespace(
        global_num_experts=experts,
        ep_size=shards,
        ep_rank=rank,
        num_fused_shared_experts=0,
        expert_mask=None,
        _expert_map=torch.arange(experts, dtype=torch.int32),
    )
    manager.expert_map = manager._expert_map

    class Layer(torch.nn.Module):
        def update_expert_map_info(self):
            self.local_num_experts = self.expert_map_manager._local_num_experts
            self.register_buffer("_expert_map", self.expert_map_manager._expert_map)

    layer = Layer()
    layer.layer_name = name
    layer.expert_map_manager = manager
    return layer


def test_apply_builds_default_dynamic_layout_and_gates_the_fetch_filter(
    dynamic_enabled,
):
    assert expert_placement.fetch_filter_ids("layers.3.ffn.experts.0.w1", 1) is None
    layer = _fake_layer("model.layers.3.ffn.experts", rank=1, experts=8, shards=2)
    assert expert_placement.apply_to_layer(layer)
    # Rank 1 holds contiguous [4, 5, 6, 7] + rank 0's first expert (0) in slot 4,
    # plus 2 staging slots -> 7 physical slots per rank (14 total across 2 ranks).
    assert layer._expert_map.tolist() == [4, -1, -1, -1, 0, 1, 2, 3]
    assert layer.local_num_experts == 7
    slots = expert_placement.expert_slot_table(layer)
    assert slots.dtype == torch.int32
    assert tuple(slots.shape) == (8, 2)
    assert expert_placement.physical_experts(layer) == 14
    # Once applied, all target layers use the initial default_ranks; drafter
    # tensors keep the default contiguous filter.
    assert expert_placement.fetch_filter_ids(
        "layers.3.ffn.experts.0.w1.weight", 1
    ) == frozenset({4, 5, 6, 7, 0})
    assert expert_placement.fetch_filter_ids(
        "layers.4.ffn.experts.0.w1.weight", 1
    ) == frozenset({4, 5, 6, 7, 0})
    assert expert_placement.fetch_filter_ids("mtp.3.ffn.experts.0.w1", 1) is None


def test_disabled_when_dynamic_knob_is_off(monkeypatch):
    monkeypatch.setitem(vars(envs), "TPU_MOE_EPLB_DYNAMIC", False)
    layer = _fake_layer("model.layers.3.ffn.experts", rank=0)
    assert not expert_placement.apply_to_layer(layer)
    assert expert_placement.expert_slot_table(layer) is None
    assert expert_placement.physical_experts(layer) is None
    expert_placement.check_layer_served(layer, fused_armed=False)


def test_moe_routing_remaps_logical_ids_to_slots_and_accumulates_load(
    dynamic_enabled, monkeypatch
):
    """Non-fused MoE backends route through `moe_routing.route`, which remaps
    logical expert IDs to physical slot IDs via `_tpu_expert_slot`, sets
    `_experts_start = ep_rank * local_num_experts`, and accumulates per-slot
    routed row counts into `_tpu_expert_load`."""
    from vllm_torchtpu.layers.adapter import expert_load, moe_routing

    monkeypatch.setattr(expert_load, "_REGISTERED", [])
    layer = _fake_layer("model.layers.3.ffn.experts", rank=1, experts=8, shards=2)
    layer.global_num_experts = 8
    layer.renormalize = True
    layer.scoring_func = "softmax"
    layer.routed_scaling_factor = 1.0
    layer.use_grouped_topk = False
    layer.num_expert_group = None
    layer.topk_group = None
    layer.e_score_correction_bias = None
    layer.custom_routing_function = None
    layer.moe_config = SimpleNamespace(
        experts_per_token=2,
        ep_size=2,
        ep_rank=1,
        num_local_experts=4,
        is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(use_ep=True),
    )

    assert expert_placement.apply_to_layer(layer)
    assert layer.moe_config.num_local_experts == 7
    moe_routing.register_experts_start_buffer(layer, device=torch.device("cpu"))
    assert int(layer._experts_start.item()) == 7  # rank 1 owns slots [7..13]
    load_buf = expert_load.operand(layer)
    assert load_buf is not None and tuple(load_buf.shape) == (14,)

    # Initial layout on 8 experts / 2 shards (slots=7 per rank):
    # rank 0 slots 0..4 hold [0, 1, 2, 3, 4]; rank 1 slots 7..11 hold [4, 5, 6, 7, 0].
    # Expert 0 has 2 replicas at slots [0, 11]; expert 5 has 1 replica at slot 8.
    x = torch.zeros((4, 16), dtype=torch.bfloat16)
    logits = torch.full((4, 8), -10.0, dtype=torch.float32)
    logits[:, 0] = 5.0
    logits[:, 5] = 4.0

    weights, slot_ids = moe_routing.route(layer, x, logits)
    assert tuple(weights.shape) == (4, 2)
    # Token rows 0..3 alternate between expert 0's two replicas (slots 0 and 11),
    # while expert 5 always maps to slot 8.
    assert slot_ids.tolist() == [[0, 8], [11, 8], [0, 8], [11, 8]]
    assert load_buf.tolist() == [2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 2, 0, 0]

    # Installing a new slot table redirects subsequent routes without recompiling.
    new_table = expert_placement.expert_slot_table(layer).clone()
    new_table[5] = torch.tensor([5, -1], dtype=torch.int32)  # move expert 5 to slot 5
    expert_placement.install_table(layer, new_table)
    _, slot_ids_after = moe_routing.route(layer, x, logits)
    assert slot_ids_after.tolist() == [[0, 5], [11, 5], [0, 5], [11, 5]]
    assert load_buf[5].item() == 4


def test_mover_parts_supports_fp8_and_compressed_tensors_moe_layouts():
    from vllm_torchtpu import expert_mover

    fp8_layer = torch.nn.Module()
    fp8_layer.w13_weight = torch.nn.Parameter(
        torch.zeros((7, 16, 32), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    fp8_layer.w2_weight = torch.nn.Parameter(
        torch.zeros((7, 16, 16), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    fp8_layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones((7, 1, 1, 32), dtype=torch.float32), requires_grad=False
    )
    fp8_layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.ones((7, 1, 1, 16), dtype=torch.float32), requires_grad=False
    )
    names = [
        name for _, name, _, kind in expert_mover._parts(fp8_layer) if kind == "slot"
    ]
    assert names == [
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    ]

    ct_layer = torch.nn.Module()
    ct_layer.w13_weight_packed = torch.nn.Parameter(
        torch.zeros((7, 4, 32), dtype=torch.int32), requires_grad=False
    )
    ct_layer.w2_weight_packed = torch.nn.Parameter(
        torch.zeros((7, 4, 16), dtype=torch.int32), requires_grad=False
    )
    ct_layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones((7, 1, 32), dtype=torch.bfloat16), requires_grad=False
    )
    ct_layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones((7, 1, 16), dtype=torch.bfloat16), requires_grad=False
    )
    ct_names = [
        name for _, name, _, kind in expert_mover._parts(ct_layer) if kind == "slot"
    ]
    assert ct_names == [
        "w13_weight_packed",
        "w2_weight_packed",
        "w13_weight_scale",
        "w2_weight_scale",
    ]


def test_drafter_layers_are_left_unplaced_by_default(dynamic_enabled, monkeypatch):
    import vllm.config

    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="draft", hf_config=SimpleNamespace(num_hidden_layers=6)
        )
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: runner)
    assert not expert_placement.apply_to_layer(
        _fake_layer("model.layers.6.ffn.experts", 1, experts=4)
    )
