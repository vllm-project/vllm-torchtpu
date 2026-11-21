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
"""Host-side rules of the fused EP v3 bridge (layers/adapter/fused_moe_ep_v3.py):
the per-layout step gate, the meeting point with the few-token kernel, the
copy-slot slice of a placed layer's weights, the routing and the kernel
limits it admits, and the operands and graph inputs of each call."""

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal.pallas._mock_device import _PallasDevice
from vllm.distributed import parallel_state

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.fused_moe.v3 import ffn_kernel
from vllm_torchtpu.kernels.fused_moe.v3 import layer as kernel_layer
from vllm_torchtpu.kernels.fused_moe.v3 import layout as kernel_layout
from vllm_torchtpu.layers.adapter import (
    expert_placement,
    moe_routing,
    sequence_parallel,
    token_padding,
)
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as v3

pytestmark = pytest.mark.cpu_test

EP = 2
# JAX's chip-record read itself, before any test replaces it.
_GET_TPU_INFO = pltpu.get_tpu_info


def _chip(version="TPU_7X"):
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion[version], 1)


def _mesh(ep=EP, platform="tpu", device_kind="TPU7x"):
    """An EP mesh as build_ep_mesh returns it in serving: torch_tpu's
    get_pallas_mesh devices (`_PallasDevice`, named after the chip torch_tpu's
    runtime reports) on one axis."""
    devices = [
        _PallasDevice(device_kind=device_kind, platform=platform, id=i)
        for i in range(ep)
    ]
    return jax.sharding.Mesh(np.array(devices, dtype=object), (v3.EP_AXIS_NAME,))


@pytest.fixture
def groups(monkeypatch):
    # An engine without a drafter: the row threshold follows the model alone.
    monkeypatch.setattr(sequence_parallel, "_HAS_DRAFTER", False)

    def set_layout(dp, tp):
        monkeypatch.setattr(
            parallel_state, "model_parallel_is_initialized", lambda: True
        )
        monkeypatch.setattr(
            parallel_state, "get_dp_group", lambda: SimpleNamespace(world_size=dp)
        )
        monkeypatch.setattr(
            parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=tp)
        )

    return set_layout


def test_step_gate_defaults_before_the_groups_exist(monkeypatch):
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    assert v3.step_min_tokens() == v3.DEFAULT_STEP_MIN_TOKENS == 64


@pytest.mark.parametrize(
    "layout,gate", [((4, 2), 32), ((1, 8), 33), ((2, 4), 64), ((1, 1), 64)]
)
def test_step_gate_follows_the_layout(groups, layout, gate):
    groups(*layout)
    assert v3.step_min_tokens() == gate
    explicit = SimpleNamespace(
        data_parallel_size=layout[0], tensor_parallel_size=layout[1]
    )
    assert v3.step_min_tokens(explicit) == gate


def test_no_few_token_kernel_means_no_meeting_point(groups):
    groups(1, 1)
    v3.check_kernel_threshold(ep=8, tp=1)
    v3.check_kernel_threshold(ep=8, tp=1, lowtok_cap=0)


@pytest.mark.parametrize("cap,meets", [(8 * 63, True), (8 * 63 - 1, False)])
def test_few_token_cap_must_reach_the_fused_gate_across_the_node(groups, cap, meets):
    """At tp 1 an engine's rows below the gate (64) are gathered from all 8
    engines, so the few-token kernel must cover 8 * 63 rows."""
    groups(1, 1)
    if meets:
        v3.check_kernel_threshold(ep=8, tp=1, lowtok_cap=cap)
    else:
        with pytest.raises(ValueError, match="leave steps between the two kernels"):
            v3.check_kernel_threshold(ep=8, tp=1, lowtok_cap=cap)


@pytest.mark.parametrize("threshold", [0, 256])
def test_tensor_parallel_gate_refuses_a_first_sliced_bucket_below_it(
    groups, monkeypatch, threshold
):
    """TP8 (gate 33) at a sequence-parallel threshold of 0 (the default: every
    sequence-parallel step is sliced, so the first sliced bucket is one row)
    or 256 (32 rows per rank): the first sliced bucket is below the gate, so
    a few-token cap is refused."""
    monkeypatch.setattr(sequence_parallel, "DEFAULT_MIN_ROWS", threshold)
    groups(1, 8)
    with pytest.raises(ValueError, match="leave steps between the two kernels"):
        v3.check_kernel_threshold(ep=8, tp=8, lowtok_cap=448)


def test_tensor_parallel_gate_is_checked_at_the_first_sliced_bucket(
    groups, monkeypatch
):
    """TP8 at a 512-row sequence-parallel threshold (the value the TP8 decode
    legs ran; the threshold defaults to 0 until a model follows each step's
    form): 64 rows per rank at the first sliced bucket, which meets the TP8
    gate of 33."""
    monkeypatch.setattr(sequence_parallel, "DEFAULT_MIN_ROWS", 512)
    groups(1, 8)
    v3.check_kernel_threshold(ep=8, tp=8, lowtok_cap=448)


@pytest.mark.parametrize("layout", [(1, 8), (4, 2)])
def test_a_model_that_follows_the_step_form_meets_the_few_token_kernel(
    groups, monkeypatch, layout
):
    """With the threshold at 0 (no model follows each step's form) the fused
    kernel and the few-token kernel (448-row cap) leave rows to neither under
    TP; once the Qwen3.5 model patch follows the form, the per-layout
    thresholds (512 at TP8, 64 at DP4 x TP2) meet the step gates (33, 32)."""
    groups(*layout)
    dp, tp = layout
    monkeypatch.setattr(sequence_parallel, "_FOLLOWS_STEP_FORM", False)
    with pytest.raises(ValueError, match="leave steps between the two kernels"):
        v3.check_kernel_threshold(ep=dp * tp, tp=tp, lowtok_cap=448)
    sequence_parallel.follow_step_form()
    v3.check_kernel_threshold(ep=dp * tp, tp=tp, lowtok_cap=448)


def _layer(slots, primaries=None, hidden=128, inter=256):
    layer = SimpleNamespace(
        w13_weight=torch.zeros(slots, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(slots, inter, hidden, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(slots, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(slots, 1, 1, hidden),
        global_num_experts=4 * EP,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            experts_per_token=2,
            is_sequence_parallel=False,
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=1, is_sequence_parallel=False
            ),
        ),
    )
    if primaries is not None:
        setattr(layer, expert_placement.PRIMARY_EXPERTS_ATTR, primaries)
    return layer


@pytest.fixture
def armed(monkeypatch):
    """Everything prebuild needs besides the layer, with _build_op recording
    the expert count it is built for."""
    built = {}
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024)
    monkeypatch.setattr(v3, "_OP", None)
    monkeypatch.setattr(v3, "_OP_CONTRACT", None)
    monkeypatch.setattr(v3, "_RANK_BUFFER", None)
    monkeypatch.setattr(v3, "_EXPERT_ORDER", None)
    # prebuild records both process-wide.
    monkeypatch.setattr(v3, "_TP_SIZE", None)
    monkeypatch.setattr(v3, "_EP_SIZE", None)
    monkeypatch.setattr(v3, "build_ep_mesh", lambda: _mesh())
    # The kernel's VMEM and generation checks read the TPU7x chip record.
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: _chip())
    monkeypatch.setattr(v3, "_max_node_tokens", lambda ep: 4096)
    monkeypatch.setattr(v3, "_expert_block_permutation", lambda ep, dev: None)

    def build_op(*args, local_experts, **kwargs):
        built["local_experts"] = local_experts
        v3._OP = "v3_op"

    monkeypatch.setattr(v3, "_build_op", build_op)
    return built


def _prebuild(layer, renormalize=True):
    with patch.object(v3, "ep_mesh_index", return_value=0):
        v3.prebuild_fused_moe_ep(
            layer, topk=2, renormalize=renormalize, activation="silu"
        )
    return getattr(layer, "_tpu_fused_ep_eligible", False)


def test_the_bridge_asks_the_kernel_for_its_lowering_by_argument(monkeypatch):
    """The bridge's op calls the kernel without its inner jit, without side
    effects and with the rank tensor as the shard index, as arguments of
    that call; the kernel package has no process-wide flags left that a
    bridge could set for every other caller."""
    calls = []
    monkeypatch.setattr(v3, "fused_ep_moe", lambda *a, **kw: calls.append(kw))
    run_moe = v3.run_moe_v3(
        SimpleNamespace(axis_names=(v3.MESH_AXIS,), shape={v3.MESH_AXIS: EP}),
        2,
        True,
        "silu",
        hidden=128,
        inter=256,
        experts_per_shard=4,
    )
    rank = object()
    run_moe(*[None] * 6, rank, None)
    (kwargs,) = calls
    assert kwargs["side_effects"] is False
    assert kwargs["wrap_in_jit"] is False
    assert kwargs["shard_index"] is rank
    # Without valid counts too: a row with a NaN logit routes nowhere.
    assert kwargs["drop_nan_rows"] is True
    for module, flag in (
        (kernel_layer, "WRAP_IN_JIT"),
        (kernel_layout, "SIDE_EFFECTS"),
        (kernel_layout, "SHARD_INDEX_OPERAND"),
    ):
        assert not hasattr(module, flag), flag


def test_a_placed_layer_streams_its_primaries_only(armed):
    """4 primaries and 2 copy slots per rank over EP 2: the kernel is built for
    the 4 primaries, which is what makes 8 experts split over 2 shards."""
    assert _prebuild(_layer(slots=6, primaries=4))
    assert armed["local_experts"] == 4


def test_without_the_attribute_every_slot_is_an_expert(armed):
    """The control: the same weights with no primary count are 6 experts per
    shard, which do not match the router's 8 experts over 2 shards."""
    assert not _prebuild(_layer(slots=6))
    assert "local_experts" not in armed


def test_an_unplaced_layer_is_unchanged(armed):
    assert _prebuild(_layer(slots=4))
    assert armed["local_experts"] == 4


def test_a_refused_contract_keeps_the_rank_and_the_expert_order(armed, monkeypatch):
    """The second layer's top_k differs from the operation's, so it is refused
    before it can replace the rank buffer or the expert order the first layer
    set."""
    order, other = torch.tensor([1, 0]), torch.tensor([0, 1])
    monkeypatch.setattr(v3, "_expert_block_permutation", lambda ep, dev: order)
    first = _layer(slots=4)
    with patch.object(v3, "ep_mesh_index", return_value=1):
        v3.prebuild_fused_moe_ep(first, topk=2, renormalize=True, activation="silu")
    assert first._tpu_fused_ep_eligible
    rank = v3._RANK_BUFFER
    assert rank.item() == 1 and v3._EXPERT_ORDER is order

    monkeypatch.setattr(v3, "_expert_block_permutation", lambda ep, dev: other)
    second = _layer(slots=4)
    with patch.object(v3, "ep_mesh_index", return_value=0):
        v3.prebuild_fused_moe_ep(second, topk=4, renormalize=True, activation="silu")
    assert not second._tpu_fused_ep_eligible
    assert v3._RANK_BUFFER is rank and rank.item() == 1
    assert v3._EXPERT_ORDER is order


def test_an_ep_mesh_index_outside_the_mesh_is_refused(armed):
    layer = _layer(slots=4)
    with patch.object(v3, "ep_mesh_index", return_value=EP):
        v3.prebuild_fused_moe_ep(layer, topk=2, renormalize=True, activation="silu")
    assert not layer._tpu_fused_ep_eligible
    assert v3._OP is None and v3._RANK_BUFFER is None


@pytest.mark.parametrize(("ep", "tp"), [(2, 2), (8, 2), (8, 8)])
def test_the_batch_floor_counts_one_scheduler_batch_per_dp_rank(
    armed, monkeypatch, ep, tp
):
    """Under TP with sequence parallelism the TP ranks of a DP rank split one
    scheduler batch, so the batch floor sees ep // tp batches, not ep: 4 at
    DP4 x TP2 (EP 8), 1 at TP8 (EP 8)."""
    batches = []
    monkeypatch.setattr(v3, "_max_node_tokens", lambda n: batches.append(n) or 4096)
    monkeypatch.setattr(v3, "_tensor_parallel_size", lambda: tp)
    monkeypatch.setattr(sequence_parallel, "enabled", lambda: True)
    monkeypatch.setattr(v3, "build_ep_mesh", lambda: _mesh(ep))
    layer = _layer(slots=4)
    layer.global_num_experts = 4 * ep
    layer.moe_config.is_sequence_parallel = True
    assert _prebuild(layer)
    assert batches == [ep // tp]


@pytest.mark.parametrize(
    "routing",
    [
        dict(scoring_func="sigmoid"),
        dict(scoring_func="sqrtsoftplus"),
        dict(e_score_correction_bias=torch.zeros(4 * EP)),
        dict(routed_scaling_factor=2.5),
    ],
    ids=["sigmoid", "sqrtsoftplus", "correction_bias", "routed_scaling"],
)
def test_routing_the_v3_router_cannot_compute_is_refused(armed, routing):
    """The v2 bridge's admission takes these (the v2 kernel has operands for
    them); the v3 router is a plain softmax with no bias or scale."""
    layer = _layer(slots=4)
    for name, value in routing.items():
        setattr(layer, name, value)
    assert not _prebuild(layer)
    assert "local_experts" not in armed


def test_simulated_routing_is_refused(armed, monkeypatch):
    """uniform_random simulation is served by the v2 kernel through routing
    operands; the v3 kernel routes by itself, so it is refused."""
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
    monkeypatch.setattr(moe_routing, "_strategy_name", "uniform_random")
    assert not _prebuild(_layer(slots=4))


def test_raw_router_weights_are_refused(armed):
    """The kernel refuses renormalize=False at trace time (no TPU case covers
    the raw weights), so the bridge declines such a layer up front, as it
    declines the other routing it cannot serve, and the layer keeps the
    grouped-matmul path."""
    assert not _prebuild(_layer(slots=4), renormalize=False)
    assert "local_experts" not in armed


def test_the_softmax_router_is_admitted(armed):
    """The control: softmax scoring, no bias, scale 1.0 and no simulation,
    the Qwen3.5 router."""
    assert _prebuild(_layer(slots=4))


def _big_layer(slots, hidden, inter, primaries=None):
    """A layer of real widths whose weights are expanded views: prebuild
    reads their shapes and dtypes, never their values."""
    layer = _layer(slots=1)
    fp8 = torch.zeros((), dtype=torch.float8_e4m3fn)
    layer.w13_weight = fp8.expand(slots, hidden, 2 * inter)
    layer.w2_weight = fp8.expand(slots, inter, hidden)
    layer.w13_weight_scale_inv = torch.ones(()).expand(slots, 1, 1, 2 * inter)
    layer.w2_weight_scale_inv = torch.ones(()).expand(slots, 1, 1, hidden)
    if primaries is not None:
        setattr(layer, expert_placement.PRIMARY_EXPERTS_ATTR, primaries)
    return layer


@pytest.mark.parametrize(
    ("dp", "tp", "slots"), [(4, 2, 64), (1, 8, 68)], ids=["dp4_tp2", "tp8"]
)
def test_the_measured_qwen3_5_layers_arm(armed, monkeypatch, dp, tp, slots):
    """The Qwen3.5-397B-A17B decode layers on TPU7x pass every kernel check
    prebuild makes: EP 8, hidden 4096, intermediate 1024, 512 experts, top-10,
    64 routed experts per shard (DP4 x TP2), and 68 weight slots of which
    64 are routed (TP8)."""
    ep = dp * tp
    monkeypatch.setattr(v3, "build_ep_mesh", lambda: _mesh(ep))
    monkeypatch.setattr(v3, "_tensor_parallel_size", lambda: tp)
    monkeypatch.setattr(sequence_parallel, "enabled", lambda: True)
    layer = _big_layer(slots, 4096, 1024, primaries=64 if slots > 64 else None)
    layer.global_num_experts = 512
    layer.moe_config.is_sequence_parallel = True
    with patch.object(v3, "ep_mesh_index", return_value=0):
        v3.prebuild_fused_moe_ep(layer, topk=10, renormalize=True, activation="silu")
    assert layer._tpu_fused_ep_eligible
    assert armed["local_experts"] == 64


@pytest.mark.parametrize(
    ("platform", "chip"), [("tpu", "TPU_V6E"), ("tpu", "TPU_V5P"), ("cpu", None)]
)
def test_a_chip_the_kernel_is_not_built_for_is_refused(
    armed, monkeypatch, platform, chip
):
    """The kernel's FFN build refuses any generation outside
    SUPPORTED_GENERATIONS; a CPU device is refused without reading a chip
    record, which only a TPU process has."""
    monkeypatch.setattr(
        v3, "build_ep_mesh", lambda: _mesh(platform=platform, device_kind=platform)
    )
    if chip is None:

        def no_record():
            raise AssertionError("a CPU device has no chip record")

        monkeypatch.setattr(pltpu, "get_tpu_info", no_record)
    else:
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: _chip(chip))
    assert not _prebuild(_layer(slots=4))
    assert "local_experts" not in armed


def test_a_chip_record_that_cannot_be_read_refuses_the_layer(armed, monkeypatch):
    """Weight load reads the chip record; when JAX cannot read it (here the
    device kind it raises on outside a TPU process), prebuild refuses the
    layer with the reason instead of failing the load."""

    def unreadable():
        raise ValueError("Unsupported TPU device kind: cpu")

    monkeypatch.setattr(pltpu, "get_tpu_info", unreadable)
    reason = v3._unsupported_kernel_reason(
        _mesh(), topk=2, hidden=128, inter=256, local_experts=4
    )
    assert reason is not None and "chip record could not be read" in reason
    assert "Unsupported TPU device kind: cpu" in reason
    assert not _prebuild(_layer(slots=4))
    assert "local_experts" not in armed


@pytest.mark.parametrize(
    ("dp", "tp", "slots"), [(4, 2, 64), (1, 8, 68)], ids=["dp4_tp2", "tp8"]
)
def test_weight_load_reads_the_chip_the_mesh_names(armed, monkeypatch, dp, tp, slots):
    """In a CPU process (JAX's default device is a CPU) the chip record is
    read unpatched, for a mesh of torch_tpu's devices named TPU7x: prebuild
    reads the chip the mesh names, generation 7 with 64 MiB of VMEM, and
    arms both measured Qwen3.5 layers. Read off JAX's default device instead,
    it is "Unsupported TPU device kind: cpu" and the layer is refused."""
    assert jax.devices()[0].platform == "cpu"
    monkeypatch.setattr(pltpu, "get_tpu_info", _GET_TPU_INFO)
    mesh = _mesh(dp * tp)
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        info = pltpu.get_tpu_info()
    assert info.generation == 7
    assert info.vmem_capacity_bytes == 64 * 2**20
    monkeypatch.setattr(v3, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(v3, "_tensor_parallel_size", lambda: tp)
    monkeypatch.setattr(sequence_parallel, "enabled", lambda: True)
    layer = _big_layer(slots, 4096, 1024, primaries=64 if slots > 64 else None)
    layer.global_num_experts = 512
    layer.moe_config.is_sequence_parallel = True
    with patch.object(v3, "ep_mesh_index", return_value=0):
        v3.prebuild_fused_moe_ep(layer, topk=10, renormalize=True, activation="silu")
    assert layer._tpu_fused_ep_eligible
    assert armed["local_experts"] == 64


@pytest.mark.parametrize(
    ("device_kind", "message"),
    [
        ("TPU v6 lite", "built and tested on TPU generations (7,) only"),
        ("TPU v99", "chip record could not be read"),
    ],
    ids=["v6e", "unknown"],
)
def test_a_mesh_naming_another_chip_is_refused(
    armed, monkeypatch, device_kind, message
):
    """The same unpatched read refuses a mesh whose devices name a chip the
    kernel is not built for (TPU v6e, generation 6) or one JAX has no record
    of."""
    monkeypatch.setattr(pltpu, "get_tpu_info", _GET_TPU_INFO)
    reason = v3._unsupported_kernel_reason(
        _mesh(device_kind=device_kind), topk=2, hidden=128, inter=256, local_experts=4
    )
    assert reason is not None and message in reason, reason
    monkeypatch.setattr(v3, "build_ep_mesh", lambda: _mesh(device_kind=device_kind))
    assert not _prebuild(_layer(slots=4))
    assert "local_experts" not in armed


def test_a_refused_layer_logs_no_kernel_settings(armed, monkeypatch):
    """The weight-load check builds the served config without logging it:
    a layer the transport refuses (here EP 6) leaves no settings line, which
    is logged once the operation is built."""
    lines = []
    monkeypatch.setattr(v3.logger, "info_once", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(v3, "build_ep_mesh", lambda: _mesh(6))
    layer = _layer(slots=4)
    layer.global_num_experts = 24
    assert not _prebuild(layer)
    assert not [m for m in lines if "settings for" in m], lines


def test_the_tp_width_outside_a_config_context_is_one(monkeypatch):
    """vLLM's get_current_vllm_config raises AssertionError when no config is
    set (outside set_current_vllm_config, as in the warm-up traces); the
    width then reads 1, which is why prebuild records it at weight load."""
    from vllm.config import get_current_vllm_config

    monkeypatch.setattr("vllm.config.vllm._current_vllm_config", None)
    with pytest.raises(AssertionError):
        get_current_vllm_config()
    assert v3._tensor_parallel_size() == 1


@pytest.mark.parametrize("ep", [6, 64])
def test_a_mesh_width_the_transport_cannot_take_is_refused(armed, monkeypatch, ep):
    """The transport takes a power-of-two mesh of at most MAX_SHARDS (32)."""
    monkeypatch.setattr(v3, "build_ep_mesh", lambda: _mesh(ep))
    layer = _layer(slots=4)
    layer.global_num_experts = 4 * ep
    assert not _prebuild(layer)
    assert "local_experts" not in armed


def _ffn_build_admits(hidden, inter):
    """Whether the FFN build's own check passes these widths, all else valid."""
    try:
        ffn_kernel._check_build(
            tile_rows=v3.TILE_ROWS,
            hidden=hidden,
            inter=inter,
            routed_rows=v3.TILE_ROWS,
            weight_slots=2,
            weight_prefetch=1,
            result_slots=2,
            row_fetch_ahead=1,
            activation_block=0,
            token_block=None,
        )
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(
    ("hidden", "inter", "admitted"),
    [(64, 256, False), (192, 256, False), (128, 64, False), (128, 128, True)],
)
def test_widths_the_kernel_ffn_refuses_are_refused(armed, hidden, inter, admitted):
    """Rows of whole 128-lane blocks and an intermediate width of at least one
    block: prebuild admits exactly the widths the FFN build admits."""
    assert _ffn_build_admits(hidden, inter) is admitted
    assert _prebuild(_layer(slots=4, hidden=hidden, inter=inter)) is admitted


def test_experts_whose_weight_slots_do_not_fit_vmem_are_refused(armed):
    """Two whole experts of 7168 x 2048 FP8 (44 MiB each) do not fit TPU7x's
    VMEM, which the layer's first forward would raise on."""
    layer = _big_layer(4, 7168, 2048)
    assert not _prebuild(layer)
    assert "local_experts" not in armed


def test_supported_answers_per_layer_after_prebuild(armed):
    assert not v3.fused_moe_ep_supported()
    armed_layer, refused_layer = _layer(slots=4), _layer(slots=4)
    assert _prebuild(armed_layer)
    assert not _prebuild(refused_layer, renormalize=False)
    assert v3.fused_moe_ep_supported()
    assert v3.fused_moe_ep_supported(armed_layer)
    assert not v3.fused_moe_ep_supported(refused_layer)


TPU7X_RANK_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)


@pytest.mark.parametrize(
    ("order", "expected"),
    [(None, None), (tuple(range(8)), None), (TPU7X_RANK_ORDER, TPU7X_RANK_ORDER)],
    ids=["no_ep", "identity", "tpu7x_host"],
)
def test_the_expert_block_permutation(monkeypatch, order, expected):
    monkeypatch.setattr(v3, "ep_rank_order", lambda: order)
    permutation = v3._expert_block_permutation(8, "cpu")
    if expected is None:
        assert permutation is None
    else:
        assert permutation.dtype == torch.int32
        assert tuple(permutation.tolist()) == expected


@pytest.mark.parametrize("order", [(0, 0, 1, 2), (0, 1, 2)])
def test_an_ep_rank_order_that_is_not_a_permutation_is_refused(monkeypatch, order):
    monkeypatch.setattr(v3, "ep_rank_order", lambda: order)
    with pytest.raises(ValueError, match="is not a permutation of range"):
        v3._expert_block_permutation(4, "cpu")


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_router_logits_move_to_mesh_order(monkeypatch, compiled):
    """On a TPU7x host EP rank 2's experts sit at mesh index 6: mesh index m
    reads the expert block of EP rank order[m], for every token count of one
    compiled graph."""
    ep, local = 8, 3
    monkeypatch.setattr(v3, "_EP_SIZE", ep)
    monkeypatch.setattr(
        v3, "_EXPERT_ORDER", torch.tensor(TPU7X_RANK_ORDER, dtype=torch.int32)
    )
    reorder = v3._to_mesh_expert_order
    if compiled:
        torch._dynamo.reset()
        reorder = torch.compile(reorder, backend="eager", fullgraph=True, dynamic=True)
    for tokens in (5, 9):
        # Logit (t, r * local + j) = 100 * r + j + 1000 * t: EP rank r, expert j.
        rank_expert = torch.arange(ep * local).view(ep, local)
        ids = (100 * (rank_expert // local) + rank_expert % local).flatten()
        logits = (ids + 1000 * torch.arange(tokens)[:, None]).float()
        out = reorder(logits).view(tokens, ep, local)
        for m, r in enumerate(TPU7X_RANK_ORDER):
            torch.testing.assert_close(out[:, m], logits.view(tokens, ep, local)[:, r])


def test_router_logits_keep_their_order_without_a_permutation(monkeypatch):
    monkeypatch.setattr(v3, "_EXPERT_ORDER", None)
    logits = torch.randn(4, 16)
    assert v3._to_mesh_expert_order(logits) is logits


@pytest.fixture
def padding(monkeypatch):
    """The runner's padding state, real: 3 valid rows of 8."""
    monkeypatch.setattr(token_padding, "_padding_state", None)
    state = token_padding.TokenPaddingState.create(8, "cpu")
    token_padding.set_padding_state(state)
    token_padding.mark_padding(3, 8)
    return state


@pytest.fixture
def built_op(monkeypatch):
    """`_build_op`'s real op function around a kernel that records its call:
    what `fused_moe_ep`'s operands arrive at `fused_ep_moe` as."""
    for name in ("_OP", "_SKIP_PADDED_TOKENS", "_RANK_BUFFER", "_EXPERT_ORDER"):
        monkeypatch.setattr(v3, name, getattr(v3, name))
    monkeypatch.setattr(v3, "_EP_SIZE", EP)
    calls = []
    monkeypatch.setattr(
        v3, "fused_ep_moe", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    def sharded_jax_op(name, fn, *, input_partition_specs, fake, **_):
        # One spec per operand of the op function, as the torch op is called.
        assert len(input_partition_specs) == len(inspect.signature(fn).parameters)
        return fn

    monkeypatch.setattr(v3, "sharded_jax_op", sharded_jax_op)

    def build(skip_padded_tokens):
        v3._build_op(
            _mesh(),
            2,
            True,
            "silu",
            skip_padded_tokens,
            hidden=128,
            inter=256,
            local_experts=4,
        )
        return calls

    return build


def _layer_operands(tokens=8):
    layer = _layer(slots=4)
    return (
        torch.randn(tokens, 128),
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        torch.randn(tokens, 4 * EP),
    )


@pytest.mark.parametrize("skip_padded_tokens", [False, True], ids=["all", "valid"])
def test_fused_moe_ep_hands_the_kernel_its_operands(
    built_op, padding, monkeypatch, skip_padded_tokens
):
    """Each operand reaches its kernel argument through the op function the
    padding setting picked: the rank buffer is the shard index, the valid
    count goes only with padding skipped, the router logits arrive in mesh
    order, the scales per output channel, and NaN rows are dropped either
    way."""
    calls = built_op(skip_padded_tokens)
    rank = torch.tensor([[1]], dtype=torch.int32)
    monkeypatch.setattr(v3, "_RANK_BUFFER", rank)
    monkeypatch.setattr(v3, "_EXPERT_ORDER", torch.tensor([1, 0], dtype=torch.int32))
    x, w1, w2, s1, s2, logits = _layer_operands()
    v3.fused_moe_ep(x, w1, w2, s1, s2, logits)
    ((args, kwargs),) = calls
    assert args[0] is x and args[1] is w1 and args[2] is w2
    assert tuple(args[3].shape) == (4, 512) and tuple(args[4].shape) == (4, 128)
    torch.testing.assert_close(args[5], logits.view(8, 2, 4).flip(1).reshape(8, 8))
    assert kwargs["shard_index"] is rank
    if skip_padded_tokens:
        assert kwargs["valid_tokens"] is padding.local_valid_tokens
        assert kwargs["valid_tokens"].tolist() == [3]
    else:
        assert kwargs["valid_tokens"] is None
    assert kwargs["drop_nan_rows"] is True


def test_skipping_padding_without_the_runner_state_raises(built_op, monkeypatch):
    built_op(True)
    monkeypatch.setattr(token_padding, "_padding_state", None)
    with pytest.raises(RuntimeError, match="did not install padding state"):
        v3.fused_moe_ep(*_layer_operands())
    with pytest.raises(RuntimeError, match="did not install padding state"):
        v3.retain_fused_ep_graph_inputs(torch.zeros(4, 8))


@pytest.mark.parametrize("skip_padded_tokens", [False, True], ids=["all", "valid"])
def test_the_graph_reads_the_buffers_and_keeps_the_output(
    padding, monkeypatch, skip_padded_tokens
):
    """The rank buffer, the expert order and, with padding skipped, the valid
    count become inputs the compiled graph computes the MoE output from,
    which comes out bit-identical."""
    monkeypatch.setattr(v3, "_SKIP_PADDED_TOKENS", skip_padded_tokens)
    monkeypatch.setattr(v3, "_RANK_BUFFER", torch.tensor([[5]], dtype=torch.int32))
    monkeypatch.setattr(v3, "_EXPERT_ORDER", torch.tensor([1, 0], dtype=torch.int32))
    graph_inputs = []
    torch._dynamo.reset()

    def backend(gm, example_inputs):
        # The graph inputs the output is computed from.
        (output,) = [node for node in gm.graph.nodes if node.op == "output"]
        seen, todo = set(), [output]
        while todo:
            for node in todo.pop().all_input_nodes:
                if node not in seen:
                    seen.add(node)
                    todo.append(node)
        graph_inputs.append(sum(node.op == "placeholder" for node in seen))
        return gm.forward

    out = torch.randn(4, 8, dtype=torch.bfloat16)
    compiled = torch.compile(
        v3.retain_fused_ep_graph_inputs, backend=backend, fullgraph=True
    )
    assert torch.equal(compiled(out), out)
    # The output, the rank buffer, the expert order, and the valid count.
    assert graph_inputs == [4 if skip_padded_tokens else 3]
