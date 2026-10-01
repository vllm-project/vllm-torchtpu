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
"""Selection, admission and dispatch of the fused EP MoE kernel v3.

None of this needs a TPU: every refusal is decided before an op is built,
and the op itself is replaced by a stub.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as legacy
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as bridge
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization import (
    fused_moe_ep_v3_dispatch as dispatch,
)

pytestmark = pytest.mark.cpu_test

_EP = 8


def _layer(experts=4, hidden=512, inter=256, **overrides):
    """A layer whose weights and routing v3 would accept."""
    layer = torch.nn.Module()
    for name, tensor in dict(
        w13_weight=torch.zeros(experts, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(experts, inter, hidden, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(experts, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(experts, 1, 1, hidden),
    ).items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.activation = "silu"
    layer.renormalize = True
    layer.global_num_experts = experts * _EP
    layer.custom_routing_function = None
    layer.scoring_func = "softmax"
    layer.e_score_correction_bias = None
    layer.use_grouped_topk = False
    layer.num_expert_group = None
    layer.topk_group = None
    layer.routed_scaling_factor = 1.0
    layer.apply_router_weight_on_input = False
    layer.moe_config = SimpleNamespace(
        experts_per_token=2,
        ep_rank=0,
        ep_size=_EP,
        moe_parallel_config=SimpleNamespace(
            use_ep=True, pcp_size=1, is_sequence_parallel=False
        ),
    )
    layer.register_buffer("_experts_start", torch.tensor(0, dtype=torch.int32))
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


@pytest.fixture
def v3(monkeypatch):
    """IMPL=v3 with the distributed lookups and the op build stubbed out."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")
    monkeypatch.setattr(bridge, "_OPS", {})
    monkeypatch.setattr(
        bridge,
        "build_ep_mesh",
        lambda: SimpleNamespace(shape={bridge.EP_AXIS_NAME: _EP}),
    )
    monkeypatch.setattr(bridge, "ep_mesh_index", lambda: 0)
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1))
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.pipelined_fused_moe."
        "enable_pipelined_collective_and_compute",
        lambda: False,
    )
    built = []

    def build_op(*args, **kwargs):
        op = Mock(side_effect=lambda x, *rest: x)
        built.append((args, kwargs, op))
        return op

    monkeypatch.setattr(bridge, "_build_op", build_op)
    return SimpleNamespace(built=built, config=config)


def _prebuild(layer, **kwargs):
    defaults = dict(topk=2, renormalize=True, activation="silu")
    defaults.update(kwargs)
    return bridge.prebuild_fused_moe_ep_v3(layer, **defaults)


@pytest.mark.parametrize(
    "selection,enabled",
    [(None, False), ("v2", True), ("adaptive", True), ("v3", True)],
)
def test_v3_is_a_selector_choice_that_enables_fused_ep(monkeypatch, selection, enabled):
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    if selection is None:
        monkeypatch.delenv("MOE_FUSED_EP_KERNEL_IMPL", raising=False)
    else:
        monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", selection)
    assert envs.environment_variables["USE_MOE_FUSED_EP_KERNEL"]() is enabled
    assert dispatch.enabled() is (selection == "v3")


def test_the_op_name_does_not_clash_with_v2():
    assert bridge.FUSED_MOE_EP_V3_OP_NAME != legacy.FUSED_MOE_EP_OP_NAME
    assert bridge.FUSED_MOE_EP_V3_OP_ATTR != legacy.FUSED_MOE_EP_OP_ATTR


def test_arms_on_a_supported_layer(v3):
    armed = _prebuild(_layer())
    assert armed.run is v3.built[0][2]
    assert armed.step_min_tokens == bridge.STEP_MIN_TOKENS_DEFAULT == 33
    args, kwargs, _ = v3.built[0]
    assert args[1:5] == (2, True, "silu", False)
    assert kwargs == dict(hidden=512, inter=256, local_experts=4)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("tp", "tensor_parallel_size=8"),
        ("pcp", "prefill context parallelism"),
        ("no_ep", "--enable-expert-parallel"),
        ("pipelined", "chunk pipelining"),
    ],
)
def test_an_unservable_deployment_raises(v3, monkeypatch, fault, match):
    layer = _layer()
    if fault == "tp":
        v3.config.parallel_config.tensor_parallel_size = 8
    elif fault == "pcp":
        layer.moe_config.moe_parallel_config.pcp_size = 2
    elif fault == "no_ep":
        layer.moe_config.moe_parallel_config.use_ep = False
    else:
        monkeypatch.setattr(
            "vllm_torchtpu.layers.adapter.pipelined_fused_moe."
            "enable_pipelined_collective_and_compute",
            lambda: True,
        )
    with pytest.raises(NotImplementedError, match=match):
        _prebuild(layer)
    assert not v3.built


@pytest.mark.parametrize(
    "overrides",
    [
        dict(scoring_func="sigmoid"),
        dict(e_score_correction_bias=torch.zeros(32)),
        dict(routed_scaling_factor=2.5),
        dict(use_grouped_topk=True, num_expert_group=8, topk_group=4),
        dict(custom_routing_function=lambda **kw: None),
        dict(apply_router_weight_on_input=True),
        dict(hash_indices_table=torch.zeros(16, 2, dtype=torch.int32)),
    ],
    ids=["sigmoid", "bias", "scaled", "grouped", "custom", "on_input", "hash"],
)
def test_routing_v2_serves_and_v3_does_not_is_declined(v3, overrides):
    """v2's guard admits score-bias and hash routing; v3 cannot compute them."""
    assert _prebuild(_layer(**overrides)) is None
    assert not v3.built


def test_declined_while_the_routing_simulator_is_on(v3, monkeypatch):
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
    assert _prebuild(_layer()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(global_num_experts=4 * _EP + 1),
        dict(w13_bias=torch.zeros(4, 512)),
    ],
    ids=["uneven", "bias"],
)
def test_an_unservable_layer_layout_is_declined(v3, overrides):
    assert _prebuild(_layer(**overrides)) is None


def test_a_non_silu_activation_is_declined(v3):
    assert _prebuild(_layer(), activation="gelu") is None


def test_a_block_scaled_weight_is_declined(v3):
    layer = _layer()
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones(4, 4, 1, 512), requires_grad=False
    )
    assert _prebuild(layer) is None


def test_the_per_step_gate_replaces_the_deployment_token_threshold(v3, monkeypatch):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", str(1 << 30))
    assert _prebuild(_layer()) is not None


def test_a_tuned_shape_overrides_the_default_gate(v3, monkeypatch):
    monkeypatch.setitem(bridge.STEP_MIN_TOKENS_BY_SHAPE, (512, 256, 4, 2), 64)
    assert _prebuild(_layer()).step_min_tokens == 64


def test_a_layer_outside_the_backbone_graph_is_declined(v3, monkeypatch):
    """The MTP draft is loaded under the "eagle_head" tag and compiled from one
    trace, which cannot carry both sides of the step gate."""
    from vllm.compilation import backends

    monkeypatch.setattr(bridge, "_ARMING", bridge.Counter())
    with backends.set_model_tag("eagle_head"):
        assert _prebuild(_layer()) is None
    assert not v3.built
    assert _prebuild(_layer()) is not None
    (reason,) = [r for r in bridge._ARMING if r != "armed"]
    assert "'eagle_head' graph" in reason


def test_declined_without_an_ep_group(v3, monkeypatch):
    monkeypatch.setattr(bridge, "build_ep_mesh", lambda: None)
    assert _prebuild(_layer()) is None


_MESH = SimpleNamespace(shape={bridge.MESH_AXIS: _EP}, axis_names=(bridge.MESH_AXIS,))


class _FakeOp:
    """What `jax_op` returns, recording the name it was registered under."""

    def __init__(self, name, fn, **kwargs):
        self.name, self.fn, self.kwargs = name, fn, kwargs

    def register_fake(self, fake):
        self.fake = fake

    def __call__(self, *args):
        return args


def test_ops_are_shared_per_closure_and_named_apart_from_v2(monkeypatch):
    """The MTP draft's MoE layer builds its own op when its shape differs and
    shares the target's when it matches; no two closures share a name."""
    monkeypatch.setattr(bridge, "_OPS", {})
    created = []

    class Op(_FakeOp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(bridge, "jax_op", Op)

    def build(inter, skip=False):
        return bridge._build_op(
            _MESH, 2, True, "silu", skip, hidden=512, inter=inter, local_experts=4
        )

    target = build(256)
    assert build(256) is target
    build(512)
    build(256, skip=True)
    assert created[0].name == "pallas::fused_moe_ep_v3"
    assert len({op.name for op in created}) == 3
    assert [len(op.kwargs["input_partition_specs"]) for op in created] == [7, 7, 8]


def _method():
    method = object.__new__(fp8.VllmFp8MoEMethodTPU)
    method.moe = SimpleNamespace(has_bias=False, is_act_and_mul=True)
    method.quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True)
    method.weight_block_size = None
    method.weight_scale_name = "weight_scale"
    return method


@pytest.fixture
def fp8_load(v3, monkeypatch):
    """The FP8 method's load with the weight conversion stubbed and v2 and
    the pipelined path forbidden."""
    forbidden = Mock(side_effect=AssertionError("v2 must not be called"))
    monkeypatch.setattr(legacy, "prebuild_fused_moe_ep", forbidden)
    monkeypatch.setattr(fp8, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(fp8, "prebuild_fused_moe_kernel", Mock())
    monkeypatch.setattr(fp8, "pipelined_fused_moe_gmm", forbidden)
    monkeypatch.setattr(fp8, "enable_pipelined_collective_and_compute", lambda: False)
    monkeypatch.setattr(fp8.moe_routing, "validate_linear_ep_placement", Mock())
    monkeypatch.setattr(fp8.moe_routing, "register_experts_start_buffer", Mock())

    def prepare(layer, **kw):
        return (
            layer.w13_weight,
            layer.w13_weight_scale_inv.data,
            layer.w2_weight,
            layer.w2_weight_scale_inv.data,
            "float8_e4m3fn",
            None,
        )

    monkeypatch.setattr(fp8, "_process_fp8_moe_weights", prepare)
    return v3


def test_fp8_method_loads_and_forwards_through_v3(fp8_load, monkeypatch):
    layer, method = _layer(), _method()
    gmm = Mock(side_effect=AssertionError("GMM must not be called"))
    monkeypatch.setattr(fp8, "fused_moe_gmm", gmm)
    assert not method.supports_internal_mk
    method.process_weights_after_loading(layer)
    op = fp8_load.built[0][2]
    assert getattr(method, bridge.FUSED_MOE_EP_V3_OP_ATTR).run is op
    assert not legacy.fused_moe_ep_supported(method)
    assert method.supports_internal_mk
    x = torch.ones(64, 512, dtype=torch.bfloat16)
    assert method.apply_monolithic(layer, x, torch.ones(64, 32)) is x
    call = op.call_args.args
    assert len(call) == 7
    assert call[6].shape == (1, 1)
    assert call[3].shape == (4, 512) and call[4].shape == (4, 512)


def test_fp8_method_keeps_the_grouped_path_for_a_declined_layer(fp8_load, monkeypatch):
    layer, method = _layer(scoring_func="sigmoid"), _method()
    monkeypatch.setattr(fp8, "fused_moe_gmm", Mock(return_value="gmm"))
    monkeypatch.setattr(fp8.moe_routing, "route", Mock(return_value=("weights", "ids")))
    layer._experts_start = None
    method.process_weights_after_loading(layer)
    assert getattr(method, bridge.FUSED_MOE_EP_V3_OP_ATTR) is None
    assert not method.supports_internal_mk
    x = torch.ones(3, 512, dtype=torch.bfloat16)
    assert method.apply_monolithic(layer, x, torch.ones(3, 32)) == "gmm"


def test_skip_padded_tokens_is_fixed_at_build_and_passes_the_padding_mask(
    monkeypatch,
):
    monkeypatch.setattr(bridge, "_OPS", {})
    monkeypatch.setattr(bridge, "jax_op", _FakeOp)
    run = bridge._build_op(
        _MESH, 2, True, "silu", True, hidden=512, inter=256, local_experts=4
    )
    owner = SimpleNamespace()
    setattr(owner, bridge.FUSED_MOE_EP_V3_OP_ATTR, bridge.ArmedOp(run, 1))
    layer = _layer()
    state = token_padding.TokenPaddingState.create(8, device="cpu")
    state.update(5, 8)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    # The forward follows the build, not the environment.
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")
    call = bridge.fused_moe_ep_v3(
        owner,
        torch.ones(8, 512, dtype=torch.bfloat16),
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        torch.ones(8, 32),
        layer._experts_start,
    )
    assert len(call) == 8
    assert call[6].shape == (1, 1)
    assert call[7].tolist() == [0] * 5 + [1] * 3


@pytest.mark.parametrize("rank", range(_EP))
def test_sequence_parallel_rows_take_their_rank_slice_of_the_mask(monkeypatch, rank):
    """On a sequence-parallel step each rank holds a contiguous slice of the
    step's rows; the padded rows it skips are that slice of the step's mask."""
    from vllm_torchtpu.layers.adapter import sequence_parallel

    monkeypatch.setattr(bridge, "_OPS", {})
    monkeypatch.setattr(bridge, "jax_op", _FakeOp)
    monkeypatch.setattr(sequence_parallel, "_active", True)
    run = bridge._build_op(
        _MESH, 2, True, "silu", True, hidden=512, inter=256, local_experts=4
    )
    owner = SimpleNamespace()
    setattr(owner, bridge.FUSED_MOE_EP_V3_OP_ATTR, bridge.ArmedOp(run, 1))
    layer = _layer()
    local_rows, valid = 8, 37
    state = token_padding.TokenPaddingState.create(_EP * local_rows, device="cpu")
    state.update(valid, _EP * local_rows)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    call = bridge.fused_moe_ep_v3(
        owner,
        torch.ones(local_rows, 512, dtype=torch.bfloat16),
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        torch.ones(local_rows, 32),
        torch.tensor(rank * 4, dtype=torch.int32),
    )
    step_mask = state.local_padding_mask.to(torch.int32)
    expected = step_mask[rank * local_rows : (rank + 1) * local_rows]
    assert call[7].tolist() == expected.tolist()


def test_mask_padded_rows_routes_the_suffix_nowhere():
    ep, t_local, hidden, experts = 2, 4, 8, 4
    x = jnp.ones((ep * t_local, hidden), jnp.bfloat16)
    gating = jnp.zeros((ep * t_local, experts), jnp.float32)
    padded = jnp.asarray([0, 0, 0, 1, 0, 1, 1, 1], jnp.int32)
    x, gating = bridge.mask_padded_rows(x, gating, padded)
    keep = np.array([1, 1, 1, 0, 1, 0, 0, 0], bool)
    assert np.all(np.isnan(np.asarray(gating)[~keep]))
    assert not np.any(np.isnan(np.asarray(gating)[keep]))
    assert not np.any(np.asarray(x, np.float32)[~keep])
    assert np.all(np.asarray(x, np.float32)[keep] == 1)


def test_the_runner_reports_an_armed_v3_layer_reduced(monkeypatch):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    import vllm_torchtpu

    class Runner:
        _fused_output_is_reduced = property(lambda self: False)

    monkeypatch.setattr(mr, "MoERunner", Runner)
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    vllm_torchtpu._patch_moe_runner_fused_output_is_reduced()
    runner = Runner()
    runner.moe_config = SimpleNamespace(skip_final_all_reduce=False)
    runner._quant_method = SimpleNamespace()
    assert runner._fused_output_is_reduced is False
    setattr(runner._quant_method, bridge.FUSED_MOE_EP_V3_OP_ATTR, object())
    assert runner._fused_output_is_reduced is True


class _FakeGroup:
    """A two-rank DP/EP group: dispatch gathers, combine keeps this rank's."""

    world_size = 2

    def __init__(self):
        self.calls = []

    def dispatch_router_logits(self, x, logits, sequence_parallel):
        self.calls.append(("dispatch", x.shape[0], sequence_parallel))
        return torch.cat([x, x]), torch.cat([logits, logits])

    def combine(self, x, sequence_parallel):
        self.calls.append(("combine", x.shape[0], sequence_parallel))
        return x[: x.shape[0] // 2]

    def all_gather(self, x, dim=0):
        return torch.cat([x, x], dim=dim)


@pytest.fixture
def armed_fp8(fp8_load, monkeypatch):
    """An armed FP8 layer with the grouped path and the EP group stubbed."""
    layer, method = _layer(), _method()
    method.process_weights_after_loading(layer)
    group = _FakeGroup()
    monkeypatch.setattr(dispatch, "get_ep_group", lambda: group)
    gmm = Mock(side_effect=lambda **kw: kw["hidden_states"])
    monkeypatch.setattr(fp8, "fused_moe_gmm", gmm)
    monkeypatch.setattr(
        fp8.moe_routing,
        "route",
        lambda layer, x, logits: torch.topk(logits.float().softmax(-1), 2),
    )
    return SimpleNamespace(
        layer=layer, method=method, group=group, gmm=gmm, v3=fp8_load.built[0][2]
    )


@pytest.mark.parametrize("rows,takes_v3", [(1, False), (32, False), (33, True)])
def test_the_gate_picks_v3_or_the_grouped_path_per_step(armed_fp8, rows, takes_v3):
    a = armed_fp8
    x = torch.ones(rows, 512, dtype=torch.bfloat16)
    out = a.method.apply_monolithic(a.layer, x, torch.ones(rows, 32))
    assert out.shape == x.shape
    assert a.v3.called is takes_v3
    assert a.gmm.called is not takes_v3
    if takes_v3:
        assert a.group.calls == []
    else:
        # vLLM's own exchange around the grouped path, on the gathered rows.
        assert a.group.calls == [
            ("dispatch", rows, False),
            ("combine", 2 * rows, False),
        ]
        assert a.gmm.call_args.kwargs["hidden_states"].shape[0] == 2 * rows


def test_tp_is_served_for_a_sequence_parallel_layer(v3):
    """Under TP a sequence-parallel layer's large steps give each rank its own
    rows (layers/adapter/sequence_parallel)."""
    v3.config.parallel_config.tensor_parallel_size = 8
    layer = _layer()
    layer.moe_config.moe_parallel_config.is_sequence_parallel = True
    assert _prebuild(layer) is not None


def test_a_draft_layer_outside_sequence_parallelism_is_declined(v3, monkeypatch):
    """The MTP draft next to a TP8 sequence-parallel target runs the plain TP
    form: its MoE layer is declined, not refused. Here the replicated draft
    (draft_tensor_parallel_size=1), built without expert parallelism."""
    monkeypatch.setenv("TPU_SEQUENCE_PARALLEL_PREFILL", "1")
    monkeypatch.setattr(bridge, "_ARMING", bridge.Counter())
    v3.config.parallel_config.tensor_parallel_size = 8
    layer = _layer()
    # Built under a one-rank TP group: no expert parallelism either.
    layer.moe_config.moe_parallel_config.use_ep = False
    assert _prebuild(layer) is None
    assert not v3.built
    assert bridge._ARMING == {"an MTP draft under TP, outside sequence parallelism": 1}


@pytest.mark.parametrize(
    "active,rows,takes_v3",
    [(True, 32, False), (True, 33, True), (False, 33, False), (False, 512, False)],
    ids=["sliced-below-gate", "sliced", "replicated", "replicated-large"],
)
def test_a_sequence_parallel_layer_runs_v3_on_sliced_steps_only(
    armed_fp8, monkeypatch, active, rows, takes_v3
):
    from vllm_torchtpu.layers.adapter import sequence_parallel

    a = armed_fp8
    a.layer.moe_config.moe_parallel_config.is_sequence_parallel = True
    monkeypatch.setattr(sequence_parallel, "_active", active)
    x = torch.ones(rows, 512, dtype=torch.bfloat16)
    a.method.apply_monolithic(a.layer, x, torch.ones(rows, 32))
    assert a.v3.called is takes_v3
    assert a.gmm.called is not takes_v3
    if not takes_v3:
        # The communicator's sequence-parallel branches serve the exchange.
        assert [call[2] for call in a.group.calls] == [True, True]


@torch.library.custom_op("v3_gate_test::fused", mutates_args=())
def _fused_stub(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    gating: torch.Tensor,
    experts_start: torch.Tensor,
) -> torch.Tensor:
    return x.clone()


@torch.library.custom_op("v3_gate_test::fused_padded", mutates_args=())
def _fused_padded_stub(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    gating: torch.Tensor,
    experts_start: torch.Tensor,
    padded: torch.Tensor,
) -> torch.Tensor:
    return x.clone()


@torch.library.custom_op("v3_gate_test::gmm", mutates_args=())
def _gmm_stub(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    experts_start: torch.Tensor,
) -> torch.Tensor:
    return x.clone()


for _stub in (_fused_stub, _fused_padded_stub, _gmm_stub):
    _stub.register_fake(lambda x, *args: torch.empty_like(x))


class _StubOp:
    """What the patched `jax_op` returns: the stub op of the built arity."""

    def __init__(self, name, fn, **kwargs):
        padded = len(kwargs["input_partition_specs"]) == 8
        self.op = _fused_padded_stub if padded else _fused_stub

    def register_fake(self, fake):
        pass

    def __call__(self, *args):
        return self.op(*args)


_REAL_BUILD_OP = bridge._build_op


@pytest.mark.parametrize("skip_padded", [False, True], ids=["plain", "padded"])
def test_both_sides_of_the_gate_lift_the_same_inputs(
    armed_fp8, monkeypatch, skip_padded
):
    """A bucket compiled by the trace on one side of the gate is handed to
    the trace on the other (compilation/shape_variants): their graphs must
    take the same inputs by name, at most reordered."""
    from vllm_torchtpu.compilation import shape_variants

    a = armed_fp8
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1" if skip_padded else "0")
    monkeypatch.setattr(bridge, "jax_op", _StubOp)
    monkeypatch.setattr(bridge, "_OPS", {})
    run = _REAL_BUILD_OP(
        _MESH, 2, True, "silu", skip_padded, hidden=512, inter=256, local_experts=4
    )
    setattr(a.method, bridge.FUSED_MOE_EP_V3_OP_ATTR, bridge.ArmedOp(run, 33))
    monkeypatch.setattr(
        fp8,
        "fused_moe_gmm",
        lambda **kw: _gmm_stub(
            kw["hidden_states"],
            kw["w1"],
            kw["w2"],
            kw["w1_scale"],
            kw["w2_scale"],
            kw["topk_weights"],
            kw["topk_ids"],
            kw["experts_start"],
        ),
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.get_dp_group", lambda: a.group)
    state = token_padding.TokenPaddingState.create(256, device="cpu")
    monkeypatch.setattr(token_padding, "_padding_state", state)

    traces = []

    def backend(graph, example_inputs):
        traces.append(shape_variants._handover_inputs(graph))
        return graph.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda x, g: a.method.apply_monolithic(a.layer, x, g),
        backend=backend,
        fullgraph=True,
    )
    for rows in (16, 64):
        # The token dimension is dynamic, as vLLM traces the backbone.
        x, g = torch.ones(rows, 512, dtype=torch.bfloat16), torch.ones(rows, 32)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(g, 0)
        compiled(x, g)
    torch._dynamo.reset()
    assert len(traces) == 2
    below, above = traces
    names = {name for name, _ in below}
    assert any("experts_start" in name for name in names)
    assert any("padding_mask" in name for name in names) is skip_padded
    assert sorted(below) == sorted(above)
    assert shape_variants._input_permutation(below, above) is not None


def test_the_boot_summary_counts_armed_and_declined_layers(v3, monkeypatch):
    monkeypatch.setattr(bridge, "_ARMING", bridge.Counter())
    for _ in range(3):
        _prebuild(_layer())
    _prebuild(_layer(scoring_func="sigmoid"))
    info = Mock()
    monkeypatch.setattr(bridge.logger, "info", info)
    bridge.log_arming_summary()
    message = info.call_args.args[0] % info.call_args.args[1:]
    assert message.startswith("fused EP v3: 3/4 MoE layers armed; declined: 1x ")
    assert "scoring_func 'sigmoid'" in message
