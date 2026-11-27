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
"""Tests for arming and dispatching the MoE combine's hierarchical reduce-scatter.

The op itself runs on real collectives in test_hier_reduce_scatter_multichip.py.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import hier_reduce_scatter as hrs

RANKS = tuple(range(8))
HIDDEN = 512


class _Group:
    def __init__(self, ranks=RANKS):
        self.ranks = list(ranks)
        self.world_size = len(self.ranks)
        self.calls = 0

    def reduce_scatter(self, x, dim):
        assert dim == 0
        self.calls += 1
        return x[: x.shape[0] // self.world_size]

    def all_gather(self, x, dim):
        assert dim == 0
        return x


class _Op:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x[: x.shape[0] // 8]


@pytest.fixture
def armed(monkeypatch):
    op = _Op()
    monkeypatch.setattr(hrs, "_ARMED", hrs._Armed(op=op, ranks=RANKS))
    return op


def test_not_armed_uses_group(monkeypatch):
    monkeypatch.setattr(hrs, "_ARMED", None)
    group = _Group()
    hrs.moe_combine_reduce_scatter(group, torch.zeros(64, HIDDEN, dtype=torch.bfloat16))
    assert group.calls == 1


def test_armed_and_eligible_uses_kernel(armed):
    group = _Group()
    out = hrs.moe_combine_reduce_scatter(
        group, torch.zeros(64, HIDDEN, dtype=torch.bfloat16)
    )
    assert armed.calls == 1 and group.calls == 0
    assert out.shape == (8, HIDDEN)


@pytest.mark.parametrize(
    "x, group",
    [
        pytest.param(
            torch.zeros(64, HIDDEN, dtype=torch.float32), _Group(), id="not_bf16"
        ),
        pytest.param(
            torch.zeros(64, 4, 4, dtype=torch.bfloat16), _Group(), id="not_2d"
        ),
        pytest.param(
            torch.zeros(60, HIDDEN, dtype=torch.bfloat16),
            _Group(),
            id="rows_not_divisible",
        ),
        pytest.param(
            torch.zeros(56, HIDDEN, dtype=torch.bfloat16),
            _Group(),
            id="under_8_output_rows",
        ),
        pytest.param(
            torch.zeros(64, 2816, dtype=torch.bfloat16),
            _Group(),
            id="hidden_not_splittable",
        ),
        pytest.param(
            torch.zeros(64, HIDDEN, dtype=torch.bfloat16),
            _Group(ranks=range(8, 16)),
            id="other_group",
        ),
    ],
)
def test_ineligible_calls_use_group(armed, x, group):
    hrs.moe_combine_reduce_scatter(group, x)
    assert armed.calls == 0 and group.calls == 1


# A world the kernel supports: one tpu7x 2x2x1 host, EP = DP = world = 8.
_SUPPORTED = dict(
    ep=SimpleNamespace(world_size=8, ranks=list(RANKS)),
    world_size=8,
    dp_ranks=list(RANKS),
    topology=SimpleNamespace(num_chips=4, cores_per_chip=2),
    chip="7x",
)


def _set_world(monkeypatch, **overrides):
    import torch.distributed as dist
    from vllm.distributed import parallel_state

    from vllm_torchtpu.distributed import chip_topology, ep_mesh

    world = {**_SUPPORTED, **overrides}
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: world["ep"])
    monkeypatch.setattr(dist, "get_world_size", lambda: world["world_size"])
    monkeypatch.setattr(
        parallel_state,
        "get_dp_group",
        lambda: SimpleNamespace(ranks=world["dp_ranks"]),
    )
    monkeypatch.setattr(chip_topology, "get_chip_topology", lambda: world["topology"])
    from jax.experimental.pallas import tpu as pltpu

    monkeypatch.setattr(hrs, "_chip_version", lambda: pltpu.ChipVersion(world["chip"]))


def test_world_reason_accepts_supported_world(monkeypatch):
    _set_world(monkeypatch)
    assert hrs._unsupported_world_reason(draft_model=False) is None


def test_world_reason_refuses_a_draft_model(monkeypatch):
    _set_world(monkeypatch)
    got = hrs._unsupported_world_reason(draft_model=True)
    assert got is not None and "draft model" in got


@pytest.mark.parametrize(
    "overrides, reason",
    [
        pytest.param(dict(ep=None), "expert parallelism is off", id="ep_off"),
        pytest.param(
            dict(ep=SimpleNamespace(world_size=4, ranks=[0, 1, 2, 3]), world_size=4),
            "EP group of 4",
            id="ep_size",
        ),
        pytest.param(
            dict(world_size=16), "does not span the whole", id="ep_not_whole_world"
        ),
        pytest.param(
            dict(dp_ranks=[0, 2, 4, 6, 1, 3, 5, 7]),
            "DP group's ranks differ",
            id="dp_ranks",
        ),
        pytest.param(dict(topology=None), "topology is unknown", id="no_topology"),
        pytest.param(
            dict(topology=SimpleNamespace(num_chips=8, cores_per_chip=1)),
            "not 8 of 1",
            id="single_core_chips",
        ),
        pytest.param(
            dict(topology=SimpleNamespace(num_chips=2, cores_per_chip=2)),
            "not 2 of 2",
            id="two_chips",
        ),
        pytest.param(dict(chip="v6e"), "tuned for tpu7x", id="not_tpu7x"),
    ],
)
def test_world_reason_refuses(monkeypatch, overrides, reason):
    _set_world(monkeypatch, **overrides)
    got = hrs._unsupported_world_reason(draft_model=False)
    assert got is not None and reason in got


def _mesh(device_ids):
    import numpy as np

    return SimpleNamespace(
        devices=np.asarray([SimpleNamespace(id=i) for i in device_ids], dtype=object)
    )


@pytest.fixture
def arming(monkeypatch):
    """A supported world, with the op build and the EP mesh stubbed out."""
    from vllm_torchtpu.distributed import ep_mesh

    monkeypatch.setattr(hrs, "_ARMED", None)
    monkeypatch.setattr(hrs, "_REFUSED", None)
    monkeypatch.setattr(hrs.envs, "TPU_MOE_EP_HIER_RS", "bf16", raising=False)
    monkeypatch.setattr(hrs, "_unsupported_world_reason", lambda draft_model: None)
    state = SimpleNamespace(builds=[], mesh=_mesh(range(8)), gathers=0, peers=[])

    def gather(reason):
        state.gathers += 1
        return [reason, *state.peers]

    monkeypatch.setattr(hrs, "_gather_reasons", gather)
    monkeypatch.setattr(ep_mesh, "build_ep_mesh", lambda axis_name: state.mesh)
    monkeypatch.setattr(ep_mesh, "ep_rank_order", lambda: (0, 1, 6, 7, 2, 3, 4, 5))
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: _Group())

    def build(mesh, order, axis_name, **wire):
        state.builds.append((mesh, order, axis_name, wire))
        return _Op()

    monkeypatch.setattr(hrs, "_build_op", build)
    return state


def test_prebuild_is_a_noop_when_off(monkeypatch):
    monkeypatch.setattr(hrs, "_ARMED", None)
    monkeypatch.setattr(hrs, "_REFUSED", None)
    monkeypatch.setattr(hrs.envs, "TPU_MOE_EP_HIER_RS", "off", raising=False)
    monkeypatch.setattr(
        hrs,
        "_unsupported_world_reason",
        lambda draft_model: pytest.fail("must not inspect the world when off"),
    )
    assert hrs.prebuild_hier_reduce_scatter() is False


def test_prebuild_arms_once(arming):
    assert hrs.prebuild_hier_reduce_scatter() is True
    assert hrs.prebuild_hier_reduce_scatter() is True
    assert len(arming.builds) == 1
    mesh, order, axis_name, wire = arming.builds[0]
    assert mesh is arming.mesh and order == (0, 1, 6, 7, 2, 3, 4, 5)
    assert wire == dict(fp8_comm=False, fp8_static_scale=None)
    assert hrs._ARMED is not None and hrs._ARMED.ranks == RANKS


def test_prebuild_arms_the_fp8_wire(arming, monkeypatch):
    monkeypatch.setattr(hrs.envs, "TPU_MOE_EP_HIER_RS", "fp8", raising=False)
    monkeypatch.setattr(
        hrs.envs, "TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE", 16.0, raising=False
    )
    assert hrs.prebuild_hier_reduce_scatter() is True
    assert arming.builds[0][3] == dict(fp8_comm=True, fp8_static_scale=16.0)


@pytest.mark.parametrize("scale", [None, 0.0, -1.0])
def test_prebuild_rejects_fp8_without_a_positive_static_scale(monkeypatch, scale):
    monkeypatch.setattr(hrs, "_ARMED", None)
    monkeypatch.setattr(hrs, "_REFUSED", None)
    monkeypatch.setattr(hrs.envs, "TPU_MOE_EP_HIER_RS", "fp8", raising=False)
    monkeypatch.setattr(
        hrs.envs, "TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE", scale, raising=False
    )
    with pytest.raises(ValueError, match="TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE"):
        hrs.prebuild_hier_reduce_scatter()


def test_fp8_wire_without_a_static_scale_is_rejected():
    from vllm_torchtpu.kernels.collectives.hierrs_tc import dispatch

    with pytest.raises(ValueError, match="no dynamic FP8 scale"):
        dispatch.make_reduce_scatter_fn(
            None, tuple(range(8)), axis_name="x", fp8_comm=True
        )


def test_prebuild_refuses_unsupported_world_once(arming, monkeypatch):
    monkeypatch.setattr(hrs, "_unsupported_world_reason", lambda draft_model: "why")
    assert hrs.prebuild_hier_reduce_scatter() is False
    assert hrs.prebuild_hier_reduce_scatter() is False
    assert hrs._ARMED is None and not arming.builds
    assert hrs._REFUSED == "why" and arming.gathers == 1


def test_prebuild_passes_draft_model(arming, monkeypatch):
    seen = []
    monkeypatch.setattr(
        hrs, "_unsupported_world_reason", lambda draft_model: seen.append(draft_model)
    )
    hrs.prebuild_hier_reduce_scatter(draft_model=True)
    assert seen == [True]


def test_prebuild_refuses_mesh_not_in_device_id_order(arming):
    arming.mesh = _mesh((0, 1, 4, 5, 6, 7, 2, 3))
    assert hrs.prebuild_hier_reduce_scatter() is False
    assert hrs._ARMED is None and not arming.builds
    assert "ascending device id" in hrs._REFUSED


def test_prebuild_refuses_on_every_rank_when_one_refuses(arming):
    """A rank that would arm must not wait on one that refused."""
    arming.peers = [None] * 6 + ["the chip topology is unknown"]
    assert hrs.prebuild_hier_reduce_scatter() is False
    assert not arming.builds and "ranks disagree" in hrs._REFUSED


def test_agreed_reason_when_every_rank_refuses_alike(monkeypatch):
    monkeypatch.setattr(hrs, "_gather_reasons", lambda reason: [reason] * 8)
    assert hrs.agreed_refusal_reason("why") == "why"
    assert hrs.agreed_refusal_reason(None) is None


@pytest.mark.parametrize(
    "x, reason",
    [
        pytest.param(torch.zeros(64, HIDDEN), "float32", id="not_bf16"),
        pytest.param(torch.zeros(64, 4, 4, dtype=torch.bfloat16), "3 dims", id="3d"),
        pytest.param(
            torch.zeros(64, 2816, dtype=torch.bfloat16), "hidden size 2816", id="hidden"
        ),
        pytest.param(
            torch.zeros(56, HIDDEN, dtype=torch.bfloat16), "row count", id="rows"
        ),
    ],
)
def test_fallback_logs_its_reason(armed, monkeypatch, x, reason):
    logged = []
    monkeypatch.setattr(hrs, "_log_fallback", logged.append)
    hrs.moe_combine_reduce_scatter(_Group(), x)
    assert len(logged) == 1 and reason in logged[0]


def test_fallback_log_compiles_without_a_graph_break(armed):
    group = _Group()
    fn = torch.compile(
        lambda x: hrs.moe_combine_reduce_scatter(group, x),
        backend="eager",
        fullgraph=True,
    )
    assert fn(torch.zeros(64, 2816, dtype=torch.bfloat16)).shape == (8, 2816)
    assert armed.calls == 0


def test_load_model_arms_through_the_runner(monkeypatch):
    import inspect

    from vllm_torchtpu.runner import tpu_runner

    runner_cls = tpu_runner.TPUModelRunner
    assert "self._arm_hier_reduce_scatter()" in inspect.getsource(runner_cls.load_model)
    seen = []
    monkeypatch.setattr(
        hrs,
        "prebuild_hier_reduce_scatter",
        lambda draft_model=False: seen.append(draft_model),
    )
    runner = object.__new__(runner_cls)
    for spec, draft in (
        (None, False),
        (SimpleNamespace(method="ngram"), False),
        (SimpleNamespace(method="mtp"), True),
    ):
        runner.speculative_config = spec
        runner._arm_hier_reduce_scatter()
        assert seen.pop() is draft


def test_communicator_combine_routes_through_dispatcher(monkeypatch):
    from vllm_torchtpu.distributed import tpu_communicator

    calls = []
    monkeypatch.setattr(
        tpu_communicator,
        "moe_combine_reduce_scatter",
        lambda group, x: calls.append((group, x)) or x[:1],
    )
    comm = object.__new__(tpu_communicator.TpuDeviceCommunicator)
    group = _Group()
    comm._moe_group = lambda is_sequence_parallel: group
    x = torch.zeros(64, HIDDEN, dtype=torch.bfloat16)
    comm.combine(x)
    assert len(calls) == 1 and calls[0][0] is group and calls[0][1] is x


@pytest.mark.parametrize("num_chunks", [1, 2])
def test_pipelined_moe_routes_single_chunk_through_dispatcher(monkeypatch, num_chunks):
    """One chunk goes through the dispatcher; the chunk pipeline keeps the
    group's collective, which overlaps the next chunk's compute."""
    from vllm_torchtpu.layers.adapter import pipelined_fused_moe as pfm

    group = _Group()
    calls = []
    monkeypatch.setattr(pfm, "_get_moe_collective_group", lambda: (group, "DP"))
    monkeypatch.setattr(
        pfm, "collection_chunks", lambda seq, size: (num_chunks, seq // num_chunks)
    )
    monkeypatch.setattr(pfm, "fused_moe_gmm", lambda **kw: kw["hidden_states"])
    monkeypatch.setattr(
        pfm,
        "moe_combine_reduce_scatter",
        lambda g, x: calls.append(g) or x[: x.shape[0] // 8],
    )
    seq = 64
    pfm.pipelined_fused_moe_gmm(
        hidden_states=torch.zeros(seq, HIDDEN, dtype=torch.bfloat16),
        w1=None,
        w2=None,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        topk_weights=torch.zeros(seq, 2),
        topk_ids=torch.zeros(seq, 2, dtype=torch.int32),
        experts_start=None,
        topk=2,
        activation="silu",
    )
    if num_chunks == 1:
        assert calls == [group] and group.calls == 0
    else:
        assert not calls and group.calls == num_chunks
