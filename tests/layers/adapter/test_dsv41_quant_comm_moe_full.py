# SPDX-License-Identifier: Apache-2.0
"""100% full-module line and branch coverage test suite for .

Covers all lines and branches in:
1. vllm_torchtpu.layers.adapter.quantization.deepseek_v41
   (deepseek_v41.py: 28 stmts, 6 branches)
2. vllm_torchtpu.distributed.tpu_communicator
   (tpu_communicator.py: 37 stmts, 12 branches)
3. vllm_torchtpu.layers.adapter.custom_ops.moe_runner
   (moe_runner.py: 51 stmts, 16 branches)
4. vllm_torchtpu.layers.adapter.moe_routing
   (moe_routing.py: 129 stmts, 46 branches)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.deepseek_v4.nvidia import model as v4_upstream

from vllm_torchtpu import envs
from vllm_torchtpu.distributed import tpu_communicator as comm
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.custom_ops import moe_runner as oot_moe_runner
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.layers.adapter.quantization import deepseek_v41 as dsv41_quant
from vllm_torchtpu.layers.adapter.quantization.deepseek_v41 import (
    DeepseekV41MoE,
    VllmDeepseekV41Fp8Config,
)

# ==============================================================================
# Helpers & Mocks
# ==============================================================================


class _FakeDistGroup:
    """Deterministic mock of a distributed collective group."""

    def __init__(self, world_size: int = 4, rank: int = 0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.all_gather_calls: list[tuple[torch.Tensor, int]] = []
        self.reduce_scatter_calls: list[tuple[torch.Tensor, int]] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.all_gather_calls.append((tensor.clone(), dim))
        repeats = [1] * tensor.ndim
        repeats[dim] = self.world_size
        return tensor.repeat(*repeats)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.reduce_scatter_calls.append((tensor.clone(), dim))
        assert tensor.shape[dim] % self.world_size == 0
        chunk_size = tensor.shape[dim] // self.world_size
        return tensor.narrow(dim, self.rank_in_group * chunk_size, chunk_size)


def _make_layer(
    *,
    topk: int = 4,
    is_sp: bool = True,
    custom_routing_fn=None,
    renormalize: bool = True,
    scoring_fn: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    use_grouped_topk: bool = False,
    num_expert_group: int = 1,
    topk_group: int = 1,
):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            experts_per_token=topk,
            is_sequence_parallel=is_sp,
            moe_parallel_config=SimpleNamespace(use_ep=False),
            ep_size=1,
            ep_rank=0,
        ),
        custom_routing_function=custom_routing_fn,
        renormalize=renormalize,
        scoring_func=scoring_fn,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        hash_indices_table=hash_indices_table,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        global_num_experts=8,
        expert_placement_strategy="linear",
        register_buffer=Mock(),
    )


# ==============================================================================
# 1. Tests for vllm_torchtpu.layers.adapter.quantization.deepseek_v41
# ==============================================================================


class TestDeepseekV41Quant:
    def test_config_get_name(self) -> None:
        """Test VllmDeepseekV41Fp8Config.get_name."""
        assert VllmDeepseekV41Fp8Config.get_name() == "deepseek_v41_fp8"

    def test_deepseek_v41_moe_init_guards(self) -> None:
        """Test DeepseekV41MoE.__init__ validation guards."""
        # 1. EPLB error
        vllm_config_eplb = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=True),
            kernel_config=SimpleNamespace(moe_backend="native"),
        )
        with pytest.raises(
            NotImplementedError, match="V4.1 TPU MoE does not support EPLB"
        ):
            DeepseekV41MoE(vllm_config_eplb)

        # 2. Mega MoE backend error
        vllm_config_mega = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=False),
            kernel_config=SimpleNamespace(
                moe_backend=next(iter(v4_upstream.MEGA_MOE_BACKENDS))
            ),
        )
        with pytest.raises(
            NotImplementedError, match="V4.1 TPU requires the native V4 MoE backend"
        ):
            DeepseekV41MoE(vllm_config_mega)

    def test_deepseek_v41_moe_init_device_hooks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test DeepseekV41MoE.__init__ with device_hooks and factory."""
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=False),
            kernel_config=SimpleNamespace(moe_backend="native"),
        )
        called_options: dict[str, object] = {}

        def mock_factory(**options):
            called_options.update(options)
            return Mock()

        monkeypatch.setattr(v4_upstream, "FusedMoEFactory", mock_factory)

        def mock_super_init(self, cfg, *args, **kwargs):
            # Inside the with device_hooks context!
            v4_upstream.FusedMoEFactory(routed_scaling=2.0)

        monkeypatch.setattr(dsv41_quant.DeepseekV4MoE, "__init__", mock_super_init)

        # Instantiate
        _ = DeepseekV41MoE(vllm_config, 1, 2, kw="test")
        assert called_options["apply_routed_scale_to_output"] is True
        assert called_options["routed_scaling"] == 2.0

    def test_deepseek_v41_moe_forward_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test DeepseekV41MoE.forward sequence parallel and input_ids all_gather
        branches."""
        moe = DeepseekV41MoE.__new__(DeepseekV41MoE)
        moe.gate = SimpleNamespace(tid2eid=torch.tensor([0, 1]))
        moe.experts = SimpleNamespace(do_naive_dispatch_combine=True)
        moe.use_sequence_parallel = True

        fake_ep = _FakeDistGroup(world_size=4, rank=0)
        monkeypatch.setattr(dsv41_quant, "get_ep_group", lambda: fake_ep)

        super_forward_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

        def mock_super_forward(self, hidden, input_ids=None):
            super_forward_calls.append((hidden, input_ids))
            return hidden * 2

        monkeypatch.setattr(dsv41_quant.DeepseekV4MoE, "forward", mock_super_forward)

        # 1. Happy path: all conditions True -> input_ids all_gathered
        hidden = torch.randn(2, 8)
        input_ids = torch.tensor([10, 20])
        out = moe.forward(hidden, input_ids)
        assert torch.equal(out, hidden * 2)
        assert len(fake_ep.all_gather_calls) == 1
        assert len(super_forward_calls) == 1
        assert super_forward_calls[-1][1].shape == (8,)  # 2 * 4

        # 2. use_sequence_parallel False
        moe.use_sequence_parallel = False
        fake_ep.all_gather_calls.clear()
        _ = moe.forward(hidden, input_ids)
        assert len(fake_ep.all_gather_calls) == 0
        assert super_forward_calls[-1][1] is input_ids

        # 3. input_ids is None
        moe.use_sequence_parallel = True
        fake_ep.all_gather_calls.clear()
        _ = moe.forward(hidden, None)
        assert len(fake_ep.all_gather_calls) == 0
        assert super_forward_calls[-1][1] is None

        # 4. gate.tid2eid is None
        moe.gate.tid2eid = None
        fake_ep.all_gather_calls.clear()
        _ = moe.forward(hidden, input_ids)
        assert len(fake_ep.all_gather_calls) == 0
        assert super_forward_calls[-1][1] is input_ids

        # 5. experts.do_naive_dispatch_combine False
        moe.gate.tid2eid = torch.tensor([0, 1])
        moe.experts.do_naive_dispatch_combine = False
        fake_ep.all_gather_calls.clear()
        _ = moe.forward(hidden, input_ids)
        assert len(fake_ep.all_gather_calls) == 0
        assert super_forward_calls[-1][1] is input_ids


# ==============================================================================
# 2. Tests for vllm_torchtpu.distributed.tpu_communicator
# ==============================================================================


class TestTpuCommunicator:
    def test_communicator_init_all2all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test TpuDeviceCommunicator.__init__ use_all2all True vs False."""
        manager_mock = Mock()
        monkeypatch.setattr(comm, "TpuNullAll2AllManager", manager_mock)

        def init_base(instance, *args, **kwargs):
            instance.use_all2all = True
            instance.cpu_group = SimpleNamespace()

        monkeypatch.setattr(DeviceCommunicatorBase, "__init__", init_base)
        tc_true = comm.TpuDeviceCommunicator()
        manager_mock.assert_called_once_with(tc_true.cpu_group)
        assert tc_true.all2all_manager is manager_mock.return_value

        def init_base_false(instance, *args, **kwargs):
            instance.use_all2all = False
            instance.cpu_group = SimpleNamespace()

        monkeypatch.setattr(DeviceCommunicatorBase, "__init__", init_base_false)
        tc_false = comm.TpuDeviceCommunicator()
        assert not hasattr(tc_false, "all2all_manager")

    def test_token_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _token_group for is_sequence_parallel True and False."""
        fake_ep = _FakeDistGroup(4)
        fake_dp = _FakeDistGroup(2)
        monkeypatch.setattr(comm, "get_ep_group", lambda: fake_ep)
        monkeypatch.setattr(comm, "get_dp_group", lambda: fake_dp)

        assert comm.TpuDeviceCommunicator._token_group(True) is fake_ep
        assert comm.TpuDeviceCommunicator._token_group(False) is fake_dp

    def test_dispatch_router_logits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test dispatch_router_logits with extra_tensors, world_size 1, same/distinct
        tensors."""
        tc = comm.TpuDeviceCommunicator.__new__(comm.TpuDeviceCommunicator)

        # 1. extra_tensors NotImplementedError
        with pytest.raises(
            NotImplementedError,
            match="extra_tensors is not supported by TpuDeviceCommunicator",
        ):
            tc.dispatch_router_logits(
                torch.empty(0), torch.empty(0), extra_tensors={"a": 1}
            )

        # 2. world_size == 1
        fake_group_1 = _FakeDistGroup(world_size=1)
        monkeypatch.setattr(comm, "get_dp_group", lambda: fake_group_1)
        h = torch.randn(4, 16)
        r = torch.randn(4, 32)
        h_out, r_out = tc.dispatch_router_logits(h, r, is_sequence_parallel=False)
        assert h_out is h
        assert r_out is r

        # 3. world_size > 1 and router_logits is hidden_states
        fake_group_4 = _FakeDistGroup(world_size=4)
        monkeypatch.setattr(comm, "get_ep_group", lambda: fake_group_4)
        h_out, r_out = tc.dispatch_router_logits(h, h, is_sequence_parallel=True)
        assert h_out.shape == (16, 16)
        assert r_out is h_out

        # 4. world_size > 1 and router_logits is distinct
        h_out, r_out = tc.dispatch_router_logits(h, r, is_sequence_parallel=True)
        assert h_out.shape == (16, 16)
        assert r_out.shape == (16, 32)
        assert r_out is not h_out

    def test_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test dispatch with extra_tensors, world_size 1, and world_size > 1."""
        tc = comm.TpuDeviceCommunicator.__new__(comm.TpuDeviceCommunicator)

        # 1. extra_tensors NotImplementedError
        with pytest.raises(
            NotImplementedError,
            match="extra_tensors is not supported by TpuDeviceCommunicator",
        ):
            tc.dispatch(
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                extra_tensors={"test": 1},
            )

        # 2. world_size == 1
        fake_group_1 = _FakeDistGroup(world_size=1)
        monkeypatch.setattr(comm, "get_dp_group", lambda: fake_group_1)
        h = torch.randn(4, 16)
        w = torch.randn(4, 8)
        i = torch.randint(0, 8, (4, 8))
        h_out, w_out, i_out = tc.dispatch(h, w, i, is_sequence_parallel=False)
        assert h_out is h
        assert w_out is w
        assert i_out is i

        # 3. world_size > 1
        fake_group_2 = _FakeDistGroup(world_size=2)
        monkeypatch.setattr(comm, "get_ep_group", lambda: fake_group_2)
        h_out, w_out, i_out = tc.dispatch(h, w, i, is_sequence_parallel=True)
        assert h_out.shape == (8, 16)
        assert w_out.shape == (8, 8)
        assert i_out.shape == (8, 8)

    def test_combine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test combine with world_size 1 and world_size > 1."""
        tc = comm.TpuDeviceCommunicator.__new__(comm.TpuDeviceCommunicator)

        # 1. world_size == 1
        fake_group_1 = _FakeDistGroup(world_size=1)
        monkeypatch.setattr(comm, "get_dp_group", lambda: fake_group_1)
        h = torch.randn(4, 16)
        out1 = tc.combine(h, is_sequence_parallel=False)
        assert out1 is h

        # 2. world_size > 1 (reduce_scatter)
        fake_group_2 = _FakeDistGroup(world_size=2, rank=1)
        monkeypatch.setattr(comm, "get_ep_group", lambda: fake_group_2)
        h = torch.randn(8, 16)
        out2 = tc.combine(h, is_sequence_parallel=True)
        assert out2.shape == (4, 16)


# ==============================================================================
# 3. Tests for vllm_torchtpu.layers.adapter.custom_ops.moe_runner
# ==============================================================================


class _MockMoERunner(VllmMoERunner):
    """Test helper subclass of VllmMoERunner bypassing full upstream __init__."""

    def __init__(
        self,
        *,
        do_naive: bool = True,
        pcp_size: int = 1,
        is_sp: bool = True,
        is_monolithic: bool = True,
        custom_routing=None,
        fused_ep_owns_pcp: bool = False,
        shared_experts=None,
        moe_config=None,
        quant_method=None,
    ):
        nn.Module.__init__(self)
        self._do_naive = do_naive
        self.moe_config = moe_config or SimpleNamespace(
            pcp_size=pcp_size,
            is_sequence_parallel=is_sp,
        )
        self.routed_experts = SimpleNamespace(
            quant_method=SimpleNamespace(is_monolithic=is_monolithic),
            custom_routing_function=custom_routing,
            _ensure_moe_quant_config_init=Mock(),
        )
        self._fused_ep_owns_pcp = fused_ep_owns_pcp
        self._mock_shared_experts = shared_experts
        self._mock_quant_method = quant_method or SimpleNamespace()

    @property
    def shared_experts(self):
        return self._mock_shared_experts

    @property
    def _quant_method(self):
        return self._mock_quant_method

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return self._do_naive

    def _fused_ep_owns_pcp_collectives(self) -> bool:
        return self._fused_ep_owns_pcp


class TestMoeRunner:
    def test_init_and_take_over_quant_config_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test __init__ and _take_over_moe_quant_config_init monolithic vs
        non-monolithic."""
        super_init = Mock()
        monkeypatch.setattr(MoERunner, "__init__", super_init)

        runner = VllmMoERunner.__new__(VllmMoERunner)
        upstream_init = Mock()
        runner.routed_experts = SimpleNamespace(
            _ensure_moe_quant_config_init=upstream_init,
            quant_method=SimpleNamespace(is_monolithic=True),
        )

        runner._take_over_moe_quant_config_init()
        # When monolithic, upstream_init is skipped
        runner.routed_experts._ensure_moe_quant_config_init()
        upstream_init.assert_not_called()

        # When non-monolithic, upstream_init is called
        runner.routed_experts.quant_method.is_monolithic = False
        runner.routed_experts._ensure_moe_quant_config_init()
        upstream_init.assert_called_once()

        # Full __init__ call
        runner.__init__("arg1", kw=2)
        super_init.assert_called_once_with("arg1", kw=2)

    def test_select_forward(self) -> None:
        """Test _select_forward shared_experts None vs not None."""
        runner_none = _MockMoERunner(shared_experts=None)
        assert runner_none._select_forward() is oot_moe_runner._moe_runner._moe_forward

        runner_shared = _MockMoERunner(shared_experts=SimpleNamespace())
        assert (
            runner_shared._select_forward()
            is oot_moe_runner._moe_runner._moe_forward_shared
        )

    def test_fused_output_is_reduced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _fused_output_is_reduced conditions and fallbacks."""
        runner = _MockMoERunner()

        # 1. USE_MOE_FUSED_EP_KERNEL is False
        monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
        monkeypatch.setattr(
            oot_moe_runner, "_upstream_fused_output_is_reduced", lambda s: "upstream"
        )
        assert runner._fused_output_is_reduced == "upstream"

        # 2. skip_final_all_reduce is True
        monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
        runner.moe_config.skip_final_all_reduce = True
        assert runner._fused_output_is_reduced == "upstream"
        runner.moe_config.skip_final_all_reduce = False

        # 3. fused_moe_ep_supported is True
        monkeypatch.setattr(oot_moe_runner, "fused_moe_ep_supported", lambda m: True)
        monkeypatch.setattr(
            oot_moe_runner, "adaptive_fused_moe_supported", lambda m: False
        )
        assert runner._fused_output_is_reduced is True

        # 4. adaptive_fused_moe_supported is True
        monkeypatch.setattr(oot_moe_runner, "fused_moe_ep_supported", lambda m: False)
        monkeypatch.setattr(
            oot_moe_runner, "adaptive_fused_moe_supported", lambda m: True
        )
        assert runner._fused_output_is_reduced is True

        # 5. Neither supported -> upstream fallback
        monkeypatch.setattr(oot_moe_runner, "fused_moe_ep_supported", lambda m: False)
        monkeypatch.setattr(
            oot_moe_runner, "adaptive_fused_moe_supported", lambda m: False
        )
        assert runner._fused_output_is_reduced == "upstream"

    def test_fused_ep_owns_pcp_collectives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _fused_ep_owns_pcp_collectives with pcp_size <= 1 and > 1."""
        runner = VllmMoERunner.__new__(VllmMoERunner)
        runner.moe_config = SimpleNamespace(pcp_size=1)
        runner._mock_quant_method = SimpleNamespace()
        runner.routed_experts = SimpleNamespace(quant_method=runner._mock_quant_method)

        # pcp_size <= 1
        assert runner._fused_ep_owns_pcp_collectives() is False

        # pcp_size > 1
        runner.moe_config.pcp_size = 2
        monkeypatch.setattr(oot_moe_runner, "fused_moe_ep_supported", lambda m: True)
        assert runner._fused_ep_owns_pcp_collectives() is True

        monkeypatch.setattr(oot_moe_runner, "fused_moe_ep_supported", lambda m: False)
        monkeypatch.setattr(
            oot_moe_runner, "adaptive_fused_moe_supported", lambda m: True
        )
        assert runner._fused_ep_owns_pcp_collectives() is True

        monkeypatch.setattr(
            oot_moe_runner, "adaptive_fused_moe_supported", lambda m: False
        )
        assert runner._fused_ep_owns_pcp_collectives() is False

    def test_maybe_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _maybe_dispatch across all conditions and fallbacks."""
        hidden = torch.randn(4, 16)
        logits = torch.randn(4, 32)

        # 1. _fused_ep_owns_pcp_collectives True
        r1 = _MockMoERunner(fused_ep_owns_pcp=True)
        h_out, r_out = r1._maybe_dispatch(hidden, logits)
        assert h_out is hidden
        assert r_out is logits

        # 2. All 6 naive conditions True with SP=True and world_size > 1
        r2 = _MockMoERunner(
            fused_ep_owns_pcp=False,
            do_naive=True,
            pcp_size=1,
            is_sp=True,
            is_monolithic=True,
        )
        fake_ep = _FakeDistGroup(world_size=2)
        monkeypatch.setattr(oot_moe_runner, "get_ep_group", lambda: fake_ep)
        monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)

        h_out, r_out = r2._maybe_dispatch(hidden, logits)
        assert h_out.shape == (8, 16)
        assert r_out is logits

        # 3. SP=False (uses DP group) with world_size == 1
        r3 = _MockMoERunner(
            fused_ep_owns_pcp=False,
            do_naive=True,
            pcp_size=1,
            is_sp=False,
            is_monolithic=True,
        )
        fake_dp = _FakeDistGroup(world_size=1)
        monkeypatch.setattr(oot_moe_runner, "get_dp_group", lambda: fake_dp)
        h_out, r_out = r3._maybe_dispatch(hidden, logits)
        assert h_out is hidden
        assert r_out is logits

        # 4. Fallback when router_logits is hidden_states
        super_dispatch = Mock(return_value=("super_h", "super_r"))
        monkeypatch.setattr(MoERunner, "_maybe_dispatch", super_dispatch)
        assert r2._maybe_dispatch(hidden, hidden) == ("super_h", "super_r")

    def test_maybe_combine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _maybe_combine with fused_ep_owns_pcp True (with/without shared) and
        False."""
        shared = torch.randn(4, 16)
        hidden = torch.randn(4, 16)

        # 1. fused_ep_owns_pcp True and shared_experts is not None
        r1 = _MockMoERunner(fused_ep_owns_pcp=True, shared_experts=SimpleNamespace())
        out1 = r1._maybe_combine(shared, hidden)
        assert out1 == (shared, hidden)

        # 2. fused_ep_owns_pcp True and shared_experts is None
        r2 = _MockMoERunner(fused_ep_owns_pcp=True, shared_experts=None)
        out2 = r2._maybe_combine(shared, hidden)
        assert out2 is hidden

        # 3. fused_ep_owns_pcp False -> calls super()._maybe_combine
        r3 = _MockMoERunner(fused_ep_owns_pcp=False)
        super_combine = Mock(return_value="combined_super")
        monkeypatch.setattr(MoERunner, "_maybe_combine", super_combine)
        assert r3._maybe_combine(shared, hidden) == "combined_super"


# ==============================================================================
# 4. Tests for vllm_torchtpu.layers.adapter.moe_routing
# ==============================================================================


class TestMoeRouting:
    def test_topk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _topk with rowmax and default torch.topk."""
        scores = torch.randn(4, 16)

        # 1. rowmax branch
        monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "rowmax")
        mock_rowmax = Mock(return_value=("rowmax_vals", "rowmax_indices"))
        monkeypatch.setattr(moe_routing, "rowmax_select", mock_rowmax)
        val, idx = moe_routing._topk(scores, 4)
        mock_rowmax.assert_called_once_with(scores, 4)
        assert val == "rowmax_vals"

        # 2. default branch
        monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "standard")
        val2, idx2 = moe_routing._topk(scores, 4)
        expected_val, expected_idx = torch.topk(scores, k=4, dim=-1)
        assert torch.equal(val2, expected_val)
        assert torch.equal(idx2, expected_idx)

    def test_simulation_strategy_import_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test routing simulation strategy name validation (ValueError and valid
        strategy)."""
        import importlib

        import vllm.envs as vllm_envs

        # 1. Invalid strategy raises ValueError
        monkeypatch.setattr(
            vllm_envs,
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
            "invalid_strategy_name_xyz",
        )
        with pytest.raises(ValueError, match="is not a known routing strategy"):
            importlib.reload(moe_routing)

        # 2. Valid strategy sets _SIMULATION_STRATEGY and logs warning
        monkeypatch.setattr(
            vllm_envs, "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "uniform_random"
        )
        importlib.reload(moe_routing)
        assert moe_routing._SIMULATION_STRATEGY is not None

        # 3. Restore to None
        monkeypatch.setattr(vllm_envs, "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "")
        importlib.reload(moe_routing)
        assert moe_routing._SIMULATION_STRATEGY is None

    def test_uniform_random_routing(self) -> None:
        """Test _uniform_random_routing."""
        hidden = torch.randn(4, 16, dtype=torch.bfloat16)
        logits = torch.randn(4, 32, dtype=torch.float32)
        weights, ids = moe_routing._uniform_random_routing(hidden, logits, topk=4)
        assert weights.shape == (4, 4)
        assert weights.dtype == torch.bfloat16
        assert ids.shape == (4, 4)
        assert ids.dtype == torch.int32
        assert torch.all(weights == 1.0)
        assert torch.all((ids >= 0) & (ids < 32))

    def test_maybe_simulate_routing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test maybe_simulate_routing across None, uniform_random, and custom
        strategy."""
        hidden = torch.randn(4, 16, dtype=torch.bfloat16)
        logits = torch.randn(4, 32, dtype=torch.float32)

        # 1. _SIMULATION_STRATEGY is None
        monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)
        assert moe_routing.maybe_simulate_routing(hidden, logits, 4) is None

        # 2. uniform_random
        mock_strategy = SimpleNamespace()
        monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", mock_strategy)
        monkeypatch.setattr(moe_routing, "_strategy_name", "uniform_random")
        res_uniform = moe_routing.maybe_simulate_routing(hidden, logits, 4)
        assert res_uniform is not None
        assert res_uniform[0].shape == (4, 4)

        # 3. custom strategy route_tokens
        custom_weights = torch.ones(4, 4, dtype=torch.float32)
        custom_ids = torch.zeros(4, 4, dtype=torch.int64)
        mock_strategy.route_tokens = Mock(return_value=(custom_weights, custom_ids))
        monkeypatch.setattr(moe_routing, "_strategy_name", "custom_strat")
        w, i = moe_routing.maybe_simulate_routing(hidden, logits, 4)
        assert w.dtype == torch.bfloat16
        assert i.dtype == torch.int32
        mock_strategy.route_tokens.assert_called_once()

    def test_gather_partitioned_topk(self) -> None:
        """Test _gather_partitioned_topk bit-packed path vs separate all-gather path."""
        group = _FakeDistGroup(world_size=2)

        # 1. Bit-packed path: bfloat16, even topk, int32 ids
        w_bf16 = torch.ones(4, 4, dtype=torch.bfloat16)
        i_s32 = torch.full((4, 4), 5, dtype=torch.int32)
        gw1, gi1 = moe_routing._gather_partitioned_topk(group, w_bf16, i_s32, 4)
        assert gw1.shape == (8, 4)
        assert gw1.dtype == torch.bfloat16
        assert gi1.shape == (8, 4)
        assert gi1.dtype == torch.int32
        assert torch.all(gw1 == 1.0)
        assert torch.all(gi1 == 5)

        # 2. Bit-packed path: float16
        w_fp16 = torch.ones(4, 4, dtype=torch.float16)
        gw2, gi2 = moe_routing._gather_partitioned_topk(group, w_fp16, i_s32, 4)
        assert gw2.dtype == torch.float16
        assert gi2.dtype == torch.int32

        # 3. Fallback separate path: float32 weights
        w_f32 = torch.ones(4, 4, dtype=torch.float32)
        gw3, gi3 = moe_routing._gather_partitioned_topk(group, w_f32, i_s32, 4)
        assert gw3.shape == (8, 4)
        assert gw3.dtype == torch.float32
        assert gi3.shape == (8, 4)

        # 4. Fallback separate path: odd topk
        w_odd = torch.ones(4, 3, dtype=torch.bfloat16)
        i_odd = torch.full((4, 3), 2, dtype=torch.int32)
        gw4, gi4 = moe_routing._gather_partitioned_topk(group, w_odd, i_odd, 3)
        assert gw4.shape == (8, 3)

        # 5. Fallback separate path: int64 ids
        i_s64 = torch.full((4, 4), 2, dtype=torch.int64)
        gw5, gi5 = moe_routing._gather_partitioned_topk(group, w_bf16, i_s64, 4)
        assert gw5.shape == (8, 4)

    def test_apply_scoring_fn(self) -> None:
        """Test _apply_scoring_fn across softmax, sigmoid, sqrtsoftplus and
        unsupported."""
        logits = torch.randn(4, 16)

        # 1. softmax
        s_sm = moe_routing._apply_scoring_fn("softmax", logits)
        assert torch.allclose(s_sm, logits.float().softmax(dim=-1))

        # 2. sigmoid
        s_sig = moe_routing._apply_scoring_fn("sigmoid", logits)
        assert torch.allclose(s_sig, logits.float().sigmoid())

        # 3. sqrtsoftplus
        s_sqrt = moe_routing._apply_scoring_fn("sqrtsoftplus", logits)
        import torch.nn.functional as F

        assert torch.allclose(s_sqrt, torch.sqrt(F.softplus(logits.float())))

        # 4. unsupported
        with pytest.raises(NotImplementedError, match="does not support invalid_fn"):
            moe_routing._apply_scoring_fn("invalid_fn", logits)

    def test_hash_moe_select(self) -> None:
        """Test _hash_moe_select renormalize True/False and scaling factor."""
        scores = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32
        )
        table = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)  # 2 token IDs
        input_ids = torch.tensor([0, 1])

        # 1. renormalize True, scale 1.0
        w1, i1 = moe_routing._hash_moe_select(
            scores, table, input_ids, renormalize=True, routed_scaling_factor=1.0
        )
        assert torch.allclose(w1.sum(dim=-1), torch.tensor([1.0, 1.0]))
        assert torch.equal(i1, torch.tensor([[0, 2], [1, 3]]))

        # 2. renormalize False, scale 2.5
        w2, i2 = moe_routing._hash_moe_select(
            scores, table, input_ids, renormalize=False, routed_scaling_factor=2.5
        )
        expected = torch.tensor([[0.1, 0.3], [0.6, 0.8]]) * 2.5
        assert torch.allclose(w2, expected)

    def test_select_experts_branches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test select_experts simulation, hash, grouped_topk, bias, and scaling."""
        hidden = torch.randn(4, 16, dtype=torch.bfloat16)
        logits = torch.randn(4, 32, dtype=torch.float32)

        # 1. Simulation override
        monkeypatch.setattr(
            moe_routing,
            "maybe_simulate_routing",
            lambda h, r, k: ("sim_w", "sim_i"),
        )
        assert moe_routing.select_experts(
            hidden, logits, topk=4, renormalize=True, scoring_fn="softmax"
        ) == ("sim_w", "sim_i")
        monkeypatch.setattr(moe_routing, "maybe_simulate_routing", lambda h, r, k: None)

        # 2. Hash routing: layer.hash_indices_table and input_ids present
        layer_hash = _make_layer(
            hash_indices_table=torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
            routed_scaling_factor=1.5,
        )
        input_ids = torch.tensor([0, 1, 2, 3])
        w_hash, i_hash = moe_routing.select_experts(
            hidden,
            logits,
            topk=2,
            renormalize=True,
            scoring_fn="softmax",
            layer=layer_hash,
            input_ids=input_ids,
        )
        assert w_hash.dtype == torch.bfloat16
        assert i_hash.dtype == torch.int32
        assert w_hash.shape == (4, 2)

        # 3. Grouped top-k routing: use_grouped = True
        layer_grouped = _make_layer(
            use_grouped_topk=True,
            num_expert_group=4,
            topk_group=2,
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
        )
        mock_grouped_fn = Mock(
            return_value=(
                torch.ones(4, 4, dtype=torch.float32),
                torch.zeros(4, 4, dtype=torch.int64),
            )
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk",
            mock_grouped_fn,
        )
        w_grp, i_grp = moe_routing.select_experts(
            hidden,
            logits,
            topk=4,
            renormalize=True,
            scoring_fn="sigmoid",
            layer=layer_grouped,
        )
        assert w_grp.dtype == torch.bfloat16
        assert i_grp.dtype == torch.int32
        mock_grouped_fn.assert_called_once()

        # 4. Standard topk with bias, renormalize=False, scaling != 1.0, layer is None
        w_std1, i_std1 = moe_routing.select_experts(
            hidden,
            logits,
            topk=4,
            renormalize=False,
            scoring_fn="sqrtsoftplus",
            layer=None,
        )
        assert w_std1.dtype == torch.bfloat16
        assert i_std1.dtype == torch.int32
        assert w_std1.shape == (4, 4)

        # 5. Standard topk with bias present on layer and scaling != 1.0
        bias = torch.randn(32)
        layer_bias = _make_layer(
            e_score_correction_bias=bias,
            routed_scaling_factor=2.0,
        )
        w_std2, i_std2 = moe_routing.select_experts(
            hidden,
            logits,
            topk=4,
            renormalize=True,
            scoring_fn="softmax",
            layer=layer_bias,
        )
        assert w_std2.shape == (4, 4)
        assert torch.allclose(
            w_std2.sum(dim=-1), torch.full((4,), 2.0, dtype=torch.bfloat16)
        )

    def test_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test route simulation, custom_routing_fn, partitioned SP/DP, and
        unpartitioned."""
        hidden = torch.randn(8, 16, dtype=torch.bfloat16)
        logits_local = torch.randn(4, 32, dtype=torch.float32)
        logits_full = torch.randn(8, 32, dtype=torch.float32)

        layer = _make_layer(topk=4, is_sp=True)

        # 1. Simulation
        monkeypatch.setattr(
            moe_routing,
            "maybe_simulate_routing",
            lambda h, r, k: ("sim_w", "sim_i"),
        )
        assert moe_routing.route(layer, hidden, logits_full) == (
            "sim_w",
            "sim_i",
        )
        monkeypatch.setattr(moe_routing, "maybe_simulate_routing", lambda h, r, k: None)

        # 2. custom_routing_function
        layer.custom_routing_function = Mock(return_value=("custom_w", "custom_i"))
        assert moe_routing.route(layer, hidden, logits_full) == (
            "custom_w",
            "custom_i",
        )
        layer.custom_routing_function = None

        # 3. Partitioned: router_logits.shape[0] < hidden_states.shape[0]
        # 3a. SP=True, input_ids matches numel
        fake_ep = _FakeDistGroup(world_size=2, rank=1)
        monkeypatch.setattr(
            "vllm.distributed.parallel_state.get_ep_group", lambda: fake_ep
        )
        input_ids = torch.arange(8)
        w_part, i_part = moe_routing.route(
            layer, hidden, logits_local, input_ids=input_ids
        )
        assert w_part.shape == (8, 4)
        assert i_part.shape == (8, 4)

        # 3b. SP=False (DP group), input_ids is None
        layer.moe_config.is_sequence_parallel = False
        fake_dp = _FakeDistGroup(world_size=2, rank=0)
        monkeypatch.setattr(
            "vllm.distributed.parallel_state.get_dp_group", lambda: fake_dp
        )
        w_part_dp, i_part_dp = moe_routing.route(
            layer, hidden, logits_local, input_ids=None
        )
        assert w_part_dp.shape == (8, 4)

        # 4. Unpartitioned: router_logits.shape[0] == hidden_states.shape[0]
        w_unpart, i_unpart = moe_routing.route(
            layer, hidden, logits_full, input_ids=input_ids
        )
        assert w_unpart.shape == (8, 4)

    def test_ep_placement_and_experts_start(self) -> None:
        """Test get_experts_start, register_experts_start_buffer, and
        validate_linear_ep_placement."""
        layer = _make_layer()
        layer.moe_config.moe_parallel_config = SimpleNamespace(use_ep=True)
        layer.moe_config.ep_size = 4
        layer.moe_config.ep_rank = 2
        layer.global_num_experts = 10
        layer.expert_placement_strategy = "linear"

        # 1. get_experts_start EP=True
        # base = 10 // 4 = 2, remainder = 10 % 4 = 2
        # ep_rank=2 -> 2 * 2 + min(2, 2) = 6
        assert moe_routing.get_experts_start(layer) == 6

        # 2. get_experts_start EP=False
        layer.moe_config.moe_parallel_config.use_ep = False
        assert moe_routing.get_experts_start(layer) is None
        layer.moe_config.moe_parallel_config.use_ep = True

        # 3. register_experts_start_buffer when not None
        moe_routing.register_experts_start_buffer(layer, device=torch.device("cpu"))
        layer.register_buffer.assert_called_once()
        name, buf = layer.register_buffer.call_args[0][:2]
        assert name == "_experts_start"
        assert isinstance(buf, torch.Tensor)
        assert buf.item() == 6

        # 4. register_experts_start_buffer when None
        layer.moe_config.moe_parallel_config.use_ep = False
        layer.register_buffer.reset_mock()
        moe_routing.register_experts_start_buffer(layer, device=torch.device("cpu"))
        layer.register_buffer.assert_called_once_with(
            "_experts_start", None, persistent=False
        )

        # 5. validate_linear_ep_placement
        moe_routing.validate_linear_ep_placement(layer)  # linear succeeds

        layer.expert_placement_strategy = "interleaved"
        with pytest.raises(NotImplementedError, match="requires linear EP placement"):
            moe_routing.validate_linear_ep_placement(layer)
