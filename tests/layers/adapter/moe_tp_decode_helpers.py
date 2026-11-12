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
"""Shared CPU test doubles and harness for moe_tp_decode arming tests."""

import math
from types import SimpleNamespace

import torch
import vllm.distributed.parallel_state as parallel_state
from jax.experimental.pallas import tpu as pltpu
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner import moe_runner as moe_runner_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import moe_routing, moe_tp_decode, token_padding
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

# One rank's slice: the smallest hidden size the kernel admits at 16 ranks.
H, E, INTER, FS, TOPK = 2048, 16, 128, 128, 8
ROUTED_SCALE = 2.5
FP8 = torch.float8_e4m3fn


class Fp8Linear(torch.nn.Module):
    """A K-major per-channel FP8 linear as VllmFp8LinearMethodTPU leaves it."""

    def __init__(self, k: int, n: int, *, reduce_results: bool | None = None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (torch.randn(k, n) / math.sqrt(k)).to(FP8), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(torch.rand(n) + 0.5, requires_grad=False)
        self.bias = None
        if reduce_results is not None:
            self.reduce_results = reduce_results
        setattr(self, WEIGHT_FLIPPED_ATTR, True)
        self.quant_method = fp8_mod.VllmFp8LinearMethodTPU.__new__(
            fp8_mod.VllmFp8LinearMethodTPU
        )


class SharedMLP(torch.nn.Module):
    """DeepseekV2MLP's structure (reduce_results=False); forward counts calls."""

    def __init__(self, hidden_size: int = H, shared_inter: int = FS):
        super().__init__()
        self.gate_up_proj = Fp8Linear(hidden_size, 2 * shared_inter)
        self.down_proj = Fp8Linear(shared_inter, hidden_size, reduce_results=False)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x * 0.5


class Gate(torch.nn.Module):
    def __init__(self, num_experts: int = E, hidden_size: int = H):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (torch.randn(num_experts, hidden_size) / math.sqrt(hidden_size)).to(
                torch.bfloat16
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.linear(x, self.weight).float(), None


class StubRoutedExperts(torch.nn.Module):
    forward_monolithic = RoutedExperts.forward_monolithic

    def __init__(self, moe_config, quant_method, scoring_func: str):
        super().__init__()
        self.moe_config = moe_config
        self.quant_method = quant_method
        self.global_num_experts = E
        self.scoring_func = scoring_func
        self.renormalize = True
        self.use_grouped_topk = True
        self.num_expert_group = 1
        self.topk_group = 1
        self.custom_routing_function = None
        self.routed_scaling_factor = 1.0
        self.expert_placement_strategy = "linear"

        def param(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.w13_weight = param(torch.randn(E, H, 2 * INTER).to(FP8))
        self.w2_weight = param(torch.randn(E, INTER, H).to(FP8))
        self.w13_weight_scale_inv = param(torch.rand(E, 1, 1, 2 * INTER))
        self.w2_weight_scale_inv = param(torch.rand(E, 1, 1, H))
        self.e_score_correction_bias = param(torch.rand(E) * 0.1)
        moe_routing.register_experts_start_buffer(self, device=torch.device("cpu"))

    def _ensure_moe_quant_config_init(self):
        pass


def make_stub_moe_config(*, use_ep: bool = False, skip_final_all_reduce: bool = False):
    par = SimpleNamespace(
        use_ep=use_ep,
        ep_size=16 if use_ep else 1,
        ep_rank=0,
        tp_size=1 if use_ep else 16,
        dp_size=1,
        pcp_size=1,
        enable_eplb=False,
        all2all_backend="allgather_reducescatter",
        use_fi_nvl_two_sided_kernels=False,
        use_all2all_kernels=False,
    )
    return SimpleNamespace(
        moe_parallel_config=par,
        experts_per_token=TOPK,
        is_sequence_parallel=False,
        skip_final_all_reduce=skip_final_all_reduce,
        tp_size=par.tp_size,
        ep_size=par.ep_size,
        ep_rank=0,
        dp_size=1,
        pcp_size=1,
        sp_size=1,
        hidden_dim=H,
        hidden_dim_unpadded=H,
    )


class MoeTpDecodeHarness:
    def __init__(self, *, op_scale: float = 3.0):
        self.layers: dict[str, MoERunner] = {}
        self.all_reduces = 0
        self.op_calls = []
        self.recorded_statics = []
        self.selftest_calls = 0
        self.selftest_failures = []
        self.logs: list[str] = []
        self.op_scale = op_scale

    def register(self, runner: MoERunner) -> MoERunner:
        self.layers[runner.layer_name] = runner
        return runner

    def runner(
        self, name: str, *, scoring_func: str = "sigmoid", shared: bool = True, **config
    ):
        mcfg = make_stub_moe_config(**config)
        qm = fp8_mod.VllmFp8MoEMethodTPU.__new__(fp8_mod.VllmFp8MoEMethodTPU)
        FusedMoEMethodBase.__init__(qm, mcfg)
        qm._tpu_activation_str = "silu"
        runner = MoERunner(
            layer_name=name,
            moe_config=mcfg,
            router=SimpleNamespace(prepare_monolithic_logits=lambda logits: logits),
            routed_experts=StubRoutedExperts(mcfg, qm, scoring_func),
            gate=Gate(),
            shared_experts=SharedMLP() if shared else None,
            routed_scaling_factor=ROUTED_SCALE,
        )
        return self.register(runner)

    def model(self, *specs, prefix: str = "model.layers"):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList(
            self.runner(f"{prefix}.{i}.mlp.experts", **spec)
            for i, spec in enumerate(specs)
        )
        return model

    def op(self, x: torch.Tensor, *operands):
        self.op_calls.append((x, *operands))
        return (x.float() * self.op_scale).to(x.dtype)


class _LogSink:
    def __init__(self, sink: list[str]):
        self.info = self.warning = lambda fmt, *args: sink.append(fmt % args)


def install_moe_tp_decode_harness(
    monkeypatch,
    *,
    n_ranks: int = 16,
    valid_tokens: int = 11,
    padded_tokens: int = 16,
    op_scale: float = 3.0,
) -> MoeTpDecodeHarness:
    torch.manual_seed(0)
    h = MoeTpDecodeHarness(op_scale=op_scale)
    # Set the attribute, not the environment: a test that monkeypatched this
    # attribute earlier in the session leaves it set on the module, where it
    # shadows envs.__getattr__ and with it any environment value.
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", True)
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    monkeypatch.setattr(moe_runner_mod, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        moe_runner_mod, "register_layer_for_moe_forward_op", lambda config, layer: None
    )
    monkeypatch.setattr(moe_runner_mod, "get_layer_from_name", h.layers.__getitem__)
    monkeypatch.setattr(
        moe_runner_mod,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=None, all_moe_layers=None),
    )
    monkeypatch.setattr(moe_runner_mod, "is_forward_context_available", lambda: False)

    def all_reduce(t):
        h.all_reduces += 1
        return t

    monkeypatch.setattr(moe_runner_mod, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        parallel_state, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(
        fp8_mod, "fused_moe_gmm", lambda hidden_states, **kw: hidden_states * 0.25
    )
    padding = token_padding.TokenPaddingState.create(64, torch.device("cpu"))
    padding.update(valid_tokens, padded_tokens)
    monkeypatch.setattr(token_padding, "_padding_state", padding)

    monkeypatch.setattr(
        moe_tp_decode,
        "_topology",
        lambda: moe_tp_decode._Topology(
            mesh=SimpleNamespace(size=n_ranks), mesh_index=0
        ),
    )
    monkeypatch.setattr(moe_tp_decode, "_gather", lambda obj: [obj])

    def get_op(mesh, statics):
        h.recorded_statics.append(statics)
        return h.op

    monkeypatch.setattr(moe_tp_decode, "_get_op", get_op)

    def selftest(topo, buckets, device):
        h.selftest_calls += 1
        return h.selftest_failures

    monkeypatch.setattr(moe_tp_decode, "_selftest", selftest)
    monkeypatch.setattr(moe_tp_decode, "_RANK_BUFFER", None)
    monkeypatch.setattr(moe_tp_decode, "_SELFTEST_DONE", False)
    monkeypatch.setattr(moe_tp_decode, "logger", _LogSink(h.logs))
    return h
