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

from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.custom_ops import moe_runner as runner_mod
from vllm_torchtpu.layers.adapter.fused_moe_ep import FUSED_MOE_EP_OP_ATTR
from vllm_torchtpu.layers.adapter.quantization import deepseek_v41 as mod
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP


def _moe(*, sp, hash_table, naive):
    moe = object.__new__(mod.DeepseekV41MoE)
    object.__setattr__(moe, "use_sequence_parallel", sp)
    object.__setattr__(moe, "gate", SimpleNamespace(tid2eid=hash_table))
    object.__setattr__(moe, "experts", SimpleNamespace(do_naive_dispatch_combine=naive))
    return moe


@pytest.mark.parametrize(
    "sp,hash_table,naive,gathered",
    [
        (True, torch.zeros(4, 2), True, True),
        (False, torch.zeros(4, 2), True, False),
        (True, None, True, False),
        # An armed fused-EP layer routes this rank's own rows.
        (True, torch.zeros(4, 2), False, False),
    ],
)
def test_hash_ids_follow_sp_dispatch(monkeypatch, sp, hash_table, naive, gathered):
    ep = Mock()
    ep.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    monkeypatch.setattr(mod, "get_ep_group", lambda: ep)
    seen = {}

    def upstream_forward(self, hidden_states, input_ids=None, mega_gate_metadata=None):
        seen["ids"] = input_ids
        return hidden_states

    monkeypatch.setattr(mod.DeepseekV4MoE, "forward", upstream_forward)
    ids = torch.tensor([1, 2])
    hidden = torch.ones(2, 3)
    assert _moe(sp=sp, hash_table=hash_table, naive=naive).forward(hidden, ids) is (
        hidden
    )
    if gathered:
        ep.all_gather.assert_called_once_with(ids, dim=0)
        torch.testing.assert_close(seen["ids"], torch.tensor([1, 2, 11, 12]))
    else:
        ep.all_gather.assert_not_called()
        assert seen["ids"] is ids


def _prequantized_moe(monkeypatch, *, internal_gate, method_type, padded=False):
    """Use the production model/runner; replace only the TPU expert kernel."""

    class ReferenceMethod(method_type):
        supports_internal_mk = True
        skip_forward_padding = False
        moe_kernel = None

        def __init__(self):
            setattr(self, FUSED_MOE_EP_OP_ATTR, "cpu-reference")

        def apply_monolithic(
            self,
            layer,
            hidden,
            logits,
            input_ids,
            *,
            hidden_states_q,
            hidden_states_scale,
        ):
            assert hidden.dtype == torch.bfloat16
            assert logits.shape == (hidden.shape[0], 2)
            assert hidden_states_q.shape == hidden.shape
            assert hidden_states_q.dtype == torch.float8_e4m3fn
            # Both logits and ids affect the result, as do both FP8 operands.
            expert = logits.argmax(dim=-1)
            multiplier = (expert + 1 + input_ids).float().unsqueeze(-1)
            dequantized = hidden_states_q.float() * hidden_states_scale
            return (dequantized * multiplier).to(torch.bfloat16)

        _forward_monolithic_tpu = apply_monolithic

    class Gate:
        tid2eid = torch.zeros(8, 1, dtype=torch.int32)
        bias_vl = None

        def __init__(self):
            self.used = False

        def __call__(self, hidden):
            if self.used:
                raise AssertionError("the BF16 gate must not be recomputed")
            self.used = True
            assert hidden.dtype == torch.bfloat16
            # Deliberately different from hidden itself and the FP8 activation.
            weight = hidden.new_tensor([[0, -1], [1, 0], [0, 0]])
            return hidden[..., :3] @ weight, None

    def shared(hidden):
        assert hidden.dtype == torch.bfloat16
        assert hidden.shape[-1] == 3
        return hidden * 2

    runner = runner_mod.VllmMoERunner.__new__(runner_mod.VllmMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace(
        quant_method=ReferenceMethod(),
        _ensure_moe_quant_config_init=lambda: None,
    )
    runner.moe_config = SimpleNamespace(
        hidden_dim=4 if padded else 3,
        is_sequence_parallel=True,
        dp_size=1,
        pcp_size=1,
        tp_size=1,
        ep_size=1,
        skip_final_all_reduce=False,
    )
    gate = Gate()
    runner.gate = gate if internal_gate else None
    runner._fse_fuse_gate = False
    runner._shared_experts = SimpleNamespace(_layer=shared)
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 2.0
    runner.router = SimpleNamespace()
    runner._encode_layer_name = lambda: "test-moe"
    monkeypatch.setattr(runner_mod.envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(runner_mod._moe_runner, "get_layer_from_name", lambda _: runner)
    monkeypatch.setattr(
        runner_mod._moe_runner,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=None),
    )

    moe = _moe(sp=True, hash_table=gate.tid2eid, naive=False)
    object.__setattr__(moe, "gate", gate)
    object.__setattr__(moe, "experts", runner)
    return moe


def _fp8_shared_mlp(monkeypatch):
    """Run the production MLP/linear math through JAX on CPU, without a TPU bridge."""

    def run_jax(fn, *args):
        dtypes = {
            torch.bfloat16: jnp.bfloat16,
            torch.float32: jnp.float32,
            torch.float8_e4m3fn: jnp.float8_e4m3fn,
        }
        arrays = [
            jnp.asarray(arg.float().numpy(), dtype=dtypes[arg.dtype]) for arg in args
        ]
        result = fn(*arrays)
        return torch.from_numpy(np.asarray(result, dtype=np.float32).copy()).to(
            torch.bfloat16
        )

    def prequantized_op(output_dtype):
        assert output_dtype == torch.bfloat16
        return lambda *args: run_jax(
            lambda *arrays: linear_common._quantized_matmul_prequantized_jax(
                *arrays, output_dtype=jnp.bfloat16
            ),
            *args,
        )

    monkeypatch.setattr(
        linear_common, "_get_quantized_matmul_prequantized_op", prequantized_op
    )
    monkeypatch.setattr(
        linear_common,
        "_get_quantized_matmul_op",
        lambda: lambda *args: run_jax(linear_common._quantized_matmul_jax, *args),
    )

    class GateUp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "weight",
                torch.tensor(
                    [[1, -1, 2, 0], [0, 1, 1, -1], [-1, 0, 0, 1]],
                    dtype=torch.float8_e4m3fn,
                ),
            )
            self.register_buffer("weight_scale", torch.tensor([0.5, 1.0, 0.25, 0.5]))
            self.gather_output = False
            self.quant_method = object.__new__(VllmFp8LinearMethodTPU)

        def forward(self, x):
            return self.quant_method.apply(self, x), None

    class Down(torch.nn.Module):
        def forward(self, x):
            weight = x.new_tensor([[1, 0, -1], [0, 2, 1]])
            return x @ weight, None

    shared = DeepseekV4MLP.__new__(DeepseekV4MLP)
    torch.nn.Module.__init__(shared)
    shared.gate_up_proj = GateUp()
    shared.act_fn = lambda x: torch.nn.functional.silu(x[..., :2]) * x[..., 2:]
    shared.down_proj = Down()
    return shared


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("distinct_shared_input", [False, True])
def test_shared_fp8_reuse_preserves_input_identity_and_routed_padding(
    monkeypatch, padded, distinct_shared_input
):
    moe = _prequantized_moe(
        monkeypatch,
        internal_gate=True,
        method_type=mod.VllmDeepseekV4Mxfp4MoEMethod,
        padded=padded,
    )
    shared = _fp8_shared_mlp(monkeypatch)
    moe.experts._shared_experts = SimpleNamespace(_layer=shared)
    hidden = torch.tensor([[1, 4, 2], [-2, -3, 1]], dtype=torch.bfloat16)
    q = torch.tensor([[2, -1, 3], [4, 1, -2]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[0.5], [2.0]])
    ids = torch.tensor([2, 5], dtype=torch.int32)
    separate = hidden + 1 if distinct_shared_input else None
    output = moe.experts.forward_prequantized(
        hidden,
        None,
        input_ids=ids,
        shared_experts_input=separate,
        hidden_states_q=q,
        hidden_states_scale=scale,
    )
    if distinct_shared_input:
        shared_expected = shared(separate)
    else:
        projected = (
            (q.float() @ shared.gate_up_proj.weight.float())
            * scale
            * shared.gate_up_proj.weight_scale
        ).to(torch.bfloat16)
        activated = torch.nn.functional.silu(projected[..., :2]) * projected[..., 2:]
        shared_expected = activated @ hidden.new_tensor([[1, 0, -1], [0, 2, 1]])
    routed = (q.float() * scale * torch.tensor([[3], [7]])).to(torch.bfloat16)
    torch.testing.assert_close(output, shared_expected + routed * 2, rtol=0, atol=0)


@pytest.mark.parametrize("internal_gate", [False, True])
@pytest.mark.parametrize(
    "method_type", [mod.VllmFp8MoEMethodTPU, mod.VllmDeepseekV4Mxfp4MoEMethod]
)
@pytest.mark.parametrize("padded", [False, True])
def test_prequantized_moe_uses_bf16_gate_and_shared_input(
    monkeypatch, internal_gate, method_type, padded
):
    moe = _prequantized_moe(
        monkeypatch,
        internal_gate=internal_gate,
        method_type=method_type,
        padded=padded,
    )
    hidden = torch.tensor([[1, 4, 2], [-2, -3, 1]], dtype=torch.bfloat16)
    q = torch.tensor([[2, -1, 3], [4, 1, -2]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[0.5], [2.0]])
    ids = torch.tensor([2, 5], dtype=torch.int32)

    def unexpected_gather(*args, **kwargs):
        raise AssertionError("internally dispatched hash ids must stay rank-local")

    monkeypatch.setattr(
        mod, "get_ep_group", lambda: SimpleNamespace(all_gather=unexpected_gather)
    )
    output = moe.forward(hidden, ids, hidden_states_q=q, hidden_states_scale=scale)
    # Gate logits are [[4, -1], [-3, 2]], selecting experts 0 and 1.
    routed = q.float() * scale * torch.tensor([[3], [7]])
    expected = hidden * 2 + routed.to(torch.bfloat16) * 2
    assert output.shape == hidden.shape
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["q", "scale"])
@pytest.mark.parametrize("entry", ["model", "runner"])
def test_prequantized_inputs_require_both_operands(monkeypatch, missing, entry):
    moe = _prequantized_moe(
        monkeypatch, internal_gate=False, method_type=mod.VllmFp8MoEMethodTPU
    )
    hidden = torch.ones(2, 3, dtype=torch.bfloat16)
    kwargs = {
        "hidden_states_q": None if missing == "q" else hidden.to(torch.float8_e4m3fn),
        "hidden_states_scale": None if missing == "scale" else torch.ones(2, 1),
    }
    with pytest.raises(ValueError, match="must be paired"):
        if entry == "model":
            moe.forward(hidden, **kwargs)
        else:
            moe.experts.forward_prequantized(hidden, None, **kwargs)


def test_prequantized_runner_requires_logits_without_internal_gate(monkeypatch):
    moe = _prequantized_moe(
        monkeypatch, internal_gate=False, method_type=mod.VllmFp8MoEMethodTPU
    )
    hidden = torch.ones(2, 3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="router logits or a gate"):
        moe.experts.forward_prequantized(
            hidden,
            None,
            hidden_states_q=hidden.to(torch.float8_e4m3fn),
            hidden_states_scale=torch.ones(2, 1),
        )


@pytest.mark.parametrize("routing", ["hash", "vision"])
def test_prequantized_model_preserves_missing_ids_error(monkeypatch, routing):
    moe = _prequantized_moe(
        monkeypatch, internal_gate=False, method_type=mod.VllmFp8MoEMethodTPU
    )
    if routing == "vision":
        moe.gate.tid2eid = None
        moe.gate.bias_vl = torch.ones(2)
    hidden = torch.ones(2, 3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=f"{routing} MoE routing requires input_ids"):
        moe.forward(
            hidden,
            hidden_states_q=hidden.to(torch.float8_e4m3fn),
            hidden_states_scale=torch.ones(2, 1),
        )
