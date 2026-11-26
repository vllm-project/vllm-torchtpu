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
"""A MiniMax-M3 MoE layer whose shared expert TPU_MOE_DENSE_FP4_SHARED armed:
the shared expert runs exactly once per step, and every bucket's traced
subgraph takes the same inputs in the same order (shape_variants hands buckets
between traces only on an equal signature; vLLM's split_graph orders subgraph
inputs by first use).

The routed op is a stand-in that consumes its tensor operands in call order;
the real nvfp4 apply_monolithic builds the call. Which jax kernel serves a
step (dense, sparse rows, GMM) lives inside that one op and adds no input."""

import types

import pytest
import torch
from torch import nn

pytest.importorskip("torch_tpu")

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.layers.adapter import dense_fp4_shared as dfs  # noqa: E402
from vllm_torchtpu.layers.adapter.quantization import nvfp4  # noqa: E402
from vllm_torchtpu.models.minimax_m3 import moe as moe_mod  # noqa: E402
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MoE  # noqa: E402

H, I_S, E, TOPK = 64, 32, 8, 2
BF16, E4M3, F32 = torch.bfloat16, torch.float8_e4m3fn, torch.float32
SP_MIN = 1024
BUCKETS = [16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 4096, 8192]


def _param(*shape, dtype=BF16):
    return nn.Parameter(torch.randn(*shape).to(dtype), requires_grad=False)


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = _param(H, E, dtype=F32)

    def forward(self, x):
        return x.float() @ self.weight, None


class _Routed(nn.Module):
    def __init__(self, skip_final_all_reduce):
        super().__init__()
        self.w13_weight = _param(E, H, 2 * 16)
        self.w2_weight = _param(E, 16, H)
        self.w13_weight_scale = _param(E, 1, 32, dtype=F32)
        self.w2_weight_scale = _param(E, 1, H, dtype=F32)
        self._experts_start = None
        self.moe_config = types.SimpleNamespace(
            experts_per_token=TOPK, skip_final_all_reduce=skip_final_all_reduce
        )
        # Armed as dense_fp4_shared does: the shared linears' e4m3 weights and
        # f32 [1, n] channel scales as non-persistent buffers.
        for name, t in zip(
            dfs._SHARED_BUFFERS,
            (
                torch.randn(H, 2 * I_S).to(E4M3),
                torch.full((1, 2 * I_S), 2.0**-4),
                torch.randn(I_S, H).to(E4M3),
                torch.full((1, H), 2.0**-5),
            ),
        ):
            self.register_buffer(name, t, persistent=False)
        setattr(self, dfs.SHARED_ROWS_ATTR, None)


def _layer(monkeypatch, skip_final_all_reduce=True):
    calls = []

    def routed_op(**kw):
        """Consumes every tensor operand, in the op's argument order."""
        calls.append(kw)
        x = kw["hidden_states"]
        out = x * 1
        tensors = [v for k, v in kw.items() if isinstance(v, torch.Tensor)]
        tensors += list(kw.get("shared") or ())
        for piece in kw.get("shared_rows") or ():
            tensors += list(piece)
        for t in tensors:
            out = out + (t.float().mean() * 0).to(x.dtype)
        return out

    monkeypatch.setattr(nvfp4, "fused_moe_gmm", routed_op)
    monkeypatch.setattr(
        moe_mod, "_materialized_bf16", lambda w: w.to(BF16), raising=True
    )
    monkeypatch.setattr(
        moe_mod,
        "_e4m3_matmul",
        lambda q, rs, w, ws: ((q.float() @ w.float()) * rs * ws.float()).to(BF16),
    )

    qm = object.__new__(nvfp4.VllmNvfp4MoEMethod)
    qm._tpu_activation_str = "swigluoai"
    qm._use_adaptive_fused_moe = False
    qm._gmm_route = lambda layer, x, logits: torch.topk(torch.sigmoid(logits), TOPK)

    routed = _Routed(skip_final_all_reduce)

    class Runner(nn.Module):
        def __init__(self):
            super().__init__()
            self.routed_experts = routed

        def forward(self, hidden_states, router_logits):
            # The armed runner holds no shared expert: routed only.
            return qm.apply_monolithic(routed, hidden_states, router_logits)

    shared = types.SimpleNamespace(
        act_fn=lambda gu: torch.nn.functional.silu(gu[..., :I_S]) * gu[..., I_S:]
    )
    moe = object.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.gate = _Gate()
    moe.experts = Runner()
    object.__setattr__(moe, "shared_experts", shared)
    return moe, routed, calls


def _forward(moe):
    def fwd(x):
        if x.shape[0] >= SP_MIN:
            # An SP step's e4m3 all-gather: rows with per-row f32 scales.
            scale = x.float().abs().amax(-1, keepdim=True).clamp(min=1e-6) / 448
            return moe(((x.float() / scale).to(E4M3), scale))
        return moe(x)

    return fwd


def _signatures(fn, tokens):
    from vllm.compilation.backends import split_graph

    from vllm_torchtpu.compilation import shape_variants

    out = []

    def backend(gm, example_inputs):
        _, items = split_graph(gm, [])
        out.extend(shape_variants.graph_signature(it.graph) for it in items)
        return gm.forward

    torch._dynamo.reset()
    x = torch.randn(tokens, H).to(BF16)
    torch._dynamo.mark_dynamic(x, 0)
    torch.compile(fn, backend=backend, fullgraph=True)(x)
    return out


@pytest.mark.parametrize("e4m3_shared", [False, True])
@pytest.mark.parametrize("model_side_min", [0, 4096])
@pytest.mark.parametrize("in_combine", [False, True])
def test_every_bucket_keeps_one_signature(
    monkeypatch, e4m3_shared, model_side_min, in_combine
):
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SP_E4M3_SHARED", e4m3_shared)
    monkeypatch.setattr(envs, "TPU_MOE_SHARED_MODEL_SIDE_MIN_TOKENS", model_side_min)
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SHARED_IN_COMBINE", in_combine)
    monkeypatch.setattr(envs, "TPU_MOE_TC_COMBINE", in_combine)
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SHARED_IN_COMBINE_MIN_TOKENS", 4096)
    moe, routed, calls = _layer(monkeypatch)
    sigs = {}
    for tokens in BUCKETS:
        calls.clear()
        sigs[tokens] = _signatures(_forward(moe), tokens)
        assert len(sigs[tokens]) == 1
        assert len(calls) == 1
        # Exactly one place runs the shared expert.
        model_side = 0 < model_side_min <= tokens
        assert (calls[0].get("shared") is None) == model_side
        rows = calls[0].get("shared_rows")
        if model_side or not (e4m3_shared and tokens >= SP_MIN):
            assert rows is None
        else:
            assert len(rows) == 1
        assert getattr(routed, dfs.SHARED_ROWS_ATTR) is None
    first = sigs[BUCKETS[0]]
    for tokens, sig in sigs.items():
        assert sig == first, (tokens, sig, first)


def test_model_side_shared_matches_routed_op_side(monkeypatch):
    """The model-side shared expert adds the same thing the routed op's
    stand-in would: out = routed + shared, once."""
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SP_E4M3_SHARED", False)
    monkeypatch.setattr(envs, "TPU_MOE_SHARED_MODEL_SIDE_MIN_TOKENS", 64)
    moe, routed, calls = _layer(monkeypatch)
    x = torch.randn(64, H).to(BF16)
    out = moe(x)
    assert calls[-1].get("shared") is None
    w1, s1, w2, s2 = dfs.shared_operands(routed)
    want = x + moe_mod._shared_expert(moe.shared_experts.act_fn, x, w1, s1, w2, s2)
    torch.testing.assert_close(out, want, rtol=0, atol=0)
    # Not the model's reduction: the routed op keeps it.
    moe, routed, calls = _layer(monkeypatch, skip_final_all_reduce=False)
    moe(x)
    assert calls[-1].get("shared") is not None
