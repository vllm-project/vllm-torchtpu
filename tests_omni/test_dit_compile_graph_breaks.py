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
"""vllm-torchtpu's DiT patches must trace as one graph under torch.compile.

A tiny Wan-style transformer block runs every vllm-torchtpu code path a
compiled Wan block takes (fused q/k RMSNorm + RoPE, pair-swap RoPE, Pallas
flash attention and its SDPA fallback, the FFN fusion fence) with the Pallas
ops replaced by CPU reference custom ops of the same signatures. It is
compiled with backend="eager", fullgraph=True, so any Python-side graph break
fails the test without a TPU. vLLM-Omni's own layers are not built here (that
needs a resolvable vLLM platform); they are covered on TPU.
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from vllm_torchtpu.omni import attention as omni_attention
from vllm_torchtpu.omni import patches
from vllm_torchtpu.omni import rope as omni_rope

pytestmark = pytest.mark.omni

_WAN = "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer"
_ROPE = "vllm_omni.diffusion.layers.rope"
_NORM = "vllm_omni.diffusion.layers.norm"

_CALLS: dict[str, int] = {}


def _count(name: str) -> None:
    _CALLS[name] = _CALLS.get(name, 0) + 1


# Reference custom ops with the Pallas ops' signatures and fakes.
@torch.library.custom_op("o15test::flash_attention", mutates_args=())
def _flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    _count("flash_attention")
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
    )
    return out.transpose(1, 2).contiguous()


@_flash_attention.register_fake
def _(q, k, v, scale):
    return q.new_empty((*q.shape[:3], v.shape[-1]))


def _rms_norm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(
        x.dtype
    )


@torch.library.custom_op("o15test::wan_qk_norm_rope", mutates_args=())
def _wan_qk_norm_rope(
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _count("qk_norm_rope")
    dim = qkv.shape[1] // 3
    swap = omni_rope._pair_swap_matrix(head_dim, qkv.dtype, qkv.device)
    outs = []
    for j, w in ((0, wq), (1, wk)):
        y = _rms_norm(qkv[:, j * dim : (j + 1) * dim], w, eps).unflatten(
            1, (-1, head_dim)
        )
        outs.append(
            (y * cos[:, None] + (y @ swap) * sin[:, None]).flatten(1).to(qkv.dtype)
        )
    return outs[0], outs[1]


@_wan_qk_norm_rope.register_fake
def _(qkv, wq, wk, cos, sin, eps, head_dim):
    shape = (qkv.shape[0], qkv.shape[1] // 3)
    return qkv.new_empty(shape), qkv.new_empty(shape)


@torch.library.custom_op("o15test::optimization_barrier", mutates_args=())
def _optimization_barrier(x: torch.Tensor) -> torch.Tensor:
    _count("optimization_barrier")
    return x.clone()


@_optimization_barrier.register_fake
def _(x):
    return torch.empty_like(x)


# Fresh stand-ins for the vLLM-Omni classes the patches replace forwards on.
class _RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size).uniform_(0.5, 1.5))
        self.variance_epsilon = eps


def _omni_modules():
    norm = types.ModuleType(_NORM)
    norm.RMSNorm = _RMSNorm

    class RotaryEmbeddingWan(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.interleaved = True
            self.half_head_dim = True
            self._forward_method = self.forward_native

        def forward_native(self, x, cos, sin):
            x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
            out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
            return out.flatten(-2, -1).to(x.dtype)

    rope = types.ModuleType(_ROPE)
    rope.RotaryEmbeddingWan = RotaryEmbeddingWan

    class _Linear(torch.nn.Linear):
        """Returns (out, bias) like vLLM's parallel linears."""

        def forward(self, x):
            return super().forward(x), None

    class WanSelfAttention(torch.nn.Module):
        def __init__(self, dim, num_heads, attn):
            super().__init__()
            self.num_heads = self.num_kv_heads = num_heads
            self.head_dim = dim // num_heads
            self.to_qkv = _Linear(dim, 3 * dim)
            self.norm_q = _RMSNorm(dim)
            self.norm_k = _RMSNorm(dim)
            self.to_gate_compress = None
            self.attn = attn
            self.dropout = torch.nn.Identity()
            self.to_out = torch.nn.Linear(dim, dim)

        def forward(self, hidden_states, rotary_emb=None, attn_metadata=None):
            raise AssertionError("the fused TPU forward must replace this")

    class ColumnParallelGELU(torch.nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.proj = torch.nn.Linear(dim_in, dim_out)

        def forward(self, x):
            return F.gelu(self.proj(x), approximate="tanh")

    wan = types.ModuleType(_WAN)
    wan.WanSelfAttention = WanSelfAttention
    wan.ColumnParallelGELU = ColumnParallelGELU
    return {_NORM: norm, _ROPE: rope, _WAN: wan}


class _AttentionLayer(torch.nn.Module):
    """vLLM-Omni Attention stand-in: dispatches to the selected impl."""

    def __init__(self, impl):
        super().__init__()
        self.impl = impl

    def forward(self, query, key, value, attn_metadata=None):
        return self.impl.forward(query, key, value, attn_metadata)


def _flash_impl(num_heads, head_dim, causal=False):
    return omni_attention.TpuFlashAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        causal=causal,
        softmax_scale=head_dim**-0.5,
    )


class _TinyWanBlock(torch.nn.Module):
    """WanTransformerBlock.forward's structure at toy size."""

    def __init__(self, modules, dim=256, num_heads=2, ffn_dim=512):
        super().__init__()
        head_dim = dim // num_heads
        wan, rope = modules[_WAN], modules[_ROPE]
        self.heads = (num_heads, head_dim)
        self.scale_shift_table = torch.nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False)
        self.attn1 = wan.WanSelfAttention(
            dim, num_heads, _AttentionLayer(_flash_impl(num_heads, head_dim))
        )
        self.rope = rope.RotaryEmbeddingWan()
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_kv = torch.nn.Linear(dim, 2 * dim)
        self.cross = _AttentionLayer(_flash_impl(num_heads, head_dim))
        # A causal layer always takes the SDPA fallback (and its warning).
        self.cross_fallback = _AttentionLayer(
            _flash_impl(num_heads, head_dim, causal=True)
        )
        self.to_out = torch.nn.Linear(dim, dim)
        self.ffn_in = wan.ColumnParallelGELU(dim, ffn_dim)
        self.ffn_out = torch.nn.Linear(ffn_dim, dim)

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        shift, scale, gate, c_shift, c_scale, c_gate = (
            self.scale_shift_table + temb
        ).chunk(6, dim=1)
        x = hidden_states
        h = (self.norm(x.float()) * (1 + scale) + shift).type_as(x)
        x = (x + self.attn1(h, rotary_emb, None) * gate).type_as(x)

        cos, sin = rotary_emb
        q = self.to_q(self.norm(x)).unflatten(2, self.heads)
        q = self.rope(q, cos, sin)
        k, v = self.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        k, v = k.unflatten(2, self.heads), v.unflatten(2, self.heads)
        attn = self.cross(q, k, v) + self.cross_fallback(q, k, v)
        x = x + self.to_out(attn.flatten(2))

        h = (self.norm(x.float()) * (1 + c_scale) + c_shift).type_as(x)
        x = (x + self.ffn_out(self.ffn_in(h)) * c_gate).type_as(x)
        return x


@pytest.fixture
def tiny_wan_block(monkeypatch):
    modules = _omni_modules()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # The Pallas ops, as built eagerly before the first compiled forward.
    monkeypatch.setattr(patches, "_optimization_barrier", _optimization_barrier)
    monkeypatch.setattr(omni_rope, "_qk_norm_rope", _wan_qk_norm_rope)
    monkeypatch.setattr(omni_attention, "_flash_attention", _flash_attention)
    patches._patch_wan_pair_swap_rope()
    patches._patch_wan_ffn_gelu_fence()
    patches._patch_wan_fused_qk_norm_rope()
    torch.manual_seed(0)
    block = _TinyWanBlock(modules).to(torch.bfloat16)
    torch._dynamo.reset()
    _CALLS.clear()
    yield block
    torch._dynamo.reset()


def _inputs(seq=64, ctx=16, dim=256, head_dim=128):
    return (
        torch.randn(1, seq, dim, dtype=torch.bfloat16),
        torch.randn(1, ctx, dim, dtype=torch.bfloat16),
        torch.randn(1, 6, dim, dtype=torch.bfloat16),
        (
            torch.randn(1, seq, 1, head_dim // 2, dtype=torch.bfloat16),
            torch.randn(1, seq, 1, head_dim // 2, dtype=torch.bfloat16),
        ),
    )


def test_tiny_wan_block_traces_as_one_graph(tiny_wan_block):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    inputs = _inputs()
    with torch.no_grad():
        eager = tiny_wan_block(*inputs)
        eager_calls = dict(_CALLS)
        _CALLS.clear()
        compiled = torch.compile(tiny_wan_block, backend=backend, fullgraph=True)
        out = compiled(*inputs)
    assert len(graphs) == 1
    # Every patched TPU path ran inside the graph.
    assert eager_calls == {
        "qk_norm_rope": 1,
        "flash_attention": 2,
        "optimization_barrier": 1,
    }
    assert eager_calls == _CALLS
    targets = {str(n.target) for n in graphs[0].graph.nodes if n.op == "call_function"}
    for op in (
        "o15test.flash_attention",
        "o15test.wan_qk_norm_rope",
        "o15test.optimization_barrier",
    ):
        assert any(op in t for t in targets), (op, sorted(targets))
    torch.testing.assert_close(out, eager)


def test_tiny_wan_block_does_not_recompile_across_steps(tiny_wan_block):
    compiled = torch.compile(
        tiny_wan_block, backend="eager", fullgraph=True, dynamic=False
    )
    with torch.no_grad():
        compiled(*_inputs())
        with torch._dynamo.config.patch(error_on_recompile=True):
            for _ in range(3):
                compiled(*_inputs())


def test_rope_tables_are_not_cached_while_tracing(monkeypatch):
    # With regional compile no eager forward runs first, so the RoPE helper
    # tables are first built inside the traced graph. They must not be
    # stored then (a traced tensor in a module-level cache); the eager call
    # afterwards fills the cache as before.
    monkeypatch.setattr(omni_rope, "_PAIR_SWAP_MATRICES", {})
    monkeypatch.setattr(omni_rope, "_FIRST_HALF_LANES", {})
    torch._dynamo.reset()
    head_dim, heads, seq = 16, 2, 8
    x = torch.randn(1, seq, heads, head_dim, dtype=torch.bfloat16)
    cos = torch.randn(1, seq, 1, head_dim // 2, dtype=torch.bfloat16)
    sin = torch.randn(1, seq, 1, head_dim // 2, dtype=torch.bfloat16)
    xs = torch.randn(1, seq, heads * head_dim, dtype=torch.bfloat16)
    cos_s = torch.randn(1, 1, seq, heads * head_dim, dtype=torch.bfloat16)
    sin_s = torch.randn(1, 1, seq, heads * head_dim, dtype=torch.bfloat16)
    try:
        pair = torch.compile(
            omni_rope.apply_rotary_pair_swap, backend="eager", fullgraph=True
        )(x, cos, sin)
        split = torch.compile(
            omni_rope.apply_split_rotary_half_swap, backend="eager", fullgraph=True
        )(xs, cos_s, sin_s, head_dim)
        assert omni_rope._PAIR_SWAP_MATRICES == {}
        assert omni_rope._FIRST_HALF_LANES == {}
    finally:
        torch._dynamo.reset()

    torch.testing.assert_close(pair, omni_rope.apply_rotary_pair_swap(x, cos, sin))
    torch.testing.assert_close(
        split, omni_rope.apply_split_rotary_half_swap(xs, cos_s, sin_s, head_dim)
    )
    assert len(omni_rope._PAIR_SWAP_MATRICES) == 1
    assert len(omni_rope._FIRST_HALF_LANES) == 1


def test_op_accessors_are_not_functools_caches():
    # Dynamo ignores functools.cache wrappers and re-traces the op builder
    # (pallas.jax_op) or tensor factory inside the compiled forward.
    assert not hasattr(patches._optimization_barrier_op, "cache_info")
    assert not hasattr(omni_rope._qk_norm_rope_op, "cache_info")
    assert not hasattr(omni_attention._flash_attention_op, "cache_info")
    assert not hasattr(omni_rope._pair_swap_matrix, "cache_info")
    assert not hasattr(omni_rope._first_half_lanes, "cache_info")


def test_op_accessors_build_once(monkeypatch):
    built = []
    monkeypatch.setattr(omni_rope, "_qk_norm_rope", None)
    monkeypatch.setattr(
        omni_rope, "_build_qk_norm_rope_op", lambda: built.append("rope") or "rope_op"
    )
    assert omni_rope._qk_norm_rope_op() == "rope_op"
    assert omni_rope._qk_norm_rope_op() == "rope_op"

    monkeypatch.setattr(omni_attention, "_flash_attention", None)
    monkeypatch.setattr(
        omni_attention,
        "_build_flash_attention_op",
        lambda: built.append("flash") or "flash_op",
    )
    assert omni_attention._flash_attention_op() == "flash_op"
    assert omni_attention._flash_attention_op() == "flash_op"
    assert built == ["rope", "flash"]

    eager_built = []
    monkeypatch.setattr(
        patches, "_optimization_barrier_op", lambda: eager_built.append("barrier")
    )
    monkeypatch.setattr(
        omni_rope, "_qk_norm_rope_op", lambda: eager_built.append("rope")
    )
    monkeypatch.setattr(
        omni_attention, "_flash_attention_op", lambda: eager_built.append("flash")
    )
    patches._build_dit_ops_eagerly()
    assert eager_built == ["barrier", "rope", "flash"]


def test_tiny_flux_tp_single_block_traces_as_one_graph(tiny_wan_block):
    # Exercises FluxSingleTransformerBlock structure under functional TP
    # collectives (all_gather_into_tensor + all_reduce) + pair-swap RoPE +
    # TpuFlashAttentionImpl with fullgraph=True.
    import torch.distributed._functional_collectives as funcol

    dim = 128
    num_heads = 2
    head_dim = dim // num_heads
    rope_mod = tiny_wan_block.rope
    attn_layer = _AttentionLayer(_flash_impl(num_heads, head_dim))
    qkv_proj = torch.nn.Linear(dim, 3 * dim, dtype=torch.bfloat16)
    mlp_proj = torch.nn.Linear(dim, 2 * dim, dtype=torch.bfloat16)
    out_proj = torch.nn.Linear(3 * dim, dim, dtype=torch.bfloat16)

    def flux_single_block(hidden_states, encoder_hidden_states, cos, sin):
        txt_len = encoder_hidden_states.shape[1]
        cat = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        q, k, v = qkv_proj(cat).chunk(3, dim=-1)
        q = rope_mod(q.unflatten(-1, (num_heads, head_dim)), cos, sin)
        k = rope_mod(k.unflatten(-1, (num_heads, head_dim)), cos, sin)
        v = v.unflatten(-1, (num_heads, head_dim))
        attn_out = attn_layer(q, k, v).flatten(2, 3)
        # Single-rank functional collective no-ops trace through the same
        # Dynamo collective path as multi-rank TP.
        if torch.distributed.is_initialized():
            attn_out = funcol.all_reduce(attn_out, "sum", group=[0])
        mlp_out = F.gelu(mlp_proj(cat), approximate="tanh")
        out = cat + out_proj(torch.cat([attn_out, mlp_out], dim=-1))
        return out[:, :txt_len], out[:, txt_len:]

    h = torch.randn(1, 16, dim, dtype=torch.bfloat16)
    eh = torch.randn(1, 8, dim, dtype=torch.bfloat16)
    cos = torch.ones(1, 24, 1, head_dim // 2, dtype=torch.bfloat16)
    sin = torch.zeros(1, 24, 1, head_dim // 2, dtype=torch.bfloat16)

    explanation = torch._dynamo.explain(flux_single_block)(h, eh, cos, sin)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    compiled = torch.compile(flux_single_block, backend="eager", fullgraph=True)
    eh_c, h_c = compiled(h, eh, cos, sin)
    eh_r, h_r = flux_single_block(h, eh, cos, sin)
    torch.testing.assert_close(eh_c, eh_r, rtol=0, atol=0)
    torch.testing.assert_close(h_c, h_r, rtol=0, atol=0)
