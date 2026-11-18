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
"""The MLA prefill rows block: which layers take it, and that a rows bucket's
backbone trace lifts the same named inputs as a decode bucket's (the warmup
hands a bucket over only between such traces). No TPU."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch._dynamo
from torch import nn

import vllm_torchtpu.envs as envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter import fused_moe_ep
from vllm_torchtpu.layers.adapter import mla_prefill_rows as rows
from vllm_torchtpu.layers.adapter import moe_prefill_sp as sp

TP, HIDDEN = 4, 8


@pytest.mark.parametrize(
    ("skip_topk", "want"),
    [
        ([False, True, True], [True, True, True]),
        # A leading layer that reuses a dense layer's top-k stays on the split.
        ([True, False, True], [False, True, True]),
        ([True, True], [False, False]),
    ],
)
def test_rows_layers_start_at_an_indexer_layer(skip_topk, want):
    assert rows.rows_layers(skip_topk) == want


@pytest.mark.parametrize(
    ("sequence_parallel", "dcp", "q_replicate", "want"),
    [
        (True, TP, True, True),
        # The head-sharded query layout has no rows variant.
        (True, TP, False, False),
        # No token-parallel split (off, dense MLA, or no DCP).
        (True, None, True, False),
        (False, TP, True, False),
        # DCP over part of the TP group: not every rank holds a DCP shard.
        (True, TP // 2, True, False),
    ],
)
def test_rows_block_serves_the_token_parallel_split_under_query_replication(
    monkeypatch, sequence_parallel, dcp, q_replicate, want
):
    monkeypatch.setattr(envs, "TPU_MOE_PREFILL_SEQUENCE_PARALLEL", sequence_parallel)
    monkeypatch.setattr(rows, "get_tensor_model_parallel_world_size", lambda: TP)
    layout = None if dcp is None else SimpleNamespace(dcp_size=dcp)
    attn = SimpleNamespace(token_parallel_layout=layout, dcp_q_replicate=q_replicate)
    assert rows.serves(attn) is want


def test_a_bucket_takes_the_block_when_every_rank_holds_the_dcp_rows(monkeypatch):
    monkeypatch.setattr(rows, "get_tensor_model_parallel_world_size", lambda: TP)
    layer = _Layer(False)
    assert not rows.takes(layer, 4096)
    setattr(layer.self_attn.mla_attn, rows.ROWS_ATTR, rows._Rows(None, None, 512))
    assert not rows.takes(layer, 1024) and rows.takes(layer, 2048)


def _served_mla(**attn):
    """An MLA wrapper with the formats the rows block takes (`attn` overrides
    the attention's)."""
    o_proj = SimpleNamespace(
        weight=torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones(2),
        bias=None,
    )
    setattr(o_proj, rows.WEIGHT_FLIPPED_ATTR, True)
    inner = dict(
        sparse_mla_tp_rows_op=object(),
        W_UK_T_dcp_qrep_scale=None,
        W_UV=torch.zeros(1, dtype=torch.bfloat16),
        kv_cache_quantized_dtype=torch.float8_e4m3fn,
    )
    return SimpleNamespace(
        mla_attn=SimpleNamespace(**(inner | attn)),
        q_lora_rank=8,
        kimi_preprocess=None,
        gate_is_fused=False,
        g_proj=None,
        is_sparse=True,
        dcp_q_replicate=True,
        o_proj=o_proj,
        skip_topk=True,
        indexer=None,
    )


def test_layers_without_the_block_formats_keep_the_split_path():
    assert rows.armable([_served_mla(), _served_mla()])
    bf16_kv = _served_mla(kv_cache_quantized_dtype=None)
    assert not rows.armable([_served_mla(), bf16_kv])


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN))

    def forward(self, x, residual=None):
        if residual is None:
            return x * self.weight
        x = x + residual
        return x * self.weight, x


class _OProj(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.randn(HIDDEN, HIDDEN).to(torch.float8_e4m3fn)
        self.weight = nn.Parameter(w, requires_grad=False)
        self.weight_scale = nn.Parameter(torch.rand(HIDDEN), requires_grad=False)


class _Backend(nn.Module):
    """The backend MLA layer: W_UV and the decode copy are plain attributes."""

    def __init__(self):
        super().__init__()
        self.W_UV = torch.randn(2, 4, 4)
        self.W_UV_dcp_qrep = torch.randn(2, 4, 4)


class _Wrapper(nn.Module):
    def __init__(self, skip_topk):
        super().__init__()
        self.qkv = nn.Parameter(torch.randn(HIDDEN, HIDDEN))
        self.o_proj = _OProj()
        self.mla_attn = _Backend()
        self.skip_topk = skip_topk
        self.prefix = "layers.0.self_attn"


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(HIDDEN, HIDDEN))
        setattr(self, sp.SHARED_ATTR, (torch.randn(HIDDEN),))


class _Layer(nn.Module):
    forward = sp.patched_decoder_layer_forward

    def __init__(self, skip_topk=False):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.mla_attn = _Wrapper(skip_topk)
        self.mlp = _MLP()
        self.input_layernorm = _Norm()
        self.post_attention_layernorm = _Norm()


class _Backbone(nn.Module):
    forward = sp.patched_model_forward

    def __init__(self, layers):
        super().__init__()
        self.embed = nn.Embedding(16, HIDDEN)
        self.layers = nn.ModuleList(layers)
        self.norm = _Norm()
        self.start_layer, self.end_layer = 0, len(layers)

    def embed_input_ids(self, input_ids):
        return self.embed(input_ids)


def _upstream_layer(self, positions, hidden_states, residual, llama_4_scaling=None):
    """The decode buckets' layer: reads the attention through the modules."""
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    mla = self.self_attn.mla_attn
    out = hidden_states @ mla.qkv + mla.mla_attn.W_UV_dcp_qrep.sum()
    out = out + mla.o_proj.weight.float().sum() * mla.o_proj.weight_scale
    hidden_states, residual = self.post_attention_layernorm(out, residual)
    return hidden_states @ self.mlp.w, residual


def _upstream_model(self, input_ids, positions, intermediate_tensors=None, embeds=None):
    hidden_states, residual = self.embed_input_ids(input_ids), None
    for layer in self.layers[self.start_layer : self.end_layer]:
        hidden_states, residual = layer(positions, hidden_states, residual)
    return self.norm(hidden_states, residual)[0]


def _split_layer(layer, positions, hidden_states, residual, llama_4_scaling):
    hidden_states, residual = _upstream_layer(layer, positions, hidden_states, residual)
    rank = fused_moe_ep.rank_operand()
    return sp.own_rows(hidden_states, rank, TP), sp.own_rows(residual, rank, TP)


def _rows_forward(mla, positions, hidden_states, gathered, rank):
    """The rows block: the wrapper's tensors through the modules, the shards
    from the prefetch."""
    w_o, w_uv, w_scale = gathered
    return hidden_states @ mla.qkv + w_o.sum() + w_uv.sum() + w_scale.sum()


def _gather(w_o, w_uv, w_scale, after):
    return w_o.float().repeat(TP, 1), w_uv.repeat(TP, 1, 1), w_scale.repeat(TP)


def _lifted(model, num_tokens):
    """The backbone trace's inputs as the warmup hand-over compares them."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    compiled(torch.zeros(num_tokens, dtype=torch.long), torch.arange(num_tokens))
    (graph,) = graphs
    return dict(shape_variants._handover_inputs(graph))


def test_rows_and_decode_traces_lift_the_same_named_inputs(monkeypatch):
    for name, value in {
        "TPU_MOE_PREFILL_SEQUENCE_PARALLEL": True,
        "TPU_DCP_PREFILL_TOKEN_PARALLEL": True,
    }.items():
        monkeypatch.setattr(envs, name, value)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: TP)
    monkeypatch.setattr(rows, "get_tensor_model_parallel_world_size", lambda: TP)
    monkeypatch.setattr(
        sp, "tensor_model_parallel_all_gather", lambda x, dim: x.repeat(TP, 1)
    )
    monkeypatch.setattr(sp, "UPSTREAM_MODEL_FORWARD", _upstream_model)
    monkeypatch.setattr(sp, "UPSTREAM_DECODER_LAYER_FORWARD", _upstream_layer)
    monkeypatch.setattr(sp, "_sequence_parallel_layer_forward", _split_layer)
    monkeypatch.setattr(sp, "_moe", lambda mlp, x, num_tokens, tp: x @ mlp.w)
    monkeypatch.setattr(rows, "forward", _rows_forward)
    monkeypatch.setattr(shape_variants, "in_warmup", lambda: True)
    monkeypatch.setattr(fused_moe_ep, "_RANK_BUFFER", torch.tensor([[1]]))

    # A dense layer, a MoE layer reusing its top-k (stays on the split), then
    # an indexer layer and one reusing its top-k (both take the rows block).
    layers = [_Layer(), _Layer(True), _Layer(False), _Layer(True)]
    armed = [False, *rows.rows_layers([True, False, True])]
    for layer, armed_rows in zip(layers[1:], armed[1:]):
        if armed_rows:
            setattr(
                layer.self_attn.mla_attn, rows.ROWS_ATTR, rows._Rows(_gather, None, 16)
            )
        setattr(layer, sp.KEEP_ATTR, sp.layer_keep(layer, armed_rows))
    model = _Backbone(layers)
    setattr(model, sp.KEEP_ATTR, True)
    with patch.object(sp.vllm_compilation_backends, "model_tag", "backbone"):
        assert rows.applies(2048) and not sp.applies(16)
        decode, rows_bucket = _lifted(model, 16), _lifted(model, 2048)
    assert sorted(rows_bucket) == sorted(decode)
    with pytest.raises(RuntimeError, match="must start with keep_list"):
        rows.check_keep(
            layers[2].self_attn.mla_attn, getattr(layers[2], sp.KEEP_ATTR)[1:]
        )
    for name, meta in decode.items():
        if not name.startswith(("l_input_ids", "l_positions")):
            assert rows_bucket[name] == meta, name
