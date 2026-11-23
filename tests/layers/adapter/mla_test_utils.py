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
"""Synthetic sparse (DSA) MLA layers for the MLA adapter and MTP tests: the
real `VllmTPUMultiHeadLatentAttentionWrapper` built through `__init__`, with
stand-in projections in the processed runtime layouts, and the same layer on
one TPU chip with the real indexer op, MLA layer, backend impl and fp8
caches."""

import dataclasses
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.model_executor.layers.mla import MLAModules

from tests.layers.adapter.indexer_test_utils import build_indexer, make_rope
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

ROPE = 64
NOPE = 192
KV_LORA = 512
Q_LORA = 128
V_DIM = 256
HIDDEN = 256
QK = NOPE + ROPE
MAX_POS = 4096
FP8_BLOCK = 128
IDX_HEAD_DIM = 128
IDX_N_HEAD = 8


def mla_rope(is_neox_style=False, scaling=False, dtype=torch.bfloat16):
    return make_rope(
        rope_dim=ROPE,
        max_position_embeddings=MAX_POS,
        dtype=dtype,
        is_neox_style=is_neox_style,
        scaling_factor=4.0 if scaling else None,
    )


class ProcessedLinear(torch.nn.Module):
    """A dense linear in a processed runtime layout (bf16 flipped, or FP8
    flipped with a blockwise scale)."""

    def __init__(self, n_in, n_out, layout, gen):
        super().__init__()
        w = torch.randn(n_in, n_out, generator=gen) / math.sqrt(n_in)
        self.layout = layout
        if layout == "fp8":
            nb = n_in // FP8_BLOCK
            blocks = w.view(nb, FP8_BLOCK, n_out)
            scale = blocks.abs().amax(1) / 448.0
            q = (blocks / scale[:, None, :]).to(torch.float8_e4m3fn)
            self.weight = torch.nn.Parameter(q.view(n_in, n_out), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(
                scale[None, :, None, :].contiguous(), requires_grad=False
            )
        else:
            self.weight = torch.nn.Parameter(w.bfloat16(), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, True)

    def dense(self):
        if self.layout == "fp8":
            s = self.weight_scale[0, :, 0, :].repeat_interleave(FP8_BLOCK, 0)
            return self.weight.float() * s
        return self.weight.float()

    def forward(self, x):
        return (x.float() @ self.dense()).to(torch.bfloat16), None


class RmsNorm(torch.nn.Module):
    def __init__(self, n, gen):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (1.0 + 0.2 * torch.randn(n, generator=gen)).bfloat16(), requires_grad=False
        )

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (xf * self.weight.float()).to(x.dtype)


class IdentityProjection(torch.nn.Module):
    def forward(self, x):
        return x, None


class RecordingAttention(torch.nn.Module):
    """Stands in for `VllmTPUMLAAttention`: records what `forward` hands it."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, q, kv_c_normed, k_pe, output_shape, **kwargs):
        self.calls.append((q, kv_c_normed, k_pe))
        return torch.zeros(output_shape, dtype=kv_c_normed.dtype)


def mla_wrapper(
    *,
    layout="flipped",
    seed=0,
    rope=None,
    gate_is_fused=False,
    q_lora=True,
    heads=4,
    indexer=None,
    mla_attn=None,
    topk_buffer=None,
):
    """A `VllmTPUMultiHeadLatentAttentionWrapper` built through `__init__`,
    with stand-in projections; the MLA attention it builds is `mla_attn`."""
    gen = torch.Generator().manual_seed(seed)
    gate = heads * V_DIM if gate_is_fused else 0
    modules = MLAModules(
        kv_a_layernorm=RmsNorm(KV_LORA, gen),
        kv_b_proj=torch.nn.Identity(),
        rotary_emb=rope if rope is not None else mla_rope(),
        o_proj=IdentityProjection(),
        fused_qkv_a_proj=(
            ProcessedLinear(HIDDEN, Q_LORA + KV_LORA + ROPE + gate, layout, gen)
            if q_lora
            else None
        ),
        kv_a_proj_with_mqa=(
            None if q_lora else ProcessedLinear(HIDDEN, KV_LORA + ROPE, layout, gen)
        ),
        q_a_layernorm=RmsNorm(Q_LORA, gen) if q_lora else None,
        q_b_proj=ProcessedLinear(Q_LORA, heads * QK, layout, gen) if q_lora else None,
        q_proj=None if q_lora else ProcessedLinear(HIDDEN, heads * QK, layout, gen),
        indexer=indexer,
        is_sparse=True,
        topk_indices_buffer=(
            topk_buffer
            if topk_buffer is not None
            else torch.zeros((MAX_POS, 8), dtype=torch.int32)
        ),
        indexer_rotary_emb=mla_rope() if indexer is not None else None,
    )
    with patch.object(
        mla_op,
        "VllmTPUMLAAttention",
        return_value=mla_attn if mla_attn is not None else RecordingAttention(),
    ):
        return mla_op.VllmTPUMultiHeadLatentAttentionWrapper(
            hidden_size=HIDDEN,
            num_heads=heads,
            scale=QK**-0.5,
            qk_nope_head_dim=NOPE,
            qk_rope_head_dim=ROPE,
            v_head_dim=V_DIM,
            q_lora_rank=Q_LORA if q_lora else None,
            kv_lora_rank=KV_LORA,
            mla_modules=modules,
            prefix="model.layers.0.self_attn.mla_attn",
            skip_topk=indexer is None,
            gate_is_fused=gate_is_fused,
        )


def dsa_indexer(seed=0, topk=8, buffer_tokens=16):
    """A `VllmTPUIndexer` with stand-in projections; its op echoes its
    arguments."""
    ind = build_indexer(
        sparse_attn_indexer=MagicMock(),
        dtype=torch.bfloat16,
        n_head=IDX_N_HEAD,
        head_dim=IDX_HEAD_DIM,
        rope_dim=ROPE,
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA,
        topk=topk,
        max_model_len=MAX_POS,
        buffer_tokens=buffer_tokens,
        prefill_workspace_factor=1,
        prefix="model.layers.0.self_attn.indexer",
    )
    gen = torch.Generator().manual_seed(seed)
    ind.wq_b = ProcessedLinear(Q_LORA, IDX_N_HEAD * IDX_HEAD_DIM, "flipped", gen)
    ind.wk_weights_proj = ProcessedLinear(
        HIDDEN, IDX_HEAD_DIM + IDX_N_HEAD, "flipped", gen
    )
    with torch.no_grad():
        ind.k_norm.weight.copy_(1.0 + 0.3 * torch.randn(IDX_HEAD_DIM, generator=gen))
        ind.k_norm.bias.copy_(0.3 * torch.randn(IDX_HEAD_DIM, generator=gen))
    ind.indexer_op = MagicMock(side_effect=lambda *a: a)
    return ind


def mla_impl(dcp_size=1):
    impl = PallasMLAttentionBackendImpl.__new__(PallasMLAttentionBackendImpl)
    impl.dcp_size = dcp_size
    impl.dcp_interleave_size = 8
    return impl


# ---------------------------------------------------------------------------
# On one TPU chip
# ---------------------------------------------------------------------------

TPU_PAGE = 128  # the default `sal` indexer writer needs a multiple of 128
TPU_PAGES = 10
# Below request 0's context in the step (352 tokens), above request 1's: the
# indexer really selects for one request and keeps every position for the
# other.
TPU_TOPK = 256
TPU_HEADS = 8
TPU_PREFIX = 320  # request 0's tokens cached before the step
TPU_TOKENS = TPU_PREFIX  # the largest chunk (the prefix)


def build_tpu_sparse_mla(
    mla_name: str, idx_name: str, cache_dtype: str = "fp8"
) -> SimpleNamespace:
    """A synthetic sparse MLA layer on one TPU chip: the real wrapper, indexer,
    indexer op (streamidx top-k + K write), MLA layer and backend impl
    (sparse MLA + latent write), with stand-in projections and fp8 caches
    (or, with `cache_dtype="turboquant_4bit_nc"`, TurboQuant caches; the
    caller sets the active TurboQuant preset) in the default layouts, under
    the writers' default modes.

    Ops register under names derived from `mla_name` and `idx_name`, so each
    build in one process needs its own pair."""
    import jax

    from vllm_torchtpu.layers.adapter.attention import (
        PallasMLAttentionBackend,
        VllmTPUDeepseekV32IndexerBackend,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    dev = torch.device("tpu")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("model",))

    op = mla_op.VllmTPUSparseAttnIndexer.__new__(mla_op.VllmTPUSparseAttnIndexer)
    torch.nn.Module.__init__(op)
    op.topk_tokens = TPU_TOPK
    op.dcp_size = 1
    op.topk_op_dcp = None
    op.kv_cache_mode = mla_op._explicit_indexer_kv_cache_mode()
    idx_shape = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(
        TPU_PAGES, TPU_PAGE, 1, IDX_HEAD_DIM
    )
    op.k_cache = SimpleNamespace(
        prefix=idx_name, kv_cache=torch.zeros(idx_shape, dtype=torch.uint8, device=dev)
    )
    op.topk_indices_buffer = torch.full(
        (TPU_TOKENS, TPU_TOPK), -1, dtype=torch.int32, device=dev
    )
    op.topk_op = op._build_streamidx_op()
    op._forward_method = op.forward_oot

    gen = torch.Generator().manual_seed(3)
    layer = mla_op.VllmTPUMLAAttention.__new__(mla_op.VllmTPUMLAAttention)
    torch.nn.Module.__init__(layer)
    impl = mla_impl()
    layer.layer_name = mla_name
    layer.impl = impl
    layer.num_heads = TPU_HEADS
    layer.kv_lora_rank = KV_LORA
    layer.qk_rope_head_dim = ROPE
    layer.v_head_dim = V_DIM
    layer.scale = QK**-0.5
    if cache_dtype == "fp8":
        layer.kv_cache_quantized_dtype = torch.float8_e4m3fn
    else:
        # A TurboQuant layer: its quantized dtype is the uint8 code storage.
        layer.kv_cache_dtype = cache_dtype
        layer.kv_cache_quantized_dtype = torch.uint8
    layer._k_scale_float = 1.0
    layer.W_UK_T = (
        (torch.randn(TPU_HEADS, NOPE, KV_LORA, generator=gen) / math.sqrt(NOPE))
        .bfloat16()
        .to(dev)
    )
    layer.W_UV = (
        (torch.randn(TPU_HEADS, KV_LORA, V_DIM, generator=gen) / math.sqrt(KV_LORA))
        .bfloat16()
        .to(dev)
    )
    specs = PallasMLAttentionBackend.get_sparse_kv_cache_specs(
        TPU_PAGES, TPU_PAGE, KV_LORA + ROPE, cache_dtype
    )
    layer.mla_kv_spec = specs
    layer.kv_cache = tuple(
        torch.zeros(tuple(spec.shape), dtype=spec.torch_dtype, device=dev)
        for spec in specs
    )

    indexer = dsa_indexer(seed=2, topk=TPU_TOPK, buffer_tokens=TPU_TOKENS)
    indexer.indexer_op = op
    w = mla_wrapper(
        heads=TPU_HEADS,
        seed=1,
        indexer=indexer,
        mla_attn=layer,
        topk_buffer=op.topk_indices_buffer,
    )
    w.to(dev)
    with set_vllm_model_wrapper_context(mesh=mesh):
        # Built eagerly, as the runner does: Dynamo cannot trace an op's
        # registration.
        layer.sparse_mla_op = impl._build_sparse_mla_op(layer)
    return SimpleNamespace(
        mesh=mesh,
        wrapper=w,
        mla_layer=layer,
        caches=(*layer.kv_cache, op.k_cache.kv_cache),
        topk_rows=op.topk_indices_buffer,
    )


def tpu_md(names, seq_lens, query_start_loc, request_distribution):
    """The step's metadata; the MLA and indexer caches share one block size
    and page table here, as their KV cache groups do. `names` is the MLA
    layer's and the indexer cache's."""
    dev = torch.device("tpu")

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=dev)

    md = AttentionMetadata(
        input_positions=i32([0]),
        # Four pages per request.
        block_tables=i32([3, 1, 6, 8, 5, 2, 7, 9]),
        seq_lens=i32(seq_lens),
        query_start_loc=i32(query_start_loc),
        request_distribution=i32(request_distribution),
    )
    mla_name, idx_name = names
    return {mla_name: md, idx_name: dataclasses.replace(md)}


def on_tpu(built, program, md, *args):
    """Run `program` under `md` on `build_tpu_sparse_mla`'s layer."""
    from vllm.forward_context import ForwardContext, override_forward_context

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    ctx = ForwardContext(
        no_compile_layers={built.mla_layer.layer_name: built.mla_layer},
        attn_metadata=md,
        slot_mapping={},
    )
    with (
        torch.no_grad(),
        override_forward_context(ctx),
        set_vllm_model_wrapper_context(mesh=built.mesh),
    ):
        return program(*args)


def tpu_compiled(module):
    return torch.compile(module, backend="tpu", fullgraph=True, dynamic=False)


def assert_same_bytes(got, want, what):
    for name, g, w in zip(("nope", "rope", "indexer K"), got, want):
        g8 = g.cpu().view(torch.uint8)
        w8 = w.cpu().view(torch.uint8)
        diff = (g8 != w8).sum().item()
        assert diff == 0, f"{what}: {name} cache differs in {diff} bytes"


def reset_caches(built, saved):
    """Put `saved` (host copies) back into `built`'s caches."""
    for cache, host in zip(built.caches, saved):
        cache.copy_(host.to(cache.device))
