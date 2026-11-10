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
"""The MLA KV write op (`MlaKvWriteOp`, the MTP KV-only pass) and its parts:
every latent it writes must be bit-identical to what the full forward hands
the attention / indexer writers.

* `VllmTPUMultiHeadLatentAttentionWrapper.kv_latents` vs `forward`'s
  `(kv_c_normed, k_pe)` (bf16 and fp8 projections, both rope classes);
* `VllmTPUIndexer.k_packed` vs `VllmTPUIndexer.forward` + the packing in
  `VllmTPUSparseAttnIndexer.forward_oot`;
* `mtp_layer_input` (spec_decode/mtp_kv_only.py) vs the upstream
  `DeepSeekMultiTokenPredictor` -> `DeepSeekMultiTokenPredictorLayer` ->
  `DeepseekV2DecoderLayer` prefix;
* the non-DCP XLA indexer-K append, read back slot by slot;
* the DCP and non-DCP cache writers against the writers' CPU goldens.

Runs on CPU; the projections are stand-ins in the processed runtime layouts.
"""

import contextlib
import dataclasses
import math
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import vllm.model_executor.custom_op as custom_op_mod
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.models.deepseek_v2 as deepseek_v2
import vllm.model_executor.parameter as parameter_mod
from vllm.config import CacheConfig
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache, Indexer

from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.spec_decode import mtp_kv_only

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


def _native_custom_ops():
    """CustomOps built outside a vLLM config dispatch to their native forward."""
    cfg = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["none"])
    return patch.object(custom_op_mod, "get_cached_compilation_config", lambda: cfg)


def _rope(is_neox_style=False, scaling=False):
    with _native_custom_ops():
        if scaling:
            return mla_op.VllmDeepseekScalingRotaryEmbedding(
                head_size=ROPE,
                rotary_dim=ROPE,
                max_position_embeddings=MAX_POS,
                base=10000.0,
                is_neox_style=is_neox_style,
                scaling_factor=4.0,
                dtype=torch.bfloat16,
            )
        return RotaryEmbedding(
            head_size=ROPE,
            rotary_dim=ROPE,
            max_position_embeddings=MAX_POS,
            base=10000.0,
            is_neox_style=is_neox_style,
            dtype=torch.bfloat16,
        )


class _Linear(torch.nn.Module):
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


class _RmsNorm(torch.nn.Module):
    def __init__(self, n, gen):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (1.0 + 0.2 * torch.randn(n, generator=gen)).bfloat16(), requires_grad=False
        )

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (xf * self.weight.float()).to(x.dtype)


def _wrapper(
    *, layout="flipped", seed=0, rope=None, gate_is_fused=False, q_lora=True, heads=4
):
    """A real wrapper with stand-in projections (bypassing the attention
    constructor)."""
    gen = torch.Generator().manual_seed(seed)
    cls = mla_op.VllmTPUMultiHeadLatentAttentionWrapper
    w = cls.__new__(cls)
    torch.nn.Module.__init__(w)
    w.kimi_preprocess = None
    w.q_lora_rank = Q_LORA
    w.kv_lora_rank = KV_LORA
    w.qk_nope_head_dim = NOPE
    w.qk_rope_head_dim = ROPE
    w.qk_head_dim = QK
    w.num_heads = heads
    w.v_head_dim = V_DIM
    w.dcp_q_replicate = False
    w.dcp_group_size = 1
    gate = heads * V_DIM if gate_is_fused else 0
    if q_lora:
        w.fused_qkv_a_proj = _Linear(
            HIDDEN, Q_LORA + KV_LORA + ROPE + gate, layout, gen
        )
        w.q_b_proj = _Linear(Q_LORA, heads * QK, layout, gen)
        w.kv_a_proj_with_mqa = None
        w.q_proj = None
    else:
        w.q_lora_rank = None
        w.fused_qkv_a_proj = None
        w.q_b_proj = None
        w.kv_a_proj_with_mqa = _Linear(HIDDEN, KV_LORA + ROPE, layout, gen)
        w.q_proj = _Linear(HIDDEN, heads * QK, layout, gen)
    w.q_a_layernorm = _RmsNorm(Q_LORA, gen)
    w.kv_a_layernorm = _RmsNorm(KV_LORA, gen)
    w.rotary_emb = rope if rope is not None else _rope()
    w.is_sparse = True
    w.indexer = None
    w.skip_topk = True
    w.topk_indices_buffer = torch.zeros((4096, 8), dtype=torch.int32)
    w.g_proj = None
    w.gate_is_fused = gate_is_fused
    w.o_proj = lambda value: (value, None)
    return w


def _wrapper_latents(w, hidden, positions):
    """What `forward` hands the MLA attention as (kv_c_normed, k_pe)."""
    w.mla_attn = MagicMock(
        return_value=torch.zeros((hidden.shape[0], w.num_heads * V_DIM))
    )
    mla_op.VllmTPUMultiHeadLatentAttentionWrapper.forward(
        w, positions=positions, hidden_states=hidden
    )
    _, kv_c, k_pe = w.mla_attn.call_args.args[:3]
    return kv_c, k_pe


def _inputs(num_tokens, seed=1):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(num_tokens, HIDDEN, generator=gen).bfloat16()
    positions = torch.randperm(MAX_POS, generator=gen)[:num_tokens]
    return hidden, positions


@pytest.mark.parametrize("layout", ["flipped", "fp8"])
@pytest.mark.parametrize("scaling", [False, True], ids=["rope", "deepseek_scaling"])
@pytest.mark.parametrize("num_tokens", [16, 96])
def test_kv_latents_match_the_wrapper(layout, scaling, num_tokens):
    w = _wrapper(layout=layout, rope=_rope(scaling=scaling))
    hidden, positions = _inputs(num_tokens)
    want_kv, want_pe = _wrapper_latents(w, hidden, positions)
    got_kv, got_pe = w.kv_latents(positions, hidden)
    assert torch.equal(got_kv, want_kv)
    assert got_pe.shape == want_pe.shape
    assert torch.equal(got_pe, want_pe)


def test_kv_latents_without_q_lora():
    w = _wrapper(q_lora=False)
    hidden, positions = _inputs(16)
    want_kv, want_pe = _wrapper_latents(w, hidden, positions)
    got_kv, got_pe = w.kv_latents(positions, hidden)
    assert torch.equal(got_kv, want_kv)
    assert torch.equal(got_pe, want_pe)


def test_kv_latents_with_fused_gate():
    w = _wrapper(gate_is_fused=True)
    hidden, positions = _inputs(16)
    want_kv, want_pe = _wrapper_latents(w, hidden, positions)
    got_kv, got_pe = w.kv_latents(positions, hidden)
    assert torch.equal(got_kv, want_kv)
    assert torch.equal(got_pe, want_pe)


def test_kv_latents_positive_control():
    """Different positions must change k_pe: the comparison above can fail."""
    w = _wrapper()
    hidden, positions = _inputs(16)
    want_kv, want_pe = _wrapper_latents(w, hidden, positions)
    _, got_pe = w.kv_latents(positions + 1, hidden)
    assert not torch.equal(got_pe, want_pe)


# ---------------------------------------------------------------------------
# Indexer K
# ---------------------------------------------------------------------------


class _StubIndexerCache(DeepseekV32IndexerCache):
    def __init__(self, head_dim, dtype, prefix, cache_config):
        torch.nn.Module.__init__(self)
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.cache_config = cache_config


@contextlib.contextmanager
def _default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _indexer(seed=0, topk=8, buffer_tokens=16):
    config = SimpleNamespace(
        index_topk=topk,
        index_n_heads=IDX_N_HEAD,
        index_head_dim=IDX_HEAD_DIM,
        qk_rope_head_dim=ROPE,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=MAX_POS),
        attention_config=SimpleNamespace(sparse_mla_prefill_workspace_factor=1),
    )
    with (
        patch.object(deepseek_v2, "DeepseekV32IndexerCache", _StubIndexerCache),
        patch.object(deepseek_v2, "SparseAttnIndexer", MagicMock()),
        patch.object(linear_mod, "get_tensor_model_parallel_rank", lambda: 0),
        patch.object(linear_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        patch.object(parameter_mod, "get_tensor_model_parallel_rank", lambda: 0),
        patch.object(parameter_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        _native_custom_ops(),
        _default_dtype(torch.bfloat16),
    ):
        ind = Indexer(
            vllm_config=vllm_config,
            config=config,
            hidden_size=HIDDEN,
            q_lora_rank=Q_LORA,
            quant_config=None,
            cache_config=CacheConfig(),
            topk_indices_buffer=torch.full(
                (buffer_tokens, topk), -1, dtype=torch.int32
            ),
            prefix="model.layers.78.self_attn.indexer",
        )
        ind = mla_op.VllmTPUIndexer.rebind(ind)
    gen = torch.Generator().manual_seed(seed)
    ind.wq_b = _Linear(Q_LORA, IDX_N_HEAD * IDX_HEAD_DIM, "flipped", gen)
    ind.wk_weights_proj = _Linear(HIDDEN, IDX_HEAD_DIM + IDX_N_HEAD, "flipped", gen)
    with torch.no_grad():
        ind.k_norm.weight.copy_(1.0 + 0.3 * torch.randn(IDX_HEAD_DIM, generator=gen))
        ind.k_norm.bias.copy_(0.3 * torch.randn(IDX_HEAD_DIM, generator=gen))
    ind.indexer_op = MagicMock(side_effect=lambda *a: a)
    return ind


class _Captured(Exception):
    def __init__(self, args):
        super().__init__()
        self.args_ = args


def _packed_by_the_indexer_op(hidden, q_fp8, k, weights):
    """The packed rows `forward_oot` hands the top-k op (and its writer)."""

    def capture(*args):
        raise _Captured(args)

    op = SimpleNamespace(
        k_cache=SimpleNamespace(prefix="idx", kv_cache=torch.zeros(1)),
        topk_op_dcp=None,
        topk_op=capture,
    )
    md = AttentionMetadata(
        input_positions=torch.zeros(1),
        block_tables=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.zeros(1, dtype=torch.int32),
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
    )
    with (
        patch.object(
            mla_op,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"idx": md}),
        ),
        pytest.raises(_Captured) as exc,
    ):
        mla_op.VllmTPUSparseAttnIndexer.forward_oot(op, hidden, q_fp8, k, weights)
    return exc.value.args_[3]


@pytest.mark.parametrize("num_tokens", [16, 96])
@pytest.mark.parametrize("scaling", [False, True], ids=["rope", "deepseek_scaling"])
def test_k_packed_matches_the_indexer(num_tokens, scaling):
    ind = _indexer()
    rope = _rope(scaling=scaling)
    hidden, positions = _inputs(num_tokens)
    gen = torch.Generator().manual_seed(5)
    qr = torch.randn(num_tokens, Q_LORA, generator=gen).bfloat16()
    with torch.no_grad():
        h, q_fp8, k, weights = ind(hidden, qr, positions, rope)
        want = _packed_by_the_indexer_op(h, q_fp8, k, weights)
        got = ind.k_packed(hidden, positions, rope)
    assert got.dtype == want.dtype == torch.uint8
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# The MTP layer prefix
# ---------------------------------------------------------------------------


class _Stop(Exception):
    pass


def _upstream_mtp(vocab=64, hidden=32, seed=0):
    from vllm.model_executor.models.deepseek_mtp import (
        DeepSeekMultiTokenPredictor,
        DeepSeekMultiTokenPredictorLayer,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer

    gen = torch.Generator().manual_seed(seed)

    class _Capture(torch.nn.Module):
        def forward(self, positions, hidden_states, *_args):
            self.seen = hidden_states
            raise _Stop

    class _Block(torch.nn.Module):
        forward = DeepseekV2DecoderLayer.forward

    class _Layer(torch.nn.Module):
        forward = DeepSeekMultiTokenPredictorLayer.forward

    class _Predictor(torch.nn.Module):
        forward = DeepSeekMultiTokenPredictor.forward

    block = _Block()
    block.use_sequence_parallel_moe = False
    block.use_mha = False
    block.input_layernorm = _RmsNorm(hidden, gen)
    block.self_attn = _Capture()
    layer = _Layer()
    layer.enorm = _RmsNorm(hidden, gen)
    layer.hnorm = _RmsNorm(hidden, gen)
    layer.eh_proj = torch.nn.Linear(2 * hidden, hidden, bias=False).bfloat16()
    torch.nn.init.normal_(layer.eh_proj.weight, generator=gen)
    layer.mtp_block = block
    predictor = _Predictor()
    predictor.layers = torch.nn.ModuleDict({"78": layer})
    predictor.mtp_start_layer_idx = 78
    predictor.num_mtp_layers = 1
    predictor.embed_tokens = torch.nn.Embedding(vocab, hidden).bfloat16()
    return SimpleNamespace(model=predictor), block.self_attn


def test_mtp_layer_input_matches_the_upstream_prefix():
    mtp, capture = _upstream_mtp()
    gen = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, 64, (12,), generator=gen)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 7, 8, 9, 10, 0, 5])
    hidden = torch.randn(12, 32, generator=gen).bfloat16()
    with torch.no_grad(), pytest.raises(_Stop):
        mtp.model(input_ids, positions, hidden, None, 0)
    with torch.no_grad():
        layer, got = mtp_kv_only.mtp_layer_input(mtp, input_ids, positions, hidden)
    assert layer is mtp.model.layers["78"]
    assert torch.equal(got, capture.seen)


# ---------------------------------------------------------------------------
# Support check
# ---------------------------------------------------------------------------


class _SelfAttn(torch.nn.Module):
    def __init__(self, mla_attn):
        super().__init__()
        self.mla_attn = mla_attn


def _supported_mtp():
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl

    w = _wrapper()
    w.indexer = _indexer()
    indexer_op = mla_op.VllmTPUSparseAttnIndexer.__new__(
        mla_op.VllmTPUSparseAttnIndexer
    )
    torch.nn.Module.__init__(indexer_op)
    w.indexer.indexer_op = indexer_op
    w.mla_attn = SimpleNamespace(
        impl=PallasMLAttentionBackendImpl.__new__(PallasMLAttentionBackendImpl)
    )
    mtp, _ = _upstream_mtp()
    mtp.model.layers["78"].mtp_block.self_attn = _SelfAttn(w)
    return mtp


def test_kv_only_supported_for_a_dsa_mtp():
    assert mtp_kv_only.kv_only_unsupported_reason(_supported_mtp()) is None


def test_kv_only_supported_for_a_subclass_that_keeps_the_forwards():
    """Pass 1 shares the wrapper's projections, so a subclass that keeps
    `forward` computes the same entries."""
    mtp = _supported_mtp()
    w = mtp.model.layers["78"].mtp_block.self_attn.mla_attn
    w.__class__ = type("_Sub", (type(w),), {})
    assert mtp_kv_only.kv_only_unsupported_reason(mtp) is None


def test_kv_only_cache_layer_names_are_read_without_building_the_op():
    mtp = _supported_mtp()
    wrapper = mtp.model.layers["78"].mtp_block.self_attn.mla_attn
    wrapper.mla_attn.layer_name = "model.layers.78.mtp_block.self_attn.attn"
    wrapper.indexer.indexer_op.k_cache = SimpleNamespace(prefix="mtp.k_cache")
    assert mtp_kv_only.kv_only_cache_layer_names(mtp) == (
        "model.layers.78.mtp_block.self_attn.attn",
        "mtp.k_cache",
    )


@pytest.mark.parametrize(
    "breakage",
    [
        "none",
        "two_layers",
        "plain_wrapper",
        "dense",
        "kimi",
        "wrapper_forward_overridden",
        "indexer_forward_overridden",
        "indexer_op_stub",
        "impl_stub",
        "patched_layer_forward",
        "patched_block_forward",
    ],
)
def test_kv_only_unsupported_reasons(breakage):
    mtp = None if breakage == "none" else _supported_mtp()
    if breakage == "two_layers":
        mtp.model.layers["79"] = mtp.model.layers["78"]
    elif breakage == "plain_wrapper":
        mtp.model.layers["78"].mtp_block.self_attn = _SelfAttn(torch.nn.Identity())
    elif breakage == "dense":
        mtp.model.layers["78"].mtp_block.self_attn.mla_attn.is_sparse = False
    elif breakage == "kimi":
        mtp.model.layers["78"].mtp_block.self_attn.mla_attn.kimi_preprocess = object()
    elif breakage == "wrapper_forward_overridden":
        w = mtp.model.layers["78"].mtp_block.self_attn.mla_attn
        w.__class__ = type("_Sub", (type(w),), {"forward": lambda *a: a})
    elif breakage == "indexer_forward_overridden":
        ind = mtp.model.layers["78"].mtp_block.self_attn.mla_attn.indexer
        ind.__class__ = type("_Sub", (type(ind),), {"forward": lambda *a: a})
    elif breakage == "indexer_op_stub":
        w = mtp.model.layers["78"].mtp_block.self_attn.mla_attn
        w.indexer._modules.pop("indexer_op")
        w.indexer.indexer_op = MagicMock()
    elif breakage == "impl_stub":
        mtp.model.layers["78"].mtp_block.self_attn.mla_attn.mla_attn = SimpleNamespace(
            impl=object()
        )
    elif breakage == "patched_layer_forward":
        layer = mtp.model.layers["78"]
        layer.__class__ = type("_Patched", (type(layer),), {"forward": lambda *a: a})
    elif breakage == "patched_block_forward":
        block = mtp.model.layers["78"].mtp_block
        block.__class__ = type("_Patched", (type(block),), {"forward": lambda *a: a})
    assert mtp_kv_only.kv_only_unsupported_reason(mtp) is not None


# ---------------------------------------------------------------------------
# Non-DCP indexer-K append (XLA writer)
# ---------------------------------------------------------------------------


def test_indexer_k_append_xla_writes_every_row_in_place():
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        KVLayout,
        get_kv_cache_shape,
    )

    page_size, num_pages, width = 16, 8, IDX_HEAD_DIM + 1
    shape = get_kv_cache_shape(num_pages, page_size, width, KVLayout.HEAD_ALONG_SUBLANE)
    cache = jnp.zeros(shape, dtype=jnp.uint8)
    # Two requests: 3 new tokens at positions 5..7, 4 new tokens at 14..17
    # (crossing a page); one padded row.
    seq_lens = jnp.array([8, 18], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 3, 7], dtype=jnp.int32)
    pages = jnp.array([[2, 6], [3, 5]], dtype=jnp.int32).reshape(-1)
    rng = np.random.default_rng(0)
    k_packed = jnp.asarray(rng.integers(1, 255, (8, width), dtype=np.uint8))
    out = np.asarray(
        mla_op.indexer_k_append(
            cache,
            k_packed,
            seq_lens,
            pages,
            cu_q_lens,
            writer="xla",
            kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        )
    )
    kv_packing = shape[2]
    expected = np.zeros_like(out)
    rows = [(0, p) for p in range(5, 8)] + [(1, p) for p in range(14, 18)]
    page_table = np.asarray(pages).reshape(2, -1)
    for t, (r, pos) in enumerate(rows):
        page = page_table[r, pos // page_size]
        slot = pos % page_size
        expected[page, slot // kv_packing, slot % kv_packing, :width] = np.asarray(
            k_packed[t]
        )
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# The cache-write op
# ---------------------------------------------------------------------------

PAGE = 32
PAGES = 16


# Both cache layouts: SPARSECORE packs the fp8 RoPE cache to 64 B/token.
MLA_LAYOUTS = ["tensorcore", "sparsecore"]


def _mla_specs(layout="tensorcore"):
    from vllm_torchtpu.kernels.mla.kv_cache_utils import (
        KVCacheLayout,
        KVCacheType,
        SparseMLAKVCacheSpec,
    )

    cache_layout = KVCacheLayout(layout)
    return (
        SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE, cache_layout, PAGES, PAGE, KV_LORA, 4
        ),
        SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE, cache_layout, PAGES, PAGE, ROPE, 4
        ),
    )


def _write_inputs(layout="tensorcore"):
    """One step's rows plus *different* MLA and indexer metadata, so a swap
    of the two metadata sets cannot go unnoticed."""
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        KVLayout,
        get_kv_cache_shape,
    )

    nope_spec, rope_spec = _mla_specs(layout)
    rng = np.random.default_rng(11)
    num_tokens = 8  # 7 real rows + 1 padded
    kv_c = jnp.asarray(rng.standard_normal((num_tokens, KV_LORA)), jnp.float8_e4m3fn)
    k_pe = jnp.asarray(rng.standard_normal((num_tokens, ROPE)), jnp.float8_e4m3fn)
    width = IDX_HEAD_DIM + 1
    k_packed = jnp.asarray(rng.integers(1, 255, (num_tokens, width), dtype=np.uint8))
    idx_shape = get_kv_cache_shape(PAGES, 16, width, KVLayout.HEAD_ALONG_SUBLANE)
    return dict(
        specs=(nope_spec, rope_spec),
        nope=jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
        rope=jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
        idx_cache=jnp.zeros(idx_shape, jnp.uint8),
        kv_c=kv_c,
        k_pe=k_pe,
        k_packed=k_packed,
        # MLA cache group: page 32, requests at positions 33..35 and 6..9
        # (with DCP 2 x interleave 8, rows 6, 7 and 33..35 sit on rank 0 and
        # rows 8, 9 on rank 1).
        seq_lens=jnp.array([36, 10], jnp.int32),
        block_tables=jnp.array([5, 2, 7, 7, 9, 7, 7, 7], jnp.int32),
        qsl=jnp.array([0, 3, 7], jnp.int32),
        # Indexer cache group: page 16, its own page table.
        idx_seq_lens=jnp.array([36, 10], jnp.int32),
        idx_pages=jnp.array([1, 3, 4, 0, 6, 0, 0, 0], jnp.int32),
        idx_cu=jnp.array([0, 3, 7], jnp.int32),
    )


def _golden(inp, *, dcp):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import KVLayout
    from vllm_torchtpu.kernels.mla import kv_cache_utils

    nope_spec, rope_spec = inp["specs"]
    args = (
        inp["nope"],
        inp["rope"],
        inp["kv_c"],
        inp["k_pe"],
        inp["seq_lens"],
        inp["block_tables"],
        inp["qsl"],
    )
    if dcp:
        nope, rope = kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax(
            *args,
            jnp.int32(0),
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=1,
            interleave_size=8,
        )
    else:
        nope, rope = kv_cache_utils.update_sparse_mla_kv_cache_jax(
            *args, nope_spec=nope_spec, rope_spec=rope_spec
        )
    idx = mla_op.indexer_k_append(
        inp["idx_cache"],
        inp["k_packed"],
        inp["idx_seq_lens"],
        inp["idx_pages"],
        inp["idx_cu"],
        writer="xla",
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    )
    return nope, rope, idx


def _ordered(inp):
    return [
        inp[k]
        for k in (
            "nope",
            "rope",
            "idx_cache",
            "kv_c",
            "k_pe",
            "k_packed",
            "seq_lens",
            "block_tables",
            "qsl",
            "idx_seq_lens",
            "idx_pages",
            "idx_cu",
        )
    ]


def _assert_caches_equal(got, want):
    for g, w in zip(got, want):
        np.testing.assert_array_equal(
            np.asarray(g).view(np.uint8), np.asarray(w).view(np.uint8)
        )
    # Something was written at all.
    assert any(np.asarray(w).view(np.uint8).any() for w in want)


@pytest.mark.parametrize("layout", MLA_LAYOUTS)
def test_write_caches_dcp_matches_the_writers(monkeypatch, layout):
    """The DCP write op on a one-rank DCP mesh: the shard_map wiring, operand
    order and metadata routing, with the CPU goldens of the two writers."""
    from vllm_torchtpu.distributed.dcp import get_or_create_dcp_mesh
    from vllm_torchtpu.kernels.mla import kv_cache_utils

    monkeypatch.setattr(
        mla_op,
        "update_sparse_mla_kv_cache_dcp",
        kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax,
    )
    inp = _write_inputs(layout)
    got = mla_op.mla_kv_write_caches_dcp(
        *_ordered(inp),
        mla_kv_spec=inp["specs"],
        mesh=get_or_create_dcp_mesh(),
        dcp_size=1,
        interleave_size=8,
        kv_cache_mode="xla",
    )
    _assert_caches_equal(got, _golden(inp, dcp=True))


@pytest.mark.parametrize("layout", MLA_LAYOUTS)
def test_write_caches_matches_the_writers(monkeypatch, layout):
    import jax

    from vllm_torchtpu.kernels.mla import kv_cache_utils

    monkeypatch.setattr(
        mla_op,
        "update_sparse_mla_kv_cache",
        kv_cache_utils.update_sparse_mla_kv_cache_jax,
    )
    inp = _write_inputs(layout)
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()[:1]), ("x",))
    got = mla_op.mla_kv_write_caches(
        *_ordered(inp), mla_kv_spec=inp["specs"], mesh=mesh, kv_cache_mode="xla"
    )
    _assert_caches_equal(got, _golden(inp, dcp=False))


def _write_op(quantized=True):
    """A `MlaKvWriteOp` around a real wrapper + indexer, its jax op replaced
    by a recorder."""
    w = _wrapper()
    w.indexer = _indexer()
    w.indexer_rope_emb = _rope()
    op = mla_op.MlaKvWriteOp.__new__(mla_op.MlaKvWriteOp)
    op.wrapper = w
    op.indexer = w.indexer
    op.mla_layer = SimpleNamespace(
        layer_name="mtp.attn",
        kv_cache_quantized_dtype=torch.float8_e4m3fn if quantized else None,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
    )
    op.impl = SimpleNamespace(_get_kv_scales=lambda layer: (None, 0.5, None))
    op.indexer_op = SimpleNamespace(
        k_cache=SimpleNamespace(
            prefix="mtp.indexer.k_cache", kv_cache=torch.zeros(4, dtype=torch.uint8)
        )
    )
    calls = []

    def recorder(*args):
        calls.append(args)
        return args[0] + 1, args[1] + 2, args[2] + 3

    op.jax_op = recorder
    return op, calls


def _md(n):
    return AttentionMetadata(
        input_positions=torch.zeros(1),
        block_tables=torch.full((4,), n, dtype=torch.int32),
        seq_lens=torch.full((2,), n + 1, dtype=torch.int32),
        query_start_loc=torch.full((3,), n + 2, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
    )


@contextlib.contextmanager
def _contexts(kv_cache, md_mla, md_idx):
    with (
        patch.object(
            mla_op,
            "get_attention_context",
            lambda name: (md_mla if name == "mtp.attn" else None, None, kv_cache, None),
        ),
        patch.object(
            mla_op,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"mtp.indexer.k_cache": md_idx}),
        ),
    ):
        yield


@pytest.mark.parametrize("quantized", [True, False])
def test_write_op_operands(quantized):
    from vllm_torchtpu.layers.core.quantization import quantize_kv

    op, calls = _write_op(quantized)
    nope, rope = torch.zeros(3), torch.zeros(3)
    idx_cache = op.indexer_op.k_cache.kv_cache
    md_mla, md_idx = _md(10), _md(20)
    hidden, positions = _inputs(16)
    with torch.no_grad(), _contexts((nope, rope), md_mla, md_idx):
        op(positions, hidden)
        kv_c, k_pe = op.wrapper.kv_latents(positions, hidden)
        k_packed = op.indexer.k_packed(hidden, positions, op.wrapper.indexer_rope_emb)
    if quantized:
        kv_c, _ = quantize_kv(torch.float8_e4m3fn, kv_c, value=None, k_scale=0.5)
        k_pe, _ = quantize_kv(torch.float8_e4m3fn, k_pe, value=None, k_scale=0.5)
    (args,) = calls
    assert args[0] is nope and args[1] is rope and args[2] is idx_cache
    assert torch.equal(
        args[3].view(torch.uint8), kv_c.view(-1, KV_LORA).view(torch.uint8)
    )
    assert torch.equal(args[4].view(torch.uint8), k_pe.view(-1, ROPE).view(torch.uint8))
    assert torch.equal(args[5], k_packed)
    # MLA metadata feeds the MLA caches, the indexer's feeds the indexer cache.
    assert args[6] is md_mla.seq_lens
    assert args[7] is md_mla.block_tables
    assert args[8] is md_mla.query_start_loc
    assert args[9] is md_idx.seq_lens
    assert args[10] is md_idx.block_tables
    assert args[11] is md_idx.query_start_loc
    assert len(args) == 12
    # Written back in place.
    assert torch.equal(nope, torch.ones(3))
    assert torch.equal(rope, torch.full((3,), 2.0))
    assert torch.equal(idx_cache, torch.full((4,), 3, dtype=torch.uint8))


def test_write_op_skips_unbound_caches():
    op, calls = _write_op()
    hidden, positions = _inputs(16)
    with torch.no_grad(), _contexts(torch.zeros(0), None, None):
        op(positions, hidden)
    assert calls == []


@pytest.mark.parametrize("missing", ["mla", "indexer"])
def test_write_op_refuses_bound_caches_without_metadata(missing):
    op, calls = _write_op()
    hidden, positions = _inputs(16)
    md_mla = None if missing == "mla" else _md(1)
    md_idx = None if missing == "indexer" else _md(2)
    with (
        torch.no_grad(),
        _contexts((torch.zeros(3), torch.zeros(3)), md_mla, md_idx),
        pytest.raises(RuntimeError, match="no attention metadata"),
    ):
        op(positions, hidden)
    assert calls == []


# ---------------------------------------------------------------------------
# Differential against the real backend's operands
# ---------------------------------------------------------------------------


def test_write_op_hands_the_writer_what_the_attention_impl_does():
    """`PallasMLAttentionBackendImpl.forward` (the full pass) and the KV-only
    op must hand their cache writers the same fp8 rows and the same metadata
    tensors: same `quantize_kv`, same scale, same views. The sparse op is
    captured on the unsharded path; the DCP path receives the same tensors."""
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl

    heads = 4
    gen = torch.Generator().manual_seed(4)
    op, calls = _write_op(quantized=True)
    impl = PallasMLAttentionBackendImpl.__new__(PallasMLAttentionBackendImpl)
    impl.dcp_size = 1
    captured = []

    def sparse_op(kv_cache, ql, qpe, kv_c, k_pe, topk, seq_lens, bt, qsl, rd):
        captured.append((kv_c, k_pe, seq_lens, bt, qsl))
        return torch.zeros(ql.shape[0], heads, KV_LORA, dtype=torch.bfloat16)

    layer = SimpleNamespace(
        layer_name="mtp.attn",
        kv_cache_quantized_dtype=torch.float8_e4m3fn,
        _k_scale_float=0.37,
        num_heads=heads,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        W_UK_T=torch.randn(heads, NOPE, KV_LORA, generator=gen).bfloat16(),
        W_UV=torch.randn(heads, KV_LORA, V_DIM, generator=gen).bfloat16(),
        sparse_mla_op=sparse_op,
    )
    op.mla_layer = layer
    op.impl = impl
    hidden, positions = _inputs(16)
    md_mla, md_idx = _md(10), _md(20)
    nope, rope = torch.zeros(3), torch.zeros(3)
    with torch.no_grad():
        kv_c, k_pe = _wrapper_latents(op.wrapper, hidden, positions)
        q_nope = torch.randn(16, heads, NOPE, generator=gen).bfloat16()
        q_pe = torch.randn(16, heads, ROPE, generator=gen).bfloat16()
        PallasMLAttentionBackendImpl.forward(
            impl,
            layer,
            (q_nope, q_pe),
            kv_c,
            k_pe,
            (nope, rope),
            md_mla,
            topk_indices=torch.zeros(16, 8, dtype=torch.int32),
        )
        with _contexts((nope, rope), md_mla, md_idx):
            op(positions, hidden)
    ((want_kv, want_pe, w_seq, w_bt, w_qsl),) = captured
    (args,) = calls
    assert args[3].dtype == want_kv.dtype == torch.float8_e4m3fn
    assert args[3].shape == want_kv.shape and args[4].shape == want_pe.shape
    assert torch.equal(args[3].view(torch.uint8), want_kv.view(torch.uint8))
    assert torch.equal(args[4].view(torch.uint8), want_pe.view(torch.uint8))
    assert (args[6], args[7], args[8]) == (w_seq, w_bt, w_qsl)


# ---------------------------------------------------------------------------
# Construction: the op registration and MtpKvOnlyModel
# ---------------------------------------------------------------------------

MTP_ATTN = "model.layers.78.mtp_block.self_attn.attn"


class _RecordedJaxOp:
    """What `pallas.jax_op` / `pcp_streaming_jax_op` were handed."""

    def __init__(self, name, fn, **kwargs):
        self.name, self.fn, self.kwargs = name, fn, kwargs
        self.fake = None
        self.calls = []

    def register_fake(self, fake):
        self.fake = fake

    def __call__(self, *args):
        self.calls.append(args)
        return args[0] + 1, args[1] + 2, args[2] + 3


def _record_jax_ops(monkeypatch):
    built = []

    def jax_op(name, fn, **kwargs):
        built.append(_RecordedJaxOp(name, fn, **kwargs))
        return built[-1]

    monkeypatch.setattr(mla_op.pallas, "jax_op", jax_op)
    monkeypatch.setattr(mla_op, "pcp_streaming_jax_op", jax_op)
    return built


def _op_wrapper(*, dcp_size=1, indexer_dcp_size=None, layout="tensorcore"):
    """A wrapper whose MLA layer, backend impl and indexer op carry what
    `MlaKvWriteOp.__init__` reads."""
    w = _wrapper()
    w.indexer = _indexer()
    w.indexer_rope_emb = _rope()
    indexer_op = mla_op.VllmTPUSparseAttnIndexer.__new__(
        mla_op.VllmTPUSparseAttnIndexer
    )
    torch.nn.Module.__init__(indexer_op)
    indexer_op.dcp_size = dcp_size if indexer_dcp_size is None else indexer_dcp_size
    indexer_op.k_cache = SimpleNamespace(
        prefix="mtp.indexer.k_cache", kv_cache=torch.zeros(4, dtype=torch.uint8)
    )
    w.indexer.indexer_op = indexer_op
    w.mla_attn = SimpleNamespace(
        layer_name=MTP_ATTN,
        mla_kv_spec=_mla_specs(layout),
        kv_cache_quantized_dtype=None,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        impl=SimpleNamespace(
            dcp_size=dcp_size,
            dcp_interleave_size=8,
            _get_kv_scales=lambda layer: (None, None, None),
        ),
    )
    return w


@pytest.mark.parametrize("layout", MLA_LAYOUTS)
def test_write_op_registers_one_donating_op_around_the_writers(monkeypatch, layout):
    """`__init__` registers one op that donates the three caches; the function
    it registers is the unsharded writer pair (checked against the goldens)."""
    import jax

    from vllm_torchtpu.kernels.mla import kv_cache_utils

    built = _record_jax_ops(monkeypatch)
    monkeypatch.setattr(
        mla_op,
        "update_sparse_mla_kv_cache",
        kv_cache_utils.update_sparse_mla_kv_cache_jax,
    )
    monkeypatch.setattr(mla_op.envs, "is_indexer_kv_cache_mode_explicit", lambda: True)
    monkeypatch.setattr(mla_op.envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE", "xla")
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()[:1]), ("x",))
    op = mla_op.MlaKvWriteOp(_op_wrapper(layout=layout), mesh)
    (jax_op,) = built
    assert op.jax_op is jax_op
    assert jax_op.name == "pallas::mla_kv_write_" + MTP_ATTN.replace(".", "_")
    assert jax_op.kwargs == {"donate_argnums": (0, 1, 2)}
    inp = _write_inputs(layout)
    _assert_caches_equal(jax_op.fn(*_ordered(inp)), _golden(inp, dcp=False))
    fake = jax_op.fake(torch.zeros(3), torch.zeros(4), torch.zeros(5), torch.zeros(1))
    assert [t.shape[0] for t in fake] == [3, 4, 5]


def test_write_op_under_dcp_registers_a_dcp_sharded_op(monkeypatch):
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME

    built = _record_jax_ops(monkeypatch)
    mesh = object()
    monkeypatch.setattr(mla_op, "get_or_create_dcp_mesh", lambda axis_name: mesh)
    op = mla_op.MlaKvWriteOp(_op_wrapper(dcp_size=2), mesh=None)
    (jax_op,) = built
    assert op.dcp_size == 2
    assert jax_op.name == "pallas::mla_kv_write_dcp_" + MTP_ATTN.replace(".", "_")
    ax = P(DCP_AXIS_NAME)
    assert jax_op.kwargs == {
        "donate_argnums": (0, 1, 2),
        "mesh": mesh,
        "input_partition_specs": (ax, ax, ax) + (P(),) * 9,
        "output_partition_specs": (ax, ax, ax),
    }
    assert jax_op.fake is not None


def test_write_op_refuses_disagreeing_dcp_sizes(monkeypatch):
    built = _record_jax_ops(monkeypatch)
    with pytest.raises(ValueError, match="disagree on the DCP size: 2 vs 1"):
        mla_op.MlaKvWriteOp(_op_wrapper(dcp_size=2, indexer_dcp_size=1), mesh=None)
    assert built == []


def _kv_only_mtp():
    """A supported MTP draft (as `_supported_mtp`) at the wrapper's hidden
    size, with the MLA layer and caches `MlaKvWriteOp` reads."""
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl

    w = _op_wrapper()
    impl = PallasMLAttentionBackendImpl.__new__(PallasMLAttentionBackendImpl)
    impl.dcp_size = 1
    impl._get_kv_scales = lambda layer: (None, None, None)
    w.mla_attn = SimpleNamespace(
        layer_name="mtp.attn",
        kv_cache_quantized_dtype=None,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        impl=impl,
    )
    mtp, _ = _upstream_mtp(hidden=HIDDEN)
    mtp.model.layers["78"].mtp_block.self_attn = _SelfAttn(w)
    return mtp


def _uncompiled_config():
    return SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )


def test_mtp_kv_only_model_builds_its_write_op_and_writes_every_row(monkeypatch):
    """Built for real (compilation off, the op registration recorded): the
    forward hands the op the latents of the MTP layer's attention input."""
    built = _record_jax_ops(monkeypatch)
    mtp = _kv_only_mtp()
    model = mtp_kv_only.MtpKvOnlyModel(
        vllm_config=_uncompiled_config(),
        prefix="mtp_prefill_kv_only",
        mtp=mtp,
        mesh=None,
    )
    (jax_op,) = built
    assert model.prefix == "mtp_prefill_kv_only"
    assert isinstance(model._write, mla_op.MlaKvWriteOp)
    w = mtp.model.layers["78"].mtp_block.self_attn.mla_attn
    gen = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, 64, (12,), generator=gen)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 7, 8, 9, 10, 0, 5])
    hidden = torch.randn(12, HIDDEN, generator=gen).bfloat16()
    nope, rope = torch.zeros(3), torch.zeros(3)
    with torch.no_grad(), _contexts((nope, rope), _md(10), _md(20)):
        model(input_ids, positions, hidden)
        _, attn_input = mtp_kv_only.mtp_layer_input(mtp, input_ids, positions, hidden)
        kv_c, k_pe = w.kv_latents(positions, attn_input)
        k_packed = w.indexer.k_packed(attn_input, positions, w.indexer_rope_emb)
    (args,) = jax_op.calls
    assert torch.equal(args[3], kv_c.view(-1, KV_LORA))
    assert torch.equal(args[4], k_pe.view(-1, ROPE))
    assert torch.equal(args[5], k_packed)


def test_mtp_kv_only_model_refuses_an_unsupported_draft(monkeypatch):
    built = _record_jax_ops(monkeypatch)
    mtp = _kv_only_mtp()
    mtp.model.layers["78"].mtp_block.self_attn.mla_attn.is_sparse = False
    with pytest.raises(ValueError, match="not sparse"):
        mtp_kv_only.MtpKvOnlyModel(vllm_config=_uncompiled_config(), mtp=mtp, mesh=None)
    assert built == []


# ---------------------------------------------------------------------------
# Two DCP ranks, on a two-device CPU mesh in a subprocess (the CPU lane runs
# this module on one device)
# ---------------------------------------------------------------------------

_TWO_RANK_WORKER_ENV = "MLA_KV_WRITE_TWO_RANK_WORKER"


@pytest.mark.parametrize("layout", MLA_LAYOUTS)
def test_write_caches_dcp_two_ranks_keep_their_own_rows(layout):
    """Each DCP rank writes exactly the rows it owns into its own shard: the
    shard_map specs and the rank (`cp_rank_as_data`) are what the per-rank
    golden says. The indexer half is `_cp_scatter_indexer_k` itself."""
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count=2",
        **{_TWO_RANK_WORKER_ENV: "1"},
    )
    node = f"{__file__}::test_two_rank_worker[{layout}]"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "1 passed" in proc.stdout, proc.stdout[-3000:]


if os.environ.get(_TWO_RANK_WORKER_ENV) == "1":

    @pytest.mark.parametrize("layout", MLA_LAYOUTS)
    def test_two_rank_worker(monkeypatch, layout):
        _two_ranks_keep_their_own_rows(monkeypatch, layout)


def _two_ranks_keep_their_own_rows(monkeypatch, layout):
    import jax

    from vllm_torchtpu.distributed.dcp import DCP_LANE_AXIS_NAME
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME
    from vllm_torchtpu.kernels.mla import kv_cache_utils

    assert len(jax.devices()) >= 2, jax.devices()
    monkeypatch.setattr(
        mla_op,
        "update_sparse_mla_kv_cache_dcp",
        kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax,
    )
    dcp, interleave = 2, 8
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:dcp]).reshape(1, dcp),
        (DCP_LANE_AXIS_NAME, DCP_AXIS_NAME),
    )
    inp = _write_inputs(layout)
    nope_spec, rope_spec = inp["specs"]
    sharded = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(DCP_AXIS_NAME)
    )
    # Global caches: dcp shards of the per-rank spec stacked on dim 0.
    for key, spec in (("nope", nope_spec), ("rope", rope_spec)):
        inp[key] = jax.device_put(
            jnp.zeros((dcp * spec.shape[0], *spec.shape[1:]), spec.jax_dtype), sharded
        )
    idx_shape = inp["idx_cache"].shape
    inp["idx_cache"] = jax.device_put(
        jnp.zeros((dcp * idx_shape[0], *idx_shape[1:]), jnp.uint8), sharded
    )
    got = mla_op.mla_kv_write_caches_dcp(
        *_ordered(inp),
        mla_kv_spec=(nope_spec, rope_spec),
        mesh=mesh,
        dcp_size=dcp,
        interleave_size=interleave,
        kv_cache_mode="xla",
    )
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        _cp_scatter_indexer_k,
    )

    for r in range(dcp):
        want_nope, want_rope = kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax(
            jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
            jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
            inp["kv_c"],
            inp["k_pe"],
            inp["seq_lens"],
            inp["block_tables"],
            inp["qsl"],
            jnp.int32(r),
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=dcp,
            interleave_size=interleave,
        )
        n0, r0 = nope_spec.shape[0], rope_spec.shape[0]
        np.testing.assert_array_equal(
            np.asarray(got[0])[r * n0 : (r + 1) * n0].view(np.uint8),
            np.asarray(want_nope).view(np.uint8),
        )
        np.testing.assert_array_equal(
            np.asarray(got[1])[r * r0 : (r + 1) * r0].view(np.uint8),
            np.asarray(want_rope).view(np.uint8),
        )
    want_idx = _cp_scatter_indexer_k(
        jax.device_put(jnp.zeros_like(inp["idx_cache"]), sharded),
        inp["k_packed"],
        inp["idx_seq_lens"],
        inp["idx_pages"],
        inp["idx_cu"],
        mesh=mesh,
        cp_size=dcp,
        interleave_size=interleave,
        axis_name=DCP_AXIS_NAME,
        writer="xla",
    )
    np.testing.assert_array_equal(np.asarray(got[2]), np.asarray(want_idx))
    # Both ranks got rows (the interleave spreads this step over both).
    for cache in got[:2]:
        halves = np.split(np.asarray(cache).view(np.uint8), dcp)
        assert all(h.any() for h in halves)


# ---------------------------------------------------------------------------
# On TPU: the compiled ops, the default writers and an fp8 cache
# ---------------------------------------------------------------------------

TPU_PAGE = 128  # the default `sal` indexer writer needs a multiple of 128
TPU_PAGES = 8
TPU_TOPK = 512  # above every sequence here: the indexer keeps every position
TPU_HEADS = 8
TPU_TOKENS = 64  # the padded chunk
TPU_ROWS = 8  # pass 2's loop bucket p
TPU_MLA = "model.layers.78.mtp_block.self_attn.mla_attn.mla_attn"
TPU_IDX = "model.layers.78.mtp_block.self_attn.mla_attn.indexer.k_cache"


class _Step0(torch.nn.Module):
    """Draft step 0 as the full pass runs it, up to the attention output: the
    MTP prefix, then the wrapper's forward (indexer, sparse MLA)."""

    def __init__(self, mtp):
        super().__init__()
        self.predictor = mtp.model
        self.mtp = mtp

    def forward(self, input_ids, positions, hidden_states):
        layer, attn_input = mtp_kv_only.mtp_layer_input(
            self.mtp, input_ids, positions, hidden_states
        )
        return layer.mtp_block.self_attn.mla_attn(positions, attn_input)


def _tpu_dsa():
    """A synthetic single-layer DSA MTP draft on one TPU chip: the real
    wrapper, indexer, indexer op (streamidx top-k + K write), MLA layer and
    backend impl (sparse MLA + latent write), with stand-in projections and
    fp8 caches in the default layouts, under the writers' default modes."""
    import jax

    from vllm_torchtpu.layers.adapter.attention import (
        PallasMLAttentionBackend,
        PallasMLAttentionBackendImpl,
        VllmTPUDeepseekV32IndexerBackend,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    dev = torch.device("tpu")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("model",))
    w = _wrapper(heads=TPU_HEADS, seed=1)
    w.skip_topk = False
    w.indexer = _indexer(seed=2, topk=TPU_TOPK, buffer_tokens=TPU_TOKENS)
    w.indexer_rope_emb = _rope()

    op = mla_op.VllmTPUSparseAttnIndexer.__new__(mla_op.VllmTPUSparseAttnIndexer)
    torch.nn.Module.__init__(op)
    op.topk_tokens = TPU_TOPK
    op.dcp_size = 1
    op.topk_op_dcp = None
    idx_shape = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(
        TPU_PAGES, TPU_PAGE, 1, IDX_HEAD_DIM
    )
    op.k_cache = SimpleNamespace(
        prefix=TPU_IDX, kv_cache=torch.zeros(idx_shape, dtype=torch.uint8, device=dev)
    )
    op.topk_indices_buffer = torch.full(
        (TPU_TOKENS, TPU_TOPK), -1, dtype=torch.int32, device=dev
    )
    op.topk_op = op._build_streamidx_op()
    op._forward_method = op.forward_oot
    w.indexer.indexer_op = op

    gen = torch.Generator().manual_seed(3)
    impl = PallasMLAttentionBackendImpl.__new__(PallasMLAttentionBackendImpl)
    impl.dcp_size = 1
    layer = mla_op.VllmTPUMLAAttention.__new__(mla_op.VllmTPUMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = TPU_MLA
    layer.impl = impl
    layer.num_heads = TPU_HEADS
    layer.kv_lora_rank = KV_LORA
    layer.qk_rope_head_dim = ROPE
    layer.v_head_dim = V_DIM
    layer.scale = QK**-0.5
    layer.kv_cache_quantized_dtype = torch.float8_e4m3fn
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
        TPU_PAGES, TPU_PAGE, KV_LORA + ROPE, "fp8"
    )
    layer.mla_kv_spec = specs
    layer.kv_cache = tuple(
        torch.zeros(tuple(spec.shape), dtype=spec.torch_dtype, device=dev)
        for spec in specs
    )
    w.mla_attn = layer

    mtp, _ = _upstream_mtp(hidden=HIDDEN)
    mtp.model.layers["78"].mtp_block.self_attn = _SelfAttn(w)
    mtp.model.to(dev)
    w.topk_indices_buffer = w.topk_indices_buffer.to(dev)
    assert mtp_kv_only.kv_only_unsupported_reason(mtp) is None
    with set_vllm_model_wrapper_context(mesh=mesh):
        # Built eagerly, as the runner does: Dynamo cannot trace an op's
        # registration.
        layer.sparse_mla_op = impl._build_sparse_mla_op(layer)
        kv_only = mtp_kv_only.MtpKvOnlyModel(
            vllm_config=SimpleNamespace(
                compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
            ),
            prefix="mtp_prefill_kv_only",
            mtp=mtp,
            mesh=mesh,
        )
    caches = (*layer.kv_cache, op.k_cache.kv_cache)
    return SimpleNamespace(
        mesh=mesh, step0=_Step0(mtp), kv_only=kv_only, caches=caches, mla_layer=layer
    )


def _tpu_md(seq_lens, query_start_loc, request_distribution):
    """The step's metadata; the MLA and indexer caches share one block size
    and page table here, as their KV cache groups do."""
    dev = torch.device("tpu")

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=dev)

    md = AttentionMetadata(
        input_positions=i32([0]),
        # Two pages per request; each request fits in its first.
        block_tables=i32([3, 1, 5, 2]),
        seq_lens=i32(seq_lens),
        query_start_loc=i32(query_start_loc),
        request_distribution=i32(request_distribution),
    )
    return {TPU_MLA: md, TPU_IDX: dataclasses.replace(md)}


def _on_tpu(dsa, program, md, *args):
    from vllm.forward_context import ForwardContext, override_forward_context

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    ctx = ForwardContext(
        no_compile_layers={TPU_MLA: dsa.mla_layer},
        attn_metadata=md,
        slot_mapping={},
    )
    with (
        torch.no_grad(),
        override_forward_context(ctx),
        set_vllm_model_wrapper_context(mesh=dsa.mesh),
    ):
        return program(*args)


def _compiled(module):
    return torch.compile(module, backend="tpu", fullgraph=True, dynamic=False)


@pytest.fixture(scope="module")
def tpu_dsa():
    """One build per module: the ops register under fixed names. Request 0
    has 16 tokens cached before the step (written here by the full pass);
    the step is a 64-token chunk: request 0's next 32 tokens, request 1's
    first 24, 8 padded rows."""
    torch._dynamo.reset()
    dsa = _tpu_dsa()
    dsa.step0_c = _compiled(dsa.step0)
    dsa.kv_only_c = _compiled(dsa.kv_only)
    gen = torch.Generator().manual_seed(9)
    dev = torch.device("tpu")

    def tokens(positions):
        n = len(positions)
        return (
            torch.randint(0, 64, (n,), generator=gen, dtype=torch.int32).to(dev),
            torch.tensor(positions, dtype=torch.int32, device=dev),
            torch.randn(n, HIDDEN, generator=gen).bfloat16().to(dev),
        )

    prefix = tokens(list(range(16)))
    _on_tpu(dsa, dsa.step0_c, _tpu_md([16, 0], [0, 16, 16], [0, 0, 1]), *prefix)
    # Host copies: a deferred on-device clone can lose its source to the
    # next op that donates the cache.
    dsa.prefix_caches = [c.cpu() for c in dsa.caches]
    dsa.step = tokens(list(range(16, 48)) + list(range(24)) + [0] * 8)
    dsa.step_md = lambda: _tpu_md([48, 24], [0, 32, 56], [0, 0, 2])
    # Pass 2's rows: each request's last row, padded to p with row 0.
    dsa.last_rows = torch.tensor([31, 55] + [0] * 6, dtype=torch.int64, device=dev)
    dsa.rows_md = lambda: _tpu_md([48, 24], [0, 1, 2], [2, 2, 2])
    return dsa


def _reset(dsa):
    for cache, saved in zip(dsa.caches, dsa.prefix_caches):
        cache.copy_(saved.to(cache.device))


def _assert_same_bytes(got, want, what):
    for name, g, w in zip(("nope", "rope", "indexer K"), got, want):
        g8 = g.cpu().view(torch.uint8)
        w8 = w.cpu().view(torch.uint8)
        diff = (g8 != w8).sum().item()
        assert diff == 0, f"{what}: {name} cache differs in {diff} bytes"


def _single_pass(dsa):
    _reset(dsa)
    out = _on_tpu(dsa, dsa.step0_c, dsa.step_md(), *dsa.step)
    return out.cpu(), [c.cpu() for c in dsa.caches]


@pytest.mark.tpu_test
def test_kv_only_pass_writes_the_full_pass_cache_entries_on_tpu(tpu_dsa):
    """`MtpKvOnlyModel`, compiled for TPU with the default `sal` indexer
    writer and an fp8 latent cache, leaves the same MLA and indexer-K caches
    as the full step-0 forward over the same rows."""
    _, want = _single_pass(tpu_dsa)
    assert any(not torch.equal(w, p) for w, p in zip(want, tpu_dsa.prefix_caches)), (
        "the step wrote nothing"
    )
    _reset(tpu_dsa)
    _on_tpu(tpu_dsa, tpu_dsa.kv_only_c, tpu_dsa.step_md(), *tpu_dsa.step)
    _assert_same_bytes(tpu_dsa.caches, want, "KV-only pass vs full pass")


@pytest.mark.tpu_test
def test_two_pass_step0_matches_the_single_pass_on_tpu(tpu_dsa):
    """Pass 1 over every row, then the full forward on one row per request
    with the narrowed metadata: same caches as the single pass, and each
    request's last row gets the same attention output."""
    want_out, want_caches = _single_pass(tpu_dsa)
    _reset(tpu_dsa)
    _on_tpu(tpu_dsa, tpu_dsa.kv_only_c, tpu_dsa.step_md(), *tpu_dsa.step)
    ids, positions, hidden = tpu_dsa.step
    rows = tpu_dsa.last_rows
    out = _on_tpu(
        tpu_dsa,
        tpu_dsa.step0_c,
        tpu_dsa.rows_md(),
        ids.index_select(0, rows),
        positions.index_select(0, rows),
        hidden.index_select(0, rows),
    )
    _assert_same_bytes(tpu_dsa.caches, want_caches, "two-pass vs single pass")
    torch.testing.assert_close(out.cpu()[:2], want_out[[31, 55]], rtol=1e-2, atol=1e-2)
