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
"""Correctness tests for `VllmTPUIndexer` (DeepSeek-V3.2 / GLM sparse indexer).

`VllmTPUIndexer.forward` does not select tokens itself: it prepares the
`(q_fp8, k, weights)` triple that the TPU `streamindex_topk` kernel consumes,
where the kernel computes

    score[t, s] = sum_h relu(q_fp8[t, h] . k[s]) * weights[t, h]

and returns the top-k `s` per query token `t`. Everything the layer does --
the fused wk/weights_proj GEMM, the k LayerNorm, RoPE on the leading
`rope_dim` slice, per-head fp8 quantization of q, and folding the q scale
(together with `softmax_scale` and `n_head ** -0.5`) into `weights` -- only
matters insofar as that score matches the unfused, unquantized reference.

So these tests mock out the kernel wrapper, capture the triple, and check it
against a plain-torch reference built from the same parameters. They run on
CPU; the kernel itself is covered by
`tests/kernels/test_dsv4_streamindex_topk.py`.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.model_executor.custom_op as custom_op_mod
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.models.deepseek_v2 as deepseek_v2
import vllm.model_executor.parameter as parameter_mod
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models.deepseek_v2 import (DeepseekV32IndexerCache,
                                                    Indexer)

from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

# Small stand-in for the DeepSeek-V3.2 indexer config. `head_dim` is kept at
# the real 128 because the streamindex_topk kernel requires
# quant_block_size == head_dim, and fp8 block quantization is per 128 lanes.
HEAD_DIM = 128
N_HEAD = 4
ROPE_DIM = 64
Q_LORA_RANK = 32
HIDDEN_SIZE = 64
TOPK = 8
MAX_MODEL_LEN = 256
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


class _StubIndexerCache(DeepseekV32IndexerCache):
    """Stands in for `DeepseekV32IndexerCache` (needs a real KV cache config).

    Subclasses the real cache so `VllmTPUIndexer.rebind` can retype it --
    `__class__` assignment requires a compatible instance layout -- but skips
    its `__init__`, which registers into `static_forward_context`.
    """

    def __init__(self, head_dim, dtype, prefix, cache_config):
        torch.nn.Module.__init__(self)
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.cache_config = cache_config


class _RecordingIndexerOp:
    """Stands in for `VllmTPUSparseAttnIndexer` (builds a Pallas op eagerly)."""

    SENTINEL = object()

    def __init__(self, *init_args):
        self.init_args = init_args
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.SENTINEL

    @property
    def last_call(self):
        """The captured `(hidden_states, q_fp8, k, weights)` of the last call."""
        assert self.calls, "indexer_op was never called"
        return self.calls[-1]


@contextlib.contextmanager
def _default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _build_indexer(dtype=torch.float32, n_head=N_HEAD, buffer_tokens=32):
    """A `VllmTPUIndexer` with the TPU-only collaborators stubbed out.

    The linear layers query the TP group at construction time, so TP rank/size
    are pinned to a single rank instead of standing up a distributed group.
    """
    config = SimpleNamespace(
        index_topk=TOPK,
        index_n_heads=n_head,
        index_head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM,
    )
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(
        max_model_len=MAX_MODEL_LEN))
    topk_indices_buffer = torch.full((buffer_tokens, TOPK),
                                     -1,
                                     dtype=torch.int32)

    # `VllmTPUIndexer` has no `__init__`: it is a retype of an in-tree `Indexer`
    # that `DeepseekV2MLAAttention` has already built. So build the in-tree
    # layer -- stubbing the collaborators where *it* looks them up -- and then
    # rebind, exactly as `VllmTPUMultiHeadLatentAttentionWrapper` does.
    with patch.object(deepseek_v2, "DeepseekV32IndexerCache",
                      _StubIndexerCache), \
         patch.object(deepseek_v2, "SparseAttnIndexer",
                      _RecordingIndexerOp), \
         patch.object(linear_mod, "get_tensor_model_parallel_rank",
                      lambda: 0), \
         patch.object(linear_mod, "get_tensor_model_parallel_world_size",
                      lambda: 1), \
         patch.object(parameter_mod, "get_tensor_model_parallel_rank",
                      lambda: 0), \
         patch.object(parameter_mod, "get_tensor_model_parallel_world_size",
                      lambda: 1), \
         _default_dtype(dtype):
        indexer = Indexer(
            vllm_config=vllm_config,
            config=config,
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            quant_config=None,
            cache_config=None,
            topk_indices_buffer=topk_indices_buffer,
            prefix="model.layers.0.self_attn.indexer",
        )
        return mla_attention_op.VllmTPUIndexer.rebind(indexer)


def _randomize(layer, seed=0):
    """Give every parameter a non-degenerate value.

    Freshly created weights are uninitialized and the LayerNorm defaults to
    weight=1/bias=0, which would hide a dropped `k_norm`.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        layer.wq_b.weight.normal_(0.0, 0.1)
        layer.wk_weights_proj.weight.normal_(0.0, 0.1)
        layer.k_norm.weight.normal_(1.0, 0.05)
        layer.k_norm.bias.normal_(0.0, 0.05)
    return layer


def _make_rope(dtype=torch.float32):
    """A real NeoX `RotaryEmbedding`, as `get_rope` would build for the indexer.

    `CustomOp.__init__` resolves its forward at construction time and needs a
    current vLLM config for that; a bare CompilationConfig is enough and binds
    `forward_native` (the TPU dispatch target is `forward_native` too).
    """
    compilation_config = CompilationConfig(mode=CompilationMode.NONE,
                                           custom_ops=["none"])
    with patch.object(custom_op_mod, "get_cached_compilation_config",
                      lambda: compilation_config):
        return RotaryEmbedding(
            head_size=ROPE_DIM,
            rotary_dim=ROPE_DIM,
            max_position_embeddings=MAX_MODEL_LEN,
            base=10000.0,
            is_neox_style=True,
            dtype=dtype,
        )


def _inputs(num_tokens, dtype=torch.float32, seed=1):
    torch.manual_seed(seed)
    hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, dtype=dtype)
    qr = torch.randn(num_tokens, Q_LORA_RANK, dtype=dtype)
    positions = torch.arange(num_tokens)
    return hidden_states, qr, positions


def _reference(layer, hidden_states, qr, positions, rope):
    """Unfused, unquantized indexer math, straight from the parameters.

    Mirrors upstream `Indexer.forward` before the fp8 step: q from wq_b, k from
    the wk half of the fused GEMM followed by k_norm, per-head raw weights from
    the weights_proj half, RoPE over the leading `rope_dim` of both q and k.
    """
    num_tokens = hidden_states.shape[0]
    n_head = layer.n_head
    with torch.no_grad():
        fused = layer.wk_weights_proj.weight
        wk, w_proj = fused[:HEAD_DIM], fused[HEAD_DIM:]

        q = (qr @ layer.wq_b.weight.T).view(num_tokens, n_head, HEAD_DIM)
        k = layer.k_norm(hidden_states @ wk.T)
        weights = hidden_states @ w_proj.T

        q_pe, q_nope = q[..., :ROPE_DIM], q[..., ROPE_DIM:]
        k_pe, k_nope = k[..., :ROPE_DIM], k[..., ROPE_DIM:]
        q_pe, k_pe = rope.forward_native(positions, q_pe.clone(),
                                         k_pe.clone().unsqueeze(1))
        q = torch.cat([q_pe.view(num_tokens, n_head, ROPE_DIM), q_nope],
                      dim=-1)
        k = torch.cat([k_pe.view(num_tokens, ROPE_DIM), k_nope], dim=-1)
    return q, k, weights


def _indexer_scores(q, k, weights):
    """The score the `streamindex_topk` kernel computes from the triple."""
    per_head = torch.einsum("thd,sd->ths", q.float(), k.float()).clamp(min=0)
    return (per_head * weights.float().unsqueeze(-1)).sum(dim=1)


def _assert_relclose(actual, expected, tol, msg=""):
    """Max deviation, measured against the magnitude of `expected`.

    The reference recomputes each projection from the parameters, so it is
    close to but not bit-identical with the layer (the layer LayerNorms a slice
    of the fused GEMM output, which reassociates by ~1 ULP). Relative-to-scale
    keeps the near-zero entries from dominating.
    """
    actual, expected = actual.float(), expected.float()
    deviation = (actual - expected).abs().max()
    scale = expected.abs().max()
    assert deviation <= tol * scale, (
        f"{msg} max deviation {deviation:.3e} > {tol:g} * {scale:.3e}")


def _run(layer, rope, num_tokens=6, dtype=torch.float32, seed=1):
    hidden_states, qr, positions = _inputs(num_tokens, dtype, seed)
    with torch.no_grad():
        out = layer(hidden_states, qr, positions, rope)
    return out, (hidden_states, qr, positions)


def test_construction_wires_kernel_contract():
    """Cache layout and op arguments the streamindex_topk kernel assumes."""
    layer = _build_indexer()

    # The kernel reads one fp8 value byte per lane plus a single trailing
    # e8m0 scale byte per quant block, so the cache row is head_dim + 1.
    assert layer.quant_block_size == HEAD_DIM  # kernel limitation
    assert layer.k_cache.head_dim == HEAD_DIM + 1
    assert layer.k_cache.dtype == torch.uint8
    assert layer.k_cache.prefix == "model.layers.0.self_attn.indexer.k_cache"
    assert layer.scale_fmt == "ue8m0"
    assert layer.softmax_scale == pytest.approx(HEAD_DIM**-0.5)

    (k_cache, quant_block_size, scale_fmt, topk_tokens, head_dim,
     max_model_len, max_total_seq_len,
     topk_indices_buffer) = layer.indexer_op.init_args
    assert k_cache is layer.k_cache
    assert quant_block_size == HEAD_DIM
    assert scale_fmt == "ue8m0"
    assert topk_tokens == TOPK
    assert head_dim == HEAD_DIM
    assert max_model_len == MAX_MODEL_LEN
    assert max_total_seq_len == MAX_MODEL_LEN * 40
    assert topk_indices_buffer is layer.topk_indices_buffer


def test_projection_shapes():
    """wq_b is replicated per-head; wk and weights_proj share one GEMM."""
    layer = _build_indexer()
    assert layer.wq_b.weight.shape == (N_HEAD * HEAD_DIM, Q_LORA_RANK)
    assert layer.wk_weights_proj.weight.shape == (HEAD_DIM + N_HEAD,
                                                  HIDDEN_SIZE)
    assert layer.wk_weights_proj.output_sizes == [HEAD_DIM, N_HEAD]
    # No TP shard: the indexer heads are replicated on every rank.
    assert layer.wk_weights_proj.tp_size == 1


def test_fused_gemm_shard_order_matches_forward_split():
    """Checkpoint shard 0/1 must land where `forward` slices k and weights.

    `forward` takes `kw[:, :head_dim]` as k and `kw[:, head_dim:]` as the
    per-head weights; if the merged loader ordered the shards the other way
    round the layer would silently score with transposed tensors.
    """
    layer = _build_indexer()
    rope = _make_rope()
    torch.manual_seed(0)
    wk_ckpt = torch.randn(HEAD_DIM, HIDDEN_SIZE) * 0.1
    w_proj_ckpt = torch.randn(N_HEAD, HIDDEN_SIZE) * 0.1

    param = layer.wk_weights_proj.weight
    param.weight_loader(param, wk_ckpt, 0)
    param.weight_loader(param, w_proj_ckpt, 1)
    torch.testing.assert_close(param.data[:HEAD_DIM], wk_ckpt, rtol=0, atol=0)
    torch.testing.assert_close(param.data[HEAD_DIM:],
                               w_proj_ckpt,
                               rtol=0,
                               atol=0)

    with torch.no_grad():  # not _randomize: it would clobber the loaded shards
        layer.wq_b.weight.normal_(0.0, 0.1)
        layer.k_norm.weight.normal_(1.0, 0.05)
        layer.k_norm.bias.normal_(0.0, 0.05)

    _, (hidden_states, qr, positions) = _run(layer, rope)
    _, _, k, weights = layer.indexer_op.last_call

    # k comes from the wk shard alone (through k_norm and RoPE)...
    k_ckpt = layer.k_norm(hidden_states @ wk_ckpt.T)
    k_pe = rope.forward_native(positions, k_ckpt[..., :ROPE_DIM].clone(),
                               None)[0]
    _assert_relclose(k, torch.cat([k_pe, k_ckpt[..., ROPE_DIM:]], dim=-1),
                     1e-5, "k")

    # ...and the head weights from the weights_proj shard alone, times the
    # per-(token, head) fold.
    q_ref, _, _ = _reference(layer, hidden_states, qr, positions, rope)
    q_scale = q_ref.float().abs().amax(dim=-1) / FP8_MAX
    fold = layer.softmax_scale * N_HEAD**-0.5
    expected = (hidden_states @ w_proj_ckpt.T) * q_scale * fold
    _assert_relclose(weights, expected, 1e-5, "weights")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_emits_expected_triple(dtype):
    """The captured triple matches the reference term by term."""
    layer = _randomize(_build_indexer(dtype))
    rope = _make_rope(dtype)
    num_tokens = 6

    out, (hidden_states, qr, positions) = _run(layer,
                                               rope,
                                               num_tokens=num_tokens,
                                               dtype=dtype)

    assert out is _RecordingIndexerOp.SENTINEL
    captured_hidden, q_fp8, k, weights = layer.indexer_op.last_call
    assert captured_hidden is hidden_states
    assert q_fp8.shape == (num_tokens, N_HEAD, HEAD_DIM)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert k.shape == (num_tokens, HEAD_DIM)
    assert k.dtype == dtype
    assert weights.shape == (num_tokens, N_HEAD)
    # The kernel wants the weights in the activation dtype, not the fp32 the
    # scale fold is computed in.
    assert weights.dtype == dtype

    q_ref, k_ref, weights_ref = _reference(layer, hidden_states, qr, positions,
                                           rope)
    # In bf16 a 1-ULP reassociation is ~0.4%; fp8 e4m3 error would be ~10x
    # that, so these still separate "kept in the activation dtype" from
    # "silently quantized".
    tol = 1e-5 if dtype == torch.float32 else 5e-3

    # k is handed over unquantized: the fp8 conversion is fused into the cache
    # insertion inside the op, so nothing may perturb it here.
    _assert_relclose(k, k_ref, tol, "k")

    # q is block-quantized per (token, head) -- one scale per head row, and a
    # plain fp32 scale (NOT ue8m0): a power-of-two scale would round amax/448
    # up to the next octave, i.e. up to 2x off, far outside this tolerance.
    expected_scale = q_ref.float().abs().amax(dim=-1, keepdim=True) / FP8_MAX
    _assert_relclose(q_fp8.float() * expected_scale, q_ref, 0.07, "q")

    # The q scale is folded into the weights along with the softmax scale and
    # the 1/sqrt(n_head) head-average factor.
    fold = layer.softmax_scale * N_HEAD**-0.5
    expected_weights = weights_ref.float() * expected_scale.squeeze(-1) * fold
    _assert_relclose(weights, expected_weights.to(dtype), tol, "weights")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scale_folding_is_algebraically_exact(dtype):
    """Folding the q scale into the weights must not change the score.

    `relu(q_fp8 . k) * (w * q_scale)` and `relu((q_fp8 * q_scale) . k) * w`
    differ only by where the positive scalar is multiplied in, so the two
    scores must agree to float round-off -- this is what makes it legal to
    hand the kernel unscaled fp8 queries.
    """
    layer = _randomize(_build_indexer(dtype))
    rope = _make_rope(dtype)

    _, (hidden_states, qr, positions) = _run(layer, rope, 6, dtype)
    _, q_fp8, k, weights = layer.indexer_op.last_call
    q_ref, _, weights_ref = _reference(layer, hidden_states, qr, positions,
                                       rope)

    q_scale = q_ref.float().abs().amax(dim=-1, keepdim=True) / FP8_MAX
    fold = layer.softmax_scale * N_HEAD**-0.5

    folded = _indexer_scores(q_fp8.float(), k, weights)
    dequantized = _indexer_scores(q_fp8.float() * q_scale, k,
                                  weights_ref.float() * fold)

    scale = dequantized.abs().max()
    # bf16 weights carry ~2^-8 relative rounding on the folded product.
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    assert (folded - dequantized).abs().max() <= tol * scale


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scores_track_unquantized_reference(dtype):
    """End to end: the kernel's score is the reference score, up to fp8.

    A wrong RoPE slice, a dropped k_norm, a missing softmax scale or a
    transposed head axis all show up here as a score mismatch, and -- what the
    indexer is actually judged on -- as a worse token selection.
    """
    layer = _randomize(_build_indexer(dtype))
    rope = _make_rope(dtype)
    num_tokens = 24

    _, (hidden_states, qr, positions) = _run(layer, rope, num_tokens, dtype)
    _, q_fp8, k, weights = layer.indexer_op.last_call
    q_ref, k_ref, weights_ref = _reference(layer, hidden_states, qr, positions,
                                           rope)

    fold = layer.softmax_scale * N_HEAD**-0.5
    actual = _indexer_scores(q_fp8.float(), k, weights)
    ideal = _indexer_scores(q_ref, k_ref, weights_ref.float() * fold)
    scale = ideal.abs().max()

    # fp8 e4m3 keeps 3 mantissa bits, so a 128-term dot product lands within a
    # few percent; anything larger is a structural error, not rounding.
    assert (actual - ideal).abs().max() <= 0.08 * scale

    # Selection quality: the tokens picked from the emitted triple are as good
    # as the exact top-k under the reference scores.
    picked = actual.topk(TOPK, dim=-1).indices
    best_scores = ideal.topk(TOPK, dim=-1).values
    picked_scores = ideal.gather(1, picked)
    regret = (best_scores.min(dim=-1, keepdim=True).values -
              picked_scores).clamp(min=0)
    assert regret.max() <= 0.05 * scale
    assert picked_scores.sum() >= 0.99 * best_scores.sum()


def test_rope_receives_pe_slices_and_tolerates_extra_leading_dims():
    """RoPE is applied to the leading rope_dim of q and to an MQA k of width 1.

    The stub returns `[1, ...]`-shaped tensors, which is what the NeoX kernel
    can produce under compilation; the layer must reshape them back and leave
    the nope halves untouched.
    """
    layer = _randomize(_build_indexer())
    num_tokens = 5
    seen = {}

    def stub_rope(positions, q_pe, k_pe):
        seen["positions"] = positions
        seen["q_pe"] = q_pe.clone()
        seen["k_pe"] = k_pe.clone()
        # Distinct, invertible marks so misrouted slices cannot pass.
        return (q_pe * 3.0).unsqueeze(0), (k_pe * 5.0).unsqueeze(0)

    _, (hidden_states, qr, positions) = _run(layer, stub_rope, num_tokens)
    _, q_fp8, k, weights = layer.indexer_op.last_call

    assert seen["positions"] is positions
    assert seen["q_pe"].shape == (num_tokens, N_HEAD, ROPE_DIM)
    # k is MQA: a single head, unsqueezed for the shared RoPE call.
    assert seen["k_pe"].shape == (num_tokens, 1, ROPE_DIM)

    with torch.no_grad():
        fused = layer.wk_weights_proj.weight
        q_pre = (qr @ layer.wq_b.weight.T).view(num_tokens, N_HEAD, HEAD_DIM)
        k_pre = layer.k_norm(hidden_states @ fused[:HEAD_DIM].T)

    torch.testing.assert_close(seen["q_pe"], q_pre[..., :ROPE_DIM])
    torch.testing.assert_close(seen["k_pe"],
                               k_pre[..., :ROPE_DIM].unsqueeze(1))

    # Rotated halves land first, unrotated halves pass through untouched.
    torch.testing.assert_close(k[:, :ROPE_DIM], k_pre[:, :ROPE_DIM] * 5.0)
    torch.testing.assert_close(k[:, ROPE_DIM:], k_pre[:, ROPE_DIM:])

    q_expected = torch.cat(
        [q_pre[..., :ROPE_DIM] * 3.0, q_pre[..., ROPE_DIM:]], dim=-1)
    q_scale = q_expected.abs().amax(dim=-1, keepdim=True) / FP8_MAX
    torch.testing.assert_close(q_fp8.float() * q_scale,
                               q_expected,
                               rtol=0.07,
                               atol=1e-5 * q_scale.max().item())
    assert weights.shape == (num_tokens, N_HEAD)


def test_decode_single_token():
    """The decode shape (one token) keeps the per-head layout."""
    layer = _randomize(_build_indexer())
    rope = _make_rope()

    _, (hidden_states, qr, positions) = _run(layer, rope, num_tokens=1)
    _, q_fp8, k, weights = layer.indexer_op.last_call

    assert q_fp8.shape == (1, N_HEAD, HEAD_DIM)
    assert k.shape == (1, HEAD_DIM)
    assert weights.shape == (1, N_HEAD)

    q_ref, k_ref, weights_ref = _reference(layer, hidden_states, qr, positions,
                                           rope)
    _assert_relclose(k, k_ref, 1e-5, "k")
    fold = layer.softmax_scale * N_HEAD**-0.5
    q_scale = q_ref.float().abs().amax(dim=-1) / FP8_MAX
    _assert_relclose(weights,
                     weights_ref.float() * q_scale * fold, 1e-5, "weights")


def test_no_nans_when_a_query_head_is_all_zero():
    """A zero q row must quantize to zeros, not 0/0 -- the scale is zero there."""
    layer = _randomize(_build_indexer())
    rope = _make_rope()
    num_tokens = 4
    hidden_states, qr, positions = _inputs(num_tokens)
    with torch.no_grad():
        qr[1].zero_()  # wq_b has no bias, so this q row is exactly zero
        layer(hidden_states, qr, positions, rope)

    _, q_fp8, k, weights = layer.indexer_op.last_call
    assert not torch.isnan(q_fp8.float()).any()
    assert not torch.isnan(k.float()).any()
    assert not torch.isnan(weights.float()).any()
    assert (q_fp8.float()[1] == 0).all()
    assert (weights[1] == 0).all()


def main(argv):
    del argv

    raise SystemExit(pytest.main([__file__, "-p", "no:cacheprovider"]))


if __name__ == "__main__":
    from absl import app

    app.run(main)
