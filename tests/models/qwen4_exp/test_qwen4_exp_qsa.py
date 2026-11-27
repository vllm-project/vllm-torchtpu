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
"""Tests for Qwen4Exp QSA against the HF reference math."""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import Qwen4ExpTextConfig
from vllm.config import (
    CacheConfig,
    CompilationMode,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers import rotary_embedding as rope_lib
from vllm.platforms import current_platform

import vllm_torchtpu.layers.adapter.rotary_embedding  # noqa: F401
from vllm_torchtpu import rope_in_graph, rope_in_graph_flag
from vllm_torchtpu.models.qwen4_exp.ops.qsa import _qsa_attention, qsa_attention
from vllm_torchtpu.models.qwen4_exp.qsa import (
    Qwen4ExpQSAAttention,
    _apply_partial_rope,
)

H, KVH, D, DI, IH = 4, 2, 32, 16, 3
RATIO, TOPK = 4, 2
PAGE, PAGES_PER_REQ = 4, 10
ROTARY = 8
SCALE = D**-0.5
_QSA_KWARGS = {
    "num_kv_heads": KVH,
    "compress_ratio": RATIO,
    "topk_blocks": TOPK,
    "scale": SCALE,
    "norm_eps": 1e-6,
}


def _cos_sin(max_pos: int) -> torch.Tensor:
    inv_freq = 1.0 / (1e4 ** (torch.arange(0, ROTARY, 2).float() / ROTARY))
    freqs = torch.arange(max_pos).float()[:, None] * inv_freq
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def _gemma_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + weight)


def _reference_pooled_keys(seq, num_blocks, k_norm_weight, cos_sin):
    raw = seq["raw"][: num_blocks * RATIO].view(num_blocks, RATIO, DI)
    pooled = _gemma_norm(raw.mean(1), k_norm_weight)
    starts = torch.arange(num_blocks) * RATIO
    return _apply_partial_rope(pooled[:, None], cos_sin[starts])[:, 0]


def _reference(seq, p, k_norm_weight, cos_sin):
    """HF's indexer and masked attention for the query at position p."""
    visible = p + 1
    num_blocks = visible // RATIO
    selected = list(range(visible))
    if num_blocks > 0:
        keys = _reference_pooled_keys(seq, num_blocks, k_norm_weight, cos_sin)
        scores = torch.relu(seq["iq"][p] @ keys.T).sum(0) / math.sqrt(DI)
        top = scores.topk(min(TOPK, num_blocks)).indices
        selected = [b * RATIO + i for b in top.tolist() for i in range(RATIO)]
        selected += list(range(num_blocks * RATIO, visible))
    idx = torch.tensor(sorted(selected))
    k = seq["k"][idx].repeat_interleave(H // KVH, dim=1)  # [S, H, D]
    v = seq["v"][idx].repeat_interleave(H // KVH, dim=1)
    logits = torch.einsum("hd,shd->hs", seq["q"][p], k) * SCALE
    return torch.einsum("hs,shd->hd", logits.softmax(-1), v)


def _dense(seq, p):
    k = seq["k"][: p + 1].repeat_interleave(H // KVH, dim=1)
    v = seq["v"][: p + 1].repeat_interleave(H // KVH, dim=1)
    logits = torch.einsum("hd,shd->hs", seq["q"][p], k) * SCALE
    return torch.einsum("hs,shd->hd", logits.softmax(-1), v)


def _write_main(pool, tables, req, positions, seq):
    """What ragged paged attention writes: K/V of each kv head at 2h / 2h+1."""
    for p in positions:
        page = tables[req, p // PAGE]
        for h in range(KVH):
            pool[page, p % PAGE, 2 * h, 0, :D] = seq["k"][p, h]
            pool[page, p % PAGE, 2 * h + 1, 0, :D] = seq["v"][p, h]


@pytest.mark.parametrize("block_tile", [4096, 3], ids=["single_tile", "multi_tile"])
@pytest.mark.parametrize(
    "chunks",
    [[[23], [30]], [[11, 1, 1, 10], [7, 9, 1, 13]], [[1] * 23, [20, 10]]],
    ids=["one_step", "mixed", "decode"],
)
def test_any_chunking_matches_reference_math(chunks, block_tile) -> None:
    with jax.default_matmul_precision("highest"):
        _run_any_chunking_matches_reference_math(chunks, block_tile=block_tile)


def _run_any_chunking_matches_reference_math(chunks, *, block_tile: int = 4096) -> None:
    torch.manual_seed(0)
    lengths = [sum(c) for c in chunks]
    seqs = [
        {
            "q": torch.randn(n, H, D),
            "k": torch.randn(n, KVH, D),
            "v": torch.randn(n, KVH, D),
            "iq": torch.randn(n, IH, DI),
            "raw": torch.randn(n, DI),
        }
        for n in lengths
    ]
    k_norm_weight = torch.randn(DI) * 0.1
    cos_sin = _cos_sin(64)
    num_pages = 1 + len(seqs) * PAGES_PER_REQ
    # Request r owns pages 1 + r, 1 + r + 2, ...; page 0 is the null page.
    tables = torch.tensor(
        [
            [1 + r + len(seqs) * i for i in range(PAGES_PER_REQ)]
            for r in range(len(seqs))
        ]
    )
    main_pool = torch.zeros(num_pages, PAGE, 2 * KVH, 1, 128)
    index_pool = jnp.zeros((num_pages, PAGE, 2 * KVH, 1, 128), jnp.float32)
    done = [0] * len(seqs)
    for step in range(max(map(len, chunks))):
        rows = []
        for r, c in enumerate(chunks):
            n = c[step] if step < len(c) else 0
            rows.append(list(range(done[r], done[r] + n)))
            _write_main(main_pool, tables, r, rows[r], seqs[r])
            done[r] += n
        flat = [(r, p) for r in range(len(seqs)) for p in rows[r]]
        pad = 3
        query_start_loc = np.cumsum([0] + [len(x) for x in rows])
        q = torch.stack([seqs[r]["q"][p] for r, p in flat] + [torch.zeros(H, D)] * pad)
        iq = torch.stack(
            [seqs[r]["iq"][p] for r, p in flat] + [torch.zeros(IH, DI)] * pad
        )
        raw = torch.stack(
            [seqs[r]["raw"][p] for r, p in flat] + [torch.zeros(DI)] * pad
        )
        dense = torch.stack(
            [_dense(seqs[r], p) for r, p in flat] + [torch.zeros(H, D)] * pad
        )
        positions = [p for _, p in flat] + [0] * pad
        pos_arr = jnp.asarray(positions, jnp.int32)
        pooled_cos_sin = jnp.asarray(cos_sin.numpy())[
            jnp.maximum(pos_arr - (RATIO - 1), 0)
        ]
        out, index_pool = _qsa_attention(
            jnp.asarray(q.numpy()),
            jnp.asarray(iq.numpy()),
            jnp.asarray(raw.numpy()),
            jnp.asarray(dense.numpy()),
            pos_arr,
            jnp.asarray(query_start_loc, jnp.int32),
            jnp.asarray(done, jnp.int32),
            jnp.asarray(tables.reshape(-1).numpy(), jnp.int32),
            jnp.asarray(main_pool.numpy()),
            index_pool,
            pooled_cos_sin,
            jnp.asarray(k_norm_weight.numpy()),
            jnp.ones(2, jnp.float32),
            **_QSA_KWARGS,
            token_batch=4,
            block_tile=block_tile,
        )
        out = torch.from_numpy(np.asarray(out))
        for i, (r, p) in enumerate(flat):
            expected = _reference(seqs[r], p, k_norm_weight, cos_sin)
            torch.testing.assert_close(out[i], expected, rtol=1e-5, atol=1e-5)

    # Index pool stores raw_keys in head 0 and pooled keys in head 1.
    pool_t = torch.from_numpy(np.asarray(index_pool).copy()).reshape(
        -1, 2 * KVH, 1, 128
    )
    for r, length in enumerate(lengths):
        num_blocks = length // RATIO
        block_ends = [b * RATIO + RATIO - 1 for b in range(num_blocks)]
        slots = [int(tables[r, p // PAGE]) * PAGE + (p % PAGE) for p in block_ends]
        expected_keys = _reference_pooled_keys(
            seqs[r], num_blocks, k_norm_weight, cos_sin
        )
        torch.testing.assert_close(
            pool_t[slots, 1, 0, :DI], expected_keys, rtol=1e-5, atol=1e-5
        )


def test_short_queries_return_dense_output_unchanged() -> None:
    """Steps whose queries all see at most TOPK complete blocks skip the
    sparse path and return dense_out unchanged."""
    torch.manual_seed(1)
    n = (TOPK + 1) * RATIO - 1
    dense = torch.randn(n, H, D)
    positions = jnp.arange(n, dtype=jnp.int32)
    pooled_cos_sin = jnp.asarray(_cos_sin(64).numpy())[
        jnp.maximum(positions - (RATIO - 1), 0)
    ]
    out, _ = _qsa_attention(
        jnp.zeros((n, H, D)),
        jnp.zeros((n, IH, DI)),
        jnp.zeros((n, DI)),
        jnp.asarray(dense.numpy()),
        positions,
        jnp.asarray([0, n], jnp.int32),
        jnp.asarray([n], jnp.int32),
        jnp.arange(1, 1 + PAGES_PER_REQ, dtype=jnp.int32),
        jnp.zeros((1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128)),
        jnp.zeros((1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128)),
        pooled_cos_sin,
        jnp.zeros(DI),
        jnp.ones(2, jnp.float32),
        **_QSA_KWARGS,
        token_batch=4,
    )
    np.testing.assert_array_equal(np.asarray(out), dense.numpy())


def test_nan_in_unwritten_tail_does_not_reach_output() -> None:
    """Uninitialized fp8 NaNs in masked-out tail slots must not poison output."""
    n = (TOPK + 1) * RATIO
    main_pool = jnp.full(
        (1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128), jnp.nan, dtype=jnp.float8_e4m3fn
    )
    # Write finite values only to the [0, n) positions used by the request.
    slots = jnp.arange(PAGE, PAGE + n)
    flat = main_pool.reshape(-1, 2 * KVH, 1, 128)
    main_pool = (
        flat.at[slots]
        .set(jnp.ones((n, 2 * KVH, 1, 128), dtype=jnp.float8_e4m3fn))
        .reshape(main_pool.shape)
    )
    positions = jnp.arange(n, dtype=jnp.int32)
    pooled_cos_sin = jnp.asarray(_cos_sin(64).numpy())[
        jnp.maximum(positions - (RATIO - 1), 0)
    ]
    out, _ = _qsa_attention(
        jnp.ones((n, H, D), dtype=jnp.bfloat16),
        jnp.ones((n, IH, DI), dtype=jnp.bfloat16),
        jnp.ones((n, DI), dtype=jnp.bfloat16),
        jnp.zeros((n, H, D), dtype=jnp.bfloat16),
        positions,
        jnp.asarray([0, n], jnp.int32),
        jnp.asarray([n], jnp.int32),
        jnp.arange(1, 1 + PAGES_PER_REQ, dtype=jnp.int32),
        main_pool,
        jnp.zeros((1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128), dtype=jnp.float8_e4m3fn),
        pooled_cos_sin,
        jnp.zeros(DI, dtype=jnp.bfloat16),
        jnp.ones(2, jnp.float32),
        **_QSA_KWARGS,
        token_batch=4,
    )
    assert np.isfinite(np.asarray(out, dtype=np.float32)).all()


def test_mismatched_input_shapes_are_rejected() -> None:
    n = 4
    pool = jnp.zeros((1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128))
    base_args = (
        jnp.zeros((n, H, D)),
        jnp.zeros((n, IH, DI)),
        jnp.zeros((n, DI)),
    )
    tail_args = (
        jnp.arange(n, dtype=jnp.int32),
        jnp.asarray([0, n], jnp.int32),
        jnp.asarray([n], jnp.int32),
        jnp.arange(1, 1 + PAGES_PER_REQ, dtype=jnp.int32),
        pool,
        pool,
    )
    with pytest.raises(ValueError, match="q and dense_out"):
        _qsa_attention(
            *base_args,
            jnp.zeros((n, H, D + 1)),
            *tail_args,
            jnp.zeros((n, ROTARY), dtype=jnp.float32),
            jnp.zeros(DI),
            jnp.ones(2, jnp.float32),
            **_QSA_KWARGS,
        )


def test_fp8_main_pool_with_kv_scales_matches_reference() -> None:
    """Sparse QSA dequantizes fp8 main_pool K/V with kv_scales accurately."""
    torch.manual_seed(2)
    n = 23
    k_scale, v_scale = 0.5, 0.25
    seq = {
        "q": torch.randn(n, H, D, dtype=torch.bfloat16),
        "k": (torch.randn(n, KVH, D) * 2.0).to(torch.float8_e4m3fn),
        "v": (torch.randn(n, KVH, D) * 2.0).to(torch.float8_e4m3fn),
        "iq": torch.randn(n, IH, DI, dtype=torch.bfloat16),
        "raw": torch.randn(n, DI, dtype=torch.bfloat16),
    }
    k_norm_weight = torch.randn(DI, dtype=torch.bfloat16) * 0.1
    cos_sin = _cos_sin(64)
    num_pages = 1 + PAGES_PER_REQ
    tables = torch.arange(1, num_pages, dtype=torch.int32)
    main_pool = torch.zeros(num_pages, PAGE, 2 * KVH, 1, 128, dtype=torch.float8_e4m3fn)
    _write_main(main_pool, tables[None], 0, range(n), seq)
    index_pool = jnp.zeros((num_pages, PAGE, 2 * KVH, 1, 128), jnp.bfloat16)

    positions = jnp.arange(n, dtype=jnp.int32)
    pooled_cos_sin = jnp.asarray(cos_sin.numpy())[
        jnp.maximum(positions - (RATIO - 1), 0)
    ]
    with jax.default_matmul_precision("highest"):
        out, _ = _qsa_attention(
            jnp.asarray(seq["q"].float().numpy(), jnp.bfloat16),
            jnp.asarray(seq["iq"].float().numpy(), jnp.bfloat16),
            jnp.asarray(seq["raw"].float().numpy(), jnp.bfloat16),
            jnp.zeros((n, H, D), jnp.bfloat16),
            positions,
            jnp.asarray([0, n], jnp.int32),
            jnp.asarray([n], jnp.int32),
            jnp.asarray(tables.numpy(), jnp.int32),
            jnp.asarray(main_pool.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn),
            index_pool,
            pooled_cos_sin,
            jnp.asarray(k_norm_weight.float().numpy(), jnp.bfloat16),
            jnp.asarray([k_scale, v_scale], jnp.float32),
            **_QSA_KWARGS,
            token_batch=4,
        )
    ref_seq = {
        "q": seq["q"].float(),
        "k": seq["k"].float() * k_scale,
        "v": seq["v"].float() * v_scale,
        "iq": seq["iq"].float(),
        "raw": seq["raw"].float(),
    }
    out_t = torch.from_numpy(np.asarray(out, dtype=np.float32))
    # Positions >= (TOPK + 1) * RATIO - 1 take the sparse path.
    first_sparse = (TOPK + 1) * RATIO - 1
    for p in range(first_sparse, n):
        expected = _reference(ref_seq, p, k_norm_weight.float(), cos_sin)
        torch.testing.assert_close(out_t[p], expected, rtol=2e-2, atol=2e-2)


def test_qsa_attention_op_traces_with_symbolic_fake_tensors() -> None:
    """Static jax_op registration traces cleanly under symbolic FakeTensorMode."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        num_tokens = shape_env.create_unbacked_symint()
        q = torch.empty((num_tokens, H, D), dtype=torch.bfloat16)
        dense_out = torch.empty((num_tokens, H, D), dtype=torch.bfloat16)
        pool = torch.empty(
            (1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128), dtype=torch.bfloat16
        )
        index_pool = torch.empty_like(pool)
        out = qsa_attention(
            q=q,
            index_q=torch.empty((num_tokens, IH, DI), dtype=torch.bfloat16),
            raw_keys=torch.empty((num_tokens, DI), dtype=torch.bfloat16),
            dense_out=dense_out,
            positions=torch.empty((num_tokens,), dtype=torch.int32),
            query_start_loc=torch.empty((2,), dtype=torch.int32),
            seq_lens=torch.empty((1,), dtype=torch.int32),
            block_tables=torch.empty((PAGES_PER_REQ,), dtype=torch.int32),
            main_pool=pool,
            index_pool=index_pool,
            cos_sin=torch.empty((num_tokens, ROTARY), dtype=torch.float32),
            k_norm_weight=torch.empty((DI,), dtype=torch.bfloat16),
            kv_scales=torch.empty((2,), dtype=torch.float32),
            **_QSA_KWARGS,
        )
        assert out.shape == (num_tokens, H, D)
        assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("in_graph", [False, True], ids=["tensor_table", "in_graph"])
def test_qwen4_exp_qsa_attention_sparse_with_rope_modes(
    monkeypatch, request, cpu_model_tests, in_graph: bool
) -> None:
    """Verify Qwen4ExpQSAAttention passes pooled cos_sin to qsa_attention."""
    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    request.getfixturevalue("dist_ctx")
    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    text_cfg = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
        max_position_embeddings=64,
        dtype="float32",
    )
    vllm_cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(block_size=16),
        additional_config={rope_in_graph_flag.KEY: in_graph},
    )
    vllm_cfg.model_config = SimpleNamespace(
        max_model_len=64,
        hf_config=text_cfg,
        hf_text_config=text_cfg,
        dtype=torch.float32,
        multimodal_config=None,
        is_mm_prefix_lm=False,
        rswa_window=None,
    )  # type: ignore[assignment]
    vllm_cfg.compilation_config.mode = CompilationMode.NONE

    with set_current_vllm_config(vllm_cfg):
        layer = Qwen4ExpQSAAttention(
            text_cfg,
            model_config=vllm_cfg.model_config,
            cache_config=vllm_cfg.cache_config,
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )

    assert layer.sparse
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_(0.0, 0.02, generator=generator)
    if in_graph:
        assert isinstance(
            layer.rotary_emb.cos_sin_cache, rope_in_graph.InGraphCosSinCache
        )
    else:
        assert isinstance(layer.rotary_emb.cos_sin_cache, torch.Tensor)

    captured: dict[str, torch.Tensor] = {}

    def _fake_qsa_attention(**kwargs):
        captured["cos_sin"] = kwargs["cos_sin"]
        captured["index_q"] = kwargs["index_q"]
        return kwargs["dense_out"].clone()

    monkeypatch.setattr(
        "vllm_torchtpu.models.qwen4_exp.qsa.qsa_attention", _fake_qsa_attention
    )

    num_tokens = 8
    positions = torch.arange(num_tokens, dtype=torch.int64)
    hidden_states = torch.randn(num_tokens, text_cfg.hidden_size, generator=generator)
    q = torch.randn(num_tokens, layer.num_heads * layer.head_dim, generator=generator)
    dense_out = torch.randn(
        num_tokens, layer.num_heads * layer.head_dim, generator=generator
    )
    main_pool = torch.zeros(2, 16, 2, 1, 16)
    layer.index_cache.kv_cache = torch.zeros(2, 16, 2, 1, 16)

    attn_metadata = {
        layer.attn.layer_name: SimpleNamespace(
            query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
            block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
        )
    }
    with set_forward_context(attn_metadata, vllm_cfg):
        out = layer._sparse_attention(positions, hidden_states, q, dense_out, main_pool)

    assert torch.equal(out, dense_out)
    with torch.no_grad():
        qk, _ = layer.indexer.index_qk_proj(hidden_states)
        idx_q, _ = qk.split(
            [layer.indexer.num_heads * layer.indexer.head_dim, layer.indexer.head_dim],
            dim=-1,
        )
        idx_q = layer.indexer.q_layernorm(
            idx_q.reshape(-1, layer.indexer.num_heads, layer.indexer.head_dim)
        )
        expected_index_q = _apply_partial_rope(
            idx_q, layer.rotary_emb.cos_sin_cache[positions]
        )
    assert torch.isfinite(expected_index_q).all()
    assert expected_index_q.abs().max() > 0
    assert torch.equal(captured["index_q"], expected_index_q)
    pooled_pos = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4], dtype=torch.int64)
    expected_pooled = layer.rotary_emb.cos_sin_cache[pooled_pos]
    assert isinstance(captured["cos_sin"], torch.Tensor)
    assert captured["cos_sin"].shape == (num_tokens, layer.rotary_emb.rotary_dim)
    assert torch.equal(captured["cos_sin"], expected_pooled)
    # Confirm pooled cos_sin uses pooled_pos rather than unshifted positions.
    assert not torch.equal(
        captured["cos_sin"],
        layer.rotary_emb.cos_sin_cache[positions],
    )
