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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.qwen4_exp.ops.qsa import _qsa_attention, qsa_attention


def _apply_partial_rope(
    x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor
) -> torch.Tensor:
    rotary_dim = cos_sin.shape[-1]
    cos, sin = cos_sin[positions].float().unsqueeze(1).chunk(2, dim=-1)
    rot, rest = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    x1, x2 = rot.chunk(2, dim=-1)
    rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rot.to(x.dtype), rest], dim=-1)


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


def _reference(seq, p, k_norm_weight, cos_sin):
    """HF's indexer and masked attention for the query at position p."""
    visible = p + 1
    num_blocks = visible // RATIO
    selected = list(range(visible))
    if num_blocks > 0:
        raw = seq["raw"][: num_blocks * RATIO].view(num_blocks, RATIO, DI)
        pooled = _gemma_norm(raw.mean(1), k_norm_weight)
        starts = torch.arange(num_blocks) * RATIO
        keys = _apply_partial_rope(pooled[:, None], starts, cos_sin)[:, 0]
        iq_roped = _apply_partial_rope(
            seq["iq"][p : p + 1], torch.tensor([p]), cos_sin
        )[0]
        scores = torch.relu(iq_roped @ keys.T).sum(0) / math.sqrt(DI)
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
        out, index_pool = _qsa_attention(
            jnp.asarray(q.numpy()),
            jnp.asarray(iq.numpy()),
            jnp.asarray(raw.numpy()),
            jnp.asarray(dense.numpy()),
            jnp.asarray(positions, jnp.int32),
            jnp.asarray(query_start_loc, jnp.int32),
            jnp.asarray(done, jnp.int32),
            jnp.asarray(tables.reshape(-1).numpy(), jnp.int32),
            jnp.asarray(main_pool.numpy()),
            index_pool,
            jnp.asarray(cos_sin.numpy()),
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


def test_short_queries_return_dense_output_unchanged() -> None:
    """Steps whose queries all see at most TOPK complete blocks skip the
    sparse path and return dense_out unchanged."""
    torch.manual_seed(1)
    n = (TOPK + 1) * RATIO - 1
    dense = torch.randn(n, H, D)
    out, _ = _qsa_attention(
        jnp.zeros((n, H, D)),
        jnp.zeros((n, IH, DI)),
        jnp.zeros((n, DI)),
        jnp.asarray(dense.numpy()),
        jnp.arange(n, dtype=jnp.int32),
        jnp.asarray([0, n], jnp.int32),
        jnp.asarray([n], jnp.int32),
        jnp.arange(1, 1 + PAGES_PER_REQ, dtype=jnp.int32),
        jnp.zeros((1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128)),
        jnp.zeros((1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128)),
        jnp.asarray(_cos_sin(64).numpy()),
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
    out, _ = _qsa_attention(
        jnp.ones((n, H, D), dtype=jnp.bfloat16),
        jnp.ones((n, IH, DI), dtype=jnp.bfloat16),
        jnp.ones((n, DI), dtype=jnp.bfloat16),
        jnp.zeros((n, H, D), dtype=jnp.bfloat16),
        jnp.arange(n, dtype=jnp.int32),
        jnp.asarray([0, n], jnp.int32),
        jnp.asarray([n], jnp.int32),
        jnp.arange(1, 1 + PAGES_PER_REQ, dtype=jnp.int32),
        main_pool,
        jnp.zeros((1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128), dtype=jnp.float8_e4m3fn),
        jnp.asarray(_cos_sin(64).numpy()),
        jnp.zeros(DI, dtype=jnp.bfloat16),
        jnp.ones(2, jnp.float32),
        **_QSA_KWARGS,
        token_batch=4,
    )
    assert np.isfinite(np.asarray(out, dtype=np.float32)).all()


def test_mismatched_input_shapes_are_rejected() -> None:
    n = 4
    pool = jnp.zeros((1 + PAGES_PER_REQ, PAGE, 2 * KVH, 1, 128))
    with pytest.raises(ValueError, match="q and dense_out"):
        _qsa_attention(
            jnp.zeros((n, H, D)),
            jnp.zeros((n, IH, DI)),
            jnp.zeros((n, DI)),
            jnp.zeros((n, H, D + 1)),
            jnp.arange(n, dtype=jnp.int32),
            jnp.asarray([0, n], jnp.int32),
            jnp.asarray([n], jnp.int32),
            jnp.arange(1, 1 + PAGES_PER_REQ, dtype=jnp.int32),
            pool,
            pool,
            jnp.asarray(_cos_sin(64).numpy()),
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

    with jax.default_matmul_precision("highest"):
        out, _ = _qsa_attention(
            jnp.asarray(seq["q"].float().numpy(), jnp.bfloat16),
            jnp.asarray(seq["iq"].float().numpy(), jnp.bfloat16),
            jnp.asarray(seq["raw"].float().numpy(), jnp.bfloat16),
            jnp.zeros((n, H, D), jnp.bfloat16),
            jnp.arange(n, dtype=jnp.int32),
            jnp.asarray([0, n], jnp.int32),
            jnp.asarray([n], jnp.int32),
            jnp.asarray(tables.numpy(), jnp.int32),
            jnp.asarray(main_pool.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn),
            index_pool,
            jnp.asarray(cos_sin.numpy()),
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
            cos_sin_cache=torch.empty((64, ROTARY), dtype=torch.float32),
            k_norm_weight=torch.empty((DI,), dtype=torch.bfloat16),
            kv_scales=torch.empty((2,), dtype=torch.float32),
            **_QSA_KWARGS,
        )
        assert out.shape == (num_tokens, H, D)
        assert out.dtype == torch.bfloat16
