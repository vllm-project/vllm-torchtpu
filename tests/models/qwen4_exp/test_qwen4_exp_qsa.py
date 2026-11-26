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


@pytest.mark.parametrize("num_req_slots", [1, 16], ids=["single_req", "padded_reqs"])
def test_padded_request_slots_match_reference(num_req_slots: int) -> None:
    """The runner pads query_start_loc to the request bucket and tokens to the
    token bucket. Padded slots point at the null page and must not change the
    real rows, for a full prefill chunk and for a padded verify window."""
    torch.manual_seed(11)
    n_total = 20
    seq = {
        "q": torch.randn(n_total, H, D),
        "k": torch.randn(n_total, KVH, D),
        "v": torch.randn(n_total, KVH, D),
        "iq": torch.randn(n_total, IH, DI),
        "raw": torch.randn(n_total, DI),
    }
    k_norm_weight = torch.randn(DI) * 0.1
    cos_sin = _cos_sin(64)
    num_pages = 1 + PAGES_PER_REQ
    # Request 0 owns pages 1..PAGES_PER_REQ; any padded slots point at page 0.
    tables = np.zeros((num_req_slots, PAGES_PER_REQ), dtype=np.int32)
    tables[0] = np.arange(1, 1 + PAGES_PER_REQ, dtype=np.int32)
    main_pool = torch.zeros(num_pages, PAGE, 2 * KVH, 1, 128)
    _write_main(main_pool, tables, 0, list(range(n_total)), seq)
    # A read of the null page would turn the output into NaN.
    main_pool[0] = float("nan")
    index_pool = jnp.zeros((num_pages, PAGE, 2 * KVH, 1, 128), jnp.float32)
    first_sparse = (TOPK + 1) * RATIO - 1

    def run_step(real_positions: list[int], index_pool):
        pad = 16 - len(real_positions)
        rows = real_positions
        q = torch.cat([seq["q"][rows], torch.zeros(pad, H, D)])
        iq = torch.cat([seq["iq"][rows], torch.zeros(pad, IH, DI)])
        raw = torch.cat([seq["raw"][rows], torch.zeros(pad, DI)])
        dense = torch.stack([_dense(seq, p) for p in rows] + [torch.zeros(H, D)] * pad)
        # Padded tokens carry a large stale position.
        positions = np.array(rows + [60] * pad, dtype=np.int32)
        qsl = np.array([0] + [len(rows)] * num_req_slots, dtype=np.int32)
        seq_lens = np.array([rows[-1] + 1] + [0] * (num_req_slots - 1), dtype=np.int32)
        out, index_pool = _qsa_attention(
            jnp.asarray(q.numpy()),
            jnp.asarray(iq.numpy()),
            jnp.asarray(raw.numpy()),
            jnp.asarray(dense.numpy()),
            jnp.asarray(positions),
            jnp.asarray(qsl),
            jnp.asarray(seq_lens),
            jnp.asarray(tables.reshape(-1), jnp.int32),
            jnp.asarray(main_pool.numpy()),
            index_pool,
            jnp.asarray(cos_sin.numpy()),
            jnp.asarray(k_norm_weight.numpy()),
            jnp.ones(2, jnp.float32),
            **_QSA_KWARGS,
            token_batch=16,
        )
        return torch.from_numpy(np.asarray(out).copy()), index_pool

    with jax.default_matmul_precision("highest"):
        # A full 16-token prefill chunk of request 0, then a 4-token verify
        # window padded to 16 tokens.
        prefill_out, index_pool = run_step(list(range(16)), index_pool)
        verify_out, _ = run_step([16, 17, 18, 19], index_pool)

    for p in range(first_sparse, 16):
        expected = _reference(seq, p, k_norm_weight, cos_sin)
        torch.testing.assert_close(prefill_out[p], expected, rtol=1e-5, atol=1e-5)
    for i, p in enumerate(range(16, 20)):
        expected = _reference(seq, p, k_norm_weight, cos_sin)
        torch.testing.assert_close(verify_out[i], expected, rtol=1e-5, atol=1e-5)
    # Padded rows keep the dense output they came in with.
    assert torch.equal(verify_out[4:], torch.zeros(12, H, D))


def test_qsa_rejects_head_dim_64() -> None:
    """At head_dim=64 the TPU KV cache packs K and V of a head into one row,
    which QSA would read as K with an all-zero V."""
    from types import SimpleNamespace

    from vllm_torchtpu.models.qwen4_exp.layout import unfused_qkv
    from vllm_torchtpu.models.qwen4_exp.qsa import Qwen4ExpQSA

    config = SimpleNamespace(
        indexer_kv_heads=1,
        indexer_compress_ratio=4,
        indexer_budget=16,
        indexer_head_dim=64,
        indexer_n_heads=2,
        hidden_size=128,
        rms_norm_eps=1e-6,
    )
    with pytest.raises(NotImplementedError, match="head_dim=64"):
        Qwen4ExpQSA(
            config,  # type: ignore[arg-type]
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            scaling=64**-0.5,
            qkv_layout=unfused_qkv(),
            cache_config=None,
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )


@pytest.mark.parametrize("pages_per_req", [0, 1, 2])
def test_qsa_narrow_table_returns_dense_without_topk_crash(
    pages_per_req: int,
) -> None:
    """When total_blocks <= topk_blocks (including 0), _qsa_attention returns
    dense_out cleanly without calling lax.top_k or dividing by zero."""
    n = 6
    dense = jnp.arange(n * H * D, dtype=jnp.float32).reshape(n, H, D)
    num_pages = max(1, 1 + pages_per_req)
    pool = jnp.zeros((num_pages, PAGE, 2 * KVH, 1, 128), jnp.float32)
    tables = jnp.arange(1, 1 + pages_per_req, dtype=jnp.int32)
    out, _ = _qsa_attention(
        jnp.ones((n, H, D), jnp.float32),
        jnp.ones((n, IH, DI), jnp.float32),
        jnp.ones((n, DI), jnp.float32),
        dense,
        jnp.arange(n, dtype=jnp.int32),
        jnp.asarray([0, n], jnp.int32),
        jnp.asarray([n], jnp.int32),
        tables,
        pool,
        pool,
        jnp.ones((64, ROTARY), jnp.float32),
        jnp.zeros((DI,), jnp.float32),
        jnp.ones((2,), jnp.float32),
        **_QSA_KWARGS,
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(dense))


@pytest.mark.parametrize("verify_batch", [4, 8])
def test_qsa_grouped_verify_matches_per_token_on_misaligned_mixed_batch(
    verify_batch: int,
) -> None:
    """Cursor-grouped verify matches verify_batch=1 within 1e-5 on random
    non-constant inputs across misaligned decode, verify, and prefill runs."""
    rng = np.random.default_rng(7)
    pages_per_req = 12
    num_reqs = 5
    q_lens = [1, 4, 3, 4, 6]
    hist_lens = [18, 22, 14, 25, 19]
    num_pages = 1 + num_reqs * pages_per_req
    tables = np.array(
        [
            [1 + r * pages_per_req + i for i in range(pages_per_req)]
            for r in range(num_reqs)
        ],
        dtype=np.int32,
    )
    main_pool = jnp.asarray(
        rng.standard_normal((num_pages, PAGE, 2 * KVH, 1, 128)).astype(np.float32)
    )
    index_pool = jnp.asarray(
        rng.standard_normal((num_pages, PAGE, 2 * KVH, 1, 128)).astype(np.float32)
    )
    cos_sin = jnp.asarray(rng.standard_normal((64, ROTARY)).astype(np.float32))
    k_norm = jnp.asarray(rng.standard_normal((DI,)).astype(np.float32) * 0.1)

    total_real = sum(q_lens)
    pad = 5
    num_tokens = total_real + pad
    qsl = np.cumsum([0] + q_lens, dtype=np.int32)
    positions = np.concatenate(
        [
            np.arange(hist_lens[r], hist_lens[r] + q_lens[r], dtype=np.int32)
            for r in range(num_reqs)
        ]
        + [np.zeros(pad, dtype=np.int32)]
    )
    seq_lens = np.array([h0 + ql for h0, ql in zip(hist_lens, q_lens)], dtype=np.int32)

    q = jnp.asarray(rng.standard_normal((num_tokens, H, D)).astype(np.float32))
    iq = jnp.asarray(rng.standard_normal((num_tokens, IH, DI)).astype(np.float32))
    rk = jnp.asarray(rng.standard_normal((num_tokens, DI)).astype(np.float32))
    dense = jnp.asarray(rng.standard_normal((num_tokens, H, D)).astype(np.float32))

    common = dict(**_QSA_KWARGS, token_batch=16, block_tile=4)
    with jax.default_matmul_precision("highest"):
        out_ref, pool_ref = _qsa_attention(
            q,
            iq,
            rk,
            dense,
            jnp.asarray(positions),
            jnp.asarray(qsl),
            jnp.asarray(seq_lens),
            jnp.asarray(tables.reshape(-1)),
            main_pool,
            index_pool,
            cos_sin,
            k_norm,
            jnp.ones(2, jnp.float32),
            verify_batch=1,
            **common,
        )
        out_grp, pool_grp = _qsa_attention(
            q,
            iq,
            rk,
            dense,
            jnp.asarray(positions),
            jnp.asarray(qsl),
            jnp.asarray(seq_lens),
            jnp.asarray(tables.reshape(-1)),
            main_pool,
            index_pool,
            cos_sin,
            k_norm,
            jnp.ones(2, jnp.float32),
            verify_batch=verify_batch,
            **common,
        )
    np.testing.assert_allclose(
        np.asarray(out_grp), np.asarray(out_ref), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(np.asarray(pool_grp), np.asarray(pool_ref))


def test_qsa_multi_req_one_token_decode_matches_reference() -> None:
    """Multi-request 1-token decode takes the M=1 per_token path (has_runs=False)
    and matches the HF reference math."""
    torch.manual_seed(19)
    num_reqs = 5
    prefill_lens = [15, 19, 13, 21, 17]
    seqs = [
        {
            "q": torch.randn(n + 1, H, D),
            "k": torch.randn(n + 1, KVH, D),
            "v": torch.randn(n + 1, KVH, D),
            "iq": torch.randn(n + 1, IH, DI),
            "raw": torch.randn(n + 1, DI),
        }
        for n in prefill_lens
    ]
    k_norm_weight = torch.randn(DI) * 0.1
    cos_sin = _cos_sin(64)
    num_pages = 1 + num_reqs * PAGES_PER_REQ
    tables = torch.tensor(
        [
            [1 + r * PAGES_PER_REQ + i for i in range(PAGES_PER_REQ)]
            for r in range(num_reqs)
        ],
        dtype=torch.int32,
    )
    main_pool = torch.zeros(num_pages, PAGE, 2 * KVH, 1, 128)
    index_pool = jnp.zeros((num_pages, PAGE, 2 * KVH, 1, 128), jnp.float32)

    with jax.default_matmul_precision("highest"):
        # Step 1: Prefill all 5 requests.
        flat_pre = [(r, p) for r, n in enumerate(prefill_lens) for p in range(n)]
        for r, n in enumerate(prefill_lens):
            _write_main(main_pool, tables, r, list(range(n)), seqs[r])
        qsl_pre = np.cumsum([0] + prefill_lens, dtype=np.int32)
        _, index_pool = _qsa_attention(
            jnp.asarray(torch.stack([seqs[r]["q"][p] for r, p in flat_pre]).numpy()),
            jnp.asarray(torch.stack([seqs[r]["iq"][p] for r, p in flat_pre]).numpy()),
            jnp.asarray(torch.stack([seqs[r]["raw"][p] for r, p in flat_pre]).numpy()),
            jnp.zeros((len(flat_pre), H, D), jnp.float32),
            jnp.asarray([p for _, p in flat_pre], jnp.int32),
            jnp.asarray(qsl_pre, jnp.int32),
            jnp.asarray(prefill_lens, jnp.int32),
            jnp.asarray(tables.reshape(-1).numpy(), jnp.int32),
            jnp.asarray(main_pool.numpy()),
            index_pool,
            jnp.asarray(cos_sin.numpy()),
            jnp.asarray(k_norm_weight.numpy()),
            jnp.ones(2, jnp.float32),
            **_QSA_KWARGS,
            token_batch=16,
            block_tile=4,
        )

        # Step 2: 1-token decode per request padded to 8 tokens (has_runs=False).
        pad = 3
        for r, n in enumerate(prefill_lens):
            _write_main(main_pool, tables, r, [n], seqs[r])
        q_dec = torch.stack(
            [seqs[r]["q"][n] for r, n in enumerate(prefill_lens)]
            + [torch.zeros(H, D)] * pad
        )
        iq_dec = torch.stack(
            [seqs[r]["iq"][n] for r, n in enumerate(prefill_lens)]
            + [torch.zeros(IH, DI)] * pad
        )
        raw_dec = torch.stack(
            [seqs[r]["raw"][n] for r, n in enumerate(prefill_lens)]
            + [torch.zeros(DI)] * pad
        )
        dense_dec = torch.stack(
            [_dense(seqs[r], n) for r, n in enumerate(prefill_lens)]
            + [torch.zeros(H, D)] * pad
        )
        out_dec, _ = _qsa_attention(
            jnp.asarray(q_dec.numpy()),
            jnp.asarray(iq_dec.numpy()),
            jnp.asarray(raw_dec.numpy()),
            jnp.asarray(dense_dec.numpy()),
            jnp.asarray(prefill_lens + [0] * pad, jnp.int32),
            jnp.asarray(list(range(num_reqs + 1)), jnp.int32),
            jnp.asarray([n + 1 for n in prefill_lens], jnp.int32),
            jnp.asarray(tables.reshape(-1).numpy(), jnp.int32),
            jnp.asarray(main_pool.numpy()),
            index_pool,
            jnp.asarray(cos_sin.numpy()),
            jnp.asarray(k_norm_weight.numpy()),
            jnp.ones(2, jnp.float32),
            **_QSA_KWARGS,
            token_batch=16,
            block_tile=4,
        )
    out_t = torch.from_numpy(np.asarray(out_dec).copy())
    for r, n in enumerate(prefill_lens):
        expected = _reference(seqs[r], n, k_norm_weight, cos_sin)
        torch.testing.assert_close(out_t[r], expected, rtol=1e-5, atol=1e-5)


def test_qsa_position_rewind_rewrites_compressed_block_boundary() -> None:
    """Rewriting speculative draft positions across a compressed-block closing
    boundary `(pos + 1) % ratio == 0` replaces the pooled index key so later
    sparse queries match the accepted sequence."""
    torch.manual_seed(23)
    n_total = 20
    seq = {
        "q": torch.randn(n_total, H, D),
        "k": torch.randn(n_total, KVH, D),
        "v": torch.randn(n_total, KVH, D),
        "iq": torch.randn(n_total, IH, DI),
        "raw": torch.randn(n_total, DI),
    }
    k_norm_weight = torch.randn(DI) * 0.1
    cos_sin = _cos_sin(64)
    num_pages = 1 + PAGES_PER_REQ
    tables = torch.arange(1, num_pages, dtype=torch.int32)[None]
    main_pool = torch.zeros(num_pages, PAGE, 2 * KVH, 1, 128)
    index_pool = jnp.zeros((num_pages, PAGE, 2 * KVH, 1, 128), jnp.float32)

    def run_positions(
        pos_list: list[int],
        cur_seq: dict[str, torch.Tensor],
        cur_index_pool: jax.Array,
    ):
        _write_main(main_pool, tables, 0, pos_list, cur_seq)
        out, cur_index_pool = _qsa_attention(
            jnp.asarray(cur_seq["q"][pos_list].numpy()),
            jnp.asarray(cur_seq["iq"][pos_list].numpy()),
            jnp.asarray(cur_seq["raw"][pos_list].numpy()),
            jnp.asarray(torch.stack([_dense(cur_seq, p) for p in pos_list]).numpy()),
            jnp.asarray(pos_list, jnp.int32),
            jnp.asarray([0, len(pos_list)], jnp.int32),
            jnp.asarray([pos_list[-1] + 1], jnp.int32),
            jnp.asarray(tables.reshape(-1).numpy(), jnp.int32),
            jnp.asarray(main_pool.numpy()),
            cur_index_pool,
            jnp.asarray(cos_sin.numpy()),
            jnp.asarray(k_norm_weight.numpy()),
            jnp.ones(2, jnp.float32),
            **_QSA_KWARGS,
            token_batch=16,
            block_tile=4,
        )
        return torch.from_numpy(np.asarray(out).copy()), cur_index_pool

    # Bogus speculative draft tokens at positions 14..16 crossing block 3's
    # closing boundary at pos=15 ((15 + 1) % 4 == 0).
    bogus_seq = {k: v.clone() for k, v in seq.items()}
    bogus_seq["k"][14:17] = 5.0
    bogus_seq["v"][14:17] = -5.0
    bogus_seq["raw"][14:17] = 10.0

    with jax.default_matmul_precision("highest"):
        _, index_pool = run_positions(list(range(14)), seq, index_pool)
        _, index_pool = run_positions([14, 15, 16], bogus_seq, index_pool)
        # Rewind to pos 14 and overwrite [14, 15, 16, 17] with accepted tokens.
        out_rewound, index_pool = run_positions([14, 15, 16, 17], seq, index_pool)
        out_next, _ = run_positions([18, 19], seq, index_pool)

    for i, p in enumerate([14, 15, 16, 17]):
        expected = _reference(seq, p, k_norm_weight, cos_sin)
        torch.testing.assert_close(out_rewound[i], expected, rtol=1e-5, atol=1e-5)
    for i, p in enumerate([18, 19]):
        expected = _reference(seq, p, k_norm_weight, cos_sin)
        torch.testing.assert_close(out_next[i], expected, rtol=1e-5, atol=1e-5)
