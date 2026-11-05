# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi-K3.1's DSA on the prefill stages (CPU): the stage's synthetic
indexer, the window's index keys and their pages."""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from tools.megakernel import dsa_reference as ref
from vllm_torchtpu.megakernel.kimi_k3 import dsa

pytestmark = pytest.mark.cpu_test

BF16 = np.dtype(ml_dtypes.bfloat16)


def test_stage_synthetic_indexer_matches_the_decoders_generators():
    ordinals = (0, 5, 23)
    got = dsa.stage_synthetic_indexer("random", 7, ordinals, 1536, 7168, BF16)
    weight, bias = dsa.synthetic_key_norm(7, 24, BF16)
    for slot, m in enumerate(ordinals):
        for h in (0, 31):
            np.testing.assert_array_equal(
                got["dsa_wq"][slot][:, 128 * h : 128 * (h + 1)],
                dsa.synthetic_query_heads(7, m, 1536, BF16, heads=(h,)),
            )
            np.testing.assert_array_equal(
                got["dsa_wv"][slot][:, h],
                dsa.synthetic_weight_rows(7, m, 7168, BF16, heads=(h,))[0],
            )
        np.testing.assert_array_equal(
            got["dsa_wk"][slot], dsa.synthetic_key_projection(7, m, 7168, BF16)
        )
        np.testing.assert_array_equal(got["dsa_knorm"][slot][0], weight[m, 0])
        np.testing.assert_array_equal(got["dsa_knorm"][slot][1], bias[m, 0])
    probe = dsa.stage_synthetic_indexer("rope-probe", 7, ordinals, 1536, 7168, BF16)
    assert not probe["dsa_wk"].any()
    np.testing.assert_array_equal(probe["dsa_knorm"][:, 0], np.ones((3, 128), BF16))
    np.testing.assert_array_equal(probe["dsa_knorm"][:, 1], np.zeros((3, 128), BF16))
    with pytest.raises(ValueError, match="synthetic indexer"):
        dsa.stage_synthetic_indexer("zeros", 7, ordinals, 1536, 7168, BF16)


@pytest.mark.parametrize("rope_probe", [False, True])
def test_index_keys_match_the_reference(rope_probe):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache, dsa_prefill

    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((256, 7168)).astype(BF16))
    latent = jnp.asarray(rng.standard_normal((256, 576)).astype(BF16))
    mode = "rope-probe" if rope_probe else "random"
    fam = dsa.stage_synthetic_indexer(mode, 0, (3,), 1536, 7168, BF16)
    wk_t, knorm = jnp.asarray(fam["dsa_wk"][0]), jnp.asarray(fam["dsa_knorm"][0])
    got = dsa_prefill.index_keys(
        x, latent, wk_t, knorm, rope_probe=rope_probe, eps=1e-5
    )
    if rope_probe:
        projected = jnp.concatenate((latent[:, 512:576], -latent[:, 512:576]), axis=-1)
    else:
        projected = ref._bf16_dot(x, wk_t)
    want = dsa_cache.quantize_index_keys(
        ref.key_from_projection(projected, knorm[0], knorm[1], 1e-5)
    )
    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want[0]))
    np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))


def test_index_pages_lay_out_keys_as_the_decoders_commit():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache, dsa_prefill

    rng = np.random.default_rng(2)
    key_bytes = jnp.asarray(rng.integers(0, 256, (1024, 128), np.uint8))
    exponents = jnp.asarray(rng.integers(100, 140, (1024,), np.uint8))
    pages = dsa_prefill.index_pages(key_bytes, exponents, 512)
    cache = jnp.zeros((2, 1, 544, 128), jnp.uint8)
    want = dsa_cache.commit_index_rows(
        cache,
        key_bytes[:, None],
        exponents[:, None],
        jnp.asarray([[0, 1]] * 1024, jnp.int32),
        jnp.arange(1024, dtype=jnp.int32),
        member=0,
        group=1,
        global_pages=True,
    )
    np.testing.assert_array_equal(np.asarray(pages), np.asarray(want[:, 0]))


def test_window_keys_land_in_their_decode_pages_of_the_stage_cache():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache, dsa_prefill

    rng = np.random.default_rng(3)
    layers, page, split, blocks = 2, 256, 4, 3
    key_bytes = jnp.asarray(rng.integers(1, 256, (layers, 4 * page, 128), np.uint8))
    exponents = jnp.asarray(rng.integers(100, 140, (layers, 4 * page), np.uint8))
    table = jnp.asarray([9, 2, 4, 7], jnp.int32)  # decode pages, across blocks
    index = jnp.zeros((blocks, layers, split * (page + 32), 128), jnp.uint8)
    got = dsa_prefill.commit_index_pages(index, key_bytes, exponents, table, page)
    decode = jnp.zeros((blocks * split, layers, page + 32, 128), jnp.uint8)
    want = dsa_cache.commit_index_rows(
        decode,
        key_bytes.transpose(1, 0, 2),
        exponents.T,
        jnp.broadcast_to(table, (4 * page, 4)),
        jnp.arange(4 * page, dtype=jnp.int32),
        member=0,
        group=1,
        global_pages=True,
    )
    want = want.reshape(blocks, split, layers, page + 32, 128).transpose(0, 2, 1, 3, 4)
    np.testing.assert_array_equal(
        np.asarray(got), np.asarray(want).reshape(index.shape)
    )


def test_candidate_scores_match_the_reference():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache, dsa_prefill

    rng = np.random.default_rng(4)
    q, _ = dsa_cache.quantize_index_queries(
        jnp.asarray(rng.standard_normal((64, 32, 128)).astype(np.float32))
    )
    w = jnp.asarray(rng.standard_normal((64, 32)).astype(np.float32))
    raw = rng.standard_normal((3000, 128)).astype(BF16)
    keys = dsa_cache.dequantize_index_keys(
        *dsa_cache.quantize_index_keys(jnp.asarray(raw))
    )
    got = np.asarray(dsa_prefill.candidate_scores(q, w, keys.astype(jnp.bfloat16)))
    want = np.asarray(ref.scores(q, w, keys))
    rel = np.abs(got - want).max(axis=1) / np.maximum(np.abs(want).max(axis=1), 1e-30)
    assert rel.max() <= 1e-5


@pytest.mark.parametrize("k", [16, 2048])
def test_select_mask_is_the_reference_top_k_with_ties_to_the_lower_index(k):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    rng = np.random.default_rng(5)
    rows, cols = 32, 4096
    scores = rng.standard_normal((rows, cols)).astype(np.float32)
    lengths = rng.integers(1, cols + 1, rows)
    lengths[:4] = [k // 2, k, k + 1, cols]  # dense, exactly k, one over, full
    lengths[4] = 0  # no valid candidate
    for r in range(5, 10):  # ties straddling the k-th value
        order = np.argsort(-scores[r, : lengths[r]])
        if lengths[r] > k + 3:
            scores[r, order[k - 2 : k + 3]] = scores[r, order[k - 2]]
    scores[10, : cols // 2] = -np.inf  # -inf candidates still valid
    valid = np.arange(cols)[None] < lengths[:, None]
    got = np.asarray(
        dsa_prefill.select_mask(jnp.asarray(scores), jnp.asarray(valid), k)
    )
    masked = jnp.asarray(np.where(valid, scores, -np.inf))
    want = np.asarray(ref.select_topk(masked, jnp.asarray(lengths), k))
    for r in range(rows):
        expected = set(int(c) for c in want[r] if c >= 0)
        assert set(np.flatnonzero(got[r])) == expected, r
        assert got[r].sum() == min(k, lengths[r])


def test_bits_round_trip():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    mask = jnp.asarray(np.random.default_rng(6).random((8, 4096)) < 0.3)
    words = dsa_prefill.pack_bits(mask)
    assert words.shape == (8, 128) and words.dtype == jnp.uint32
    np.testing.assert_array_equal(
        np.asarray(dsa_prefill.unpack_bits(words, 4096)), np.asarray(mask)
    )


def test_order_keys_follow_the_float_order():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    values = [-np.inf, -3e38, -1.5, -1e-40, -0.0, 0.0, 1e-40, 1.5, 3e38, np.inf]
    keys = np.asarray(dsa_prefill.order_keys(jnp.asarray(values, jnp.float32)))
    steps = np.diff(keys.astype(np.int64))
    assert steps[4] == 0  # -0 and +0
    assert (np.delete(steps, 4) > 0).all()


@pytest.mark.parametrize("k", [16, 2048])
def test_select_mask_is_a_stable_sort_on_duplicate_scores(k):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    rng = np.random.default_rng(11)
    rows, cols = 24, 4096
    levels = np.asarray([-np.inf, -2.0, -0.5, -0.0, 0.0, 0.75, 1.0, 3.0], np.float32)
    scores = levels[rng.integers(0, len(levels), (rows, cols))]
    scores[:4] = levels[rng.integers(5, 7, (4, cols))]  # two values per row
    scores[4:8] = levels[rng.integers(3, 5, (4, cols))]  # zeros of both signs
    lengths = rng.integers(k // 2, cols + 1, rows)
    lengths[:8] = cols
    valid = np.arange(cols)[None] < lengths[:, None]
    got = np.asarray(
        dsa_prefill.select_mask(jnp.asarray(scores), jnp.asarray(valid), k)
    )
    for r in range(rows):
        order = np.argsort(-scores[r, : lengths[r]], kind="stable")[:k]
        assert set(np.flatnonzero(got[r])) == set(order.tolist()), r


def _query_inputs(rows=64):
    rng = np.random.default_rng(3)
    q_c = jnp.asarray(rng.standard_normal((rows, 1536)).astype(BF16))
    x = jnp.asarray(rng.standard_normal((rows, 7168)).astype(BF16))
    fam = dsa.stage_synthetic_indexer("random", 0, (3,), 1536, 7168, BF16)
    return q_c, x, jnp.asarray(fam["dsa_wq"][0]), jnp.asarray(fam["dsa_wv"][0])


def test_index_query_matches_the_reference():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    q_c, x, wq_t, wv_t = _query_inputs()
    got, want = (
        dsa_prefill.index_query(q_c, wq_t, x, wv_t),
        ref.index_query(q_c, wq_t, x, wv_t),
    )
    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want[0]))
    np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))


# The masked-flash fixture: two sequences in a 512-row window (rows 456+ pad),
# each continuing a cached prefix in a 1024-row arena.
TOKENS, ARENA, HEADS, COUNT = 512, 1024, 3, 2
OFFSETS, LENGTHS, BASES, CACHED = (0, 256), (256, 200), (0, 512), (300, 400)
LENGTH = OFFSETS[-1] + LENGTHS[-1]


@pytest.fixture
def interpret(monkeypatch):
    """Run the kernels' pallas_calls in TPU interpret mode on CPU."""
    import functools

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    monkeypatch.setattr(
        pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )


def _flash_inputs():
    rng = np.random.default_rng(7)
    shape = (HEADS, TOKENS)
    q = jnp.asarray((0.05 * rng.standard_normal((*shape, 256))).astype(BF16))
    key = jnp.asarray(rng.standard_normal((*shape, 256)).astype(BF16))
    value = jnp.asarray(rng.standard_normal((*shape, 128)).astype(BF16))
    arena = (
        jnp.asarray(rng.standard_normal((1, HEADS, ARENA, 256)).astype(BF16)),
        jnp.asarray(rng.standard_normal((1, HEADS, 128, ARENA)).astype(BF16)),
    )
    # offsets, lengths, checkpoints, arena bases, cached lengths
    descriptor = [*OFFSETS, *LENGTHS, 0, 0, *BASES, *CACHED]
    return q, key, value, arena, jnp.asarray(descriptor, jnp.int32)


def _reference_valid():
    """The flash kernel's mask in numpy: a row of sequence s sees arena rows
    [base_s, base_s + cached_s), then window rows [offset_s, row]."""
    valid = np.zeros((TOKENS, ARENA + TOKENS), bool)
    for s in range(COUNT):
        for row in range(OFFSETS[s], OFFSETS[s] + LENGTHS[s]):
            valid[row, BASES[s] : BASES[s] + CACHED[s]] = True
            valid[row, ARENA + OFFSETS[s] : ARENA + row + 1] = True
    return valid


def _reference_attention(q, key, value, arena, mask):
    """FP32 base-2 softmax attention (the kernel's queries come pre-scaled by
    SCALE_LOG2) over the [arena | window] columns that ``mask`` admits."""
    keys = np.concatenate(
        [np.asarray(arena[0][0], np.float32), np.asarray(key, np.float32)], axis=1
    )
    arena_values = np.swapaxes(np.asarray(arena[1][0], np.float32), 1, 2)
    values = np.concatenate([arena_values, np.asarray(value, np.float32)], axis=1)
    scores = np.einsum("htd,hcd->htc", np.asarray(q, np.float32), keys)
    scores = np.where(mask[None], scores, -np.inf)
    top = scores.max(axis=-1, keepdims=True)
    weights = np.exp2(scores - np.where(np.isfinite(top), top, 0.0))
    total = weights.sum(axis=-1, keepdims=True)
    attended = np.einsum("htc,hcd->htd", weights, values)
    return attended / np.where(total > 0, total, 1.0)


def _flash(inputs, **select):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k

    q, key, value, arena, segments = inputs
    length = jnp.asarray([LENGTH], jnp.int32)
    out = k.mla_flash_prefill_packed(
        q, key, value, length, segments, count=COUNT, arena=arena, **select
    )
    return np.asarray(out.astype(jnp.float32))


def test_candidate_valid_matches_the_kernels_mask():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    segments = _flash_inputs()[4]
    got = dsa_prefill.candidate_valid(
        segments,
        jnp.asarray([LENGTH], jnp.int32),
        count=COUNT,
        tokens=TOKENS,
        capacity=ARENA,
    )
    np.testing.assert_array_equal(np.asarray(got), _reference_valid())


def test_masked_flash_with_every_bit_set_is_the_dense_flash(interpret):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    inputs = _flash_inputs()
    dense = _flash(inputs)
    words = (TOKENS, (ARENA + TOKENS) // 32)
    every = jnp.full(words, 0xFFFFFFFF, jnp.uint32)
    valid = dsa_prefill.pack_bits(jnp.asarray(_reference_valid()))
    noise = jnp.asarray(np.random.default_rng(8).integers(0, 2**32, words, np.uint32))
    masked, skipped = jnp.asarray([0], jnp.int32), jnp.asarray([1], jnp.int32)
    np.testing.assert_array_equal(_flash(inputs, select=every, dense=masked), dense)
    np.testing.assert_array_equal(_flash(inputs, select=valid, dense=masked), dense)
    np.testing.assert_array_equal(_flash(inputs, select=noise, dense=skipped), dense)


def test_masked_flash_matches_masked_reference_attention(interpret):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    inputs = _flash_inputs()
    valid = _reference_valid()
    chosen = valid & (np.random.default_rng(9).random(valid.shape) < 0.3)
    rows = np.arange(LENGTH)
    chosen[rows, ARENA + rows] = True  # every real row keeps its own key
    reference = _reference_attention(*inputs[:4], valid)
    dense_error = np.abs(_flash(inputs) - reference).max()
    select = dsa_prefill.pack_bits(jnp.asarray(chosen))
    got = _flash(inputs, select=select, dense=jnp.asarray([0], jnp.int32))
    error = np.abs(got - _reference_attention(*inputs[:4], chosen)).max()
    assert error <= 2 * dense_error, (error, dense_error)
    assert not got[:, LENGTH:].any()  # padding rows stay zero


def test_masked_flash_selects_in_streamed_blocks(interpret):
    """One 1400-row sequence in a 1536-row window over 1500 cached rows of a
    2048-row arena: its later query tiles stream whole window and arena key
    blocks. Rows 1100+ select no key of the edge block or the first streamed
    one, so the selection enters the streamed blocks with no key yet."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k

    tokens, arena_rows, rows, cached = 1536, 2048, 1400, 1500
    rng = np.random.default_rng(10)
    q = jnp.asarray((0.05 * rng.standard_normal((1, tokens, 256))).astype(BF16))
    key = jnp.asarray(rng.standard_normal((1, tokens, 256)).astype(BF16))
    value = jnp.asarray(rng.standard_normal((1, tokens, 128)).astype(BF16))
    arena = (
        jnp.asarray(rng.standard_normal((1, 1, arena_rows, 256)).astype(BF16)),
        jnp.asarray(rng.standard_normal((1, 1, 128, arena_rows)).astype(BF16)),
    )
    segments = jnp.asarray([0, rows, 0, 0, cached], jnp.int32)
    length = jnp.asarray([rows], jnp.int32)
    valid = np.asarray(
        dsa_prefill.candidate_valid(
            segments, length, count=1, tokens=tokens, capacity=arena_rows
        )
    )
    chosen = valid & (rng.random(valid.shape) < 0.3)
    chosen[1100:, :512] = chosen[1100:, 1024:arena_rows] = False
    chosen[np.arange(rows), arena_rows + np.arange(rows)] = True

    def flash(**select):
        out = k.mla_flash_prefill_packed(
            q, key, value, length, segments, count=1, arena=arena, **select
        )
        return np.asarray(out.astype(jnp.float32))

    words = (tokens, (arena_rows + tokens) // 32)
    every = jnp.full(words, 0xFFFFFFFF, jnp.uint32)
    masked = jnp.asarray([0], jnp.int32)
    np.testing.assert_array_equal(flash(select=every, dense=masked), flash())
    got = flash(select=dsa_prefill.pack_bits(jnp.asarray(chosen)), dense=masked)
    reference = _reference_attention(q, key, value, arena, chosen)
    # within a BF16 step of the largest output (BF16 probabilities and output)
    assert np.abs(got - reference).max() <= 2**-7 * np.abs(reference).max()


@pytest.mark.parametrize(
    "cached, topk", [((300, 400), 128), ((0, 0), 64), ((300, 400), 2048)]
)
def test_stage_selection_is_each_window_rows_top_k(monkeypatch, cached, topk):
    """``prefill_stage._dsa_selection`` with the XLA selection
    (``KernelConfig.dsa_prefill_select`` ``xla``) on the stage mesh (the
    flash fixture's window over a 1536-row arena whose last two pages were
    never written) against the window's rows scored and selected in one
    piece: with a cached prefix, without one (the window's keys only), and
    dense."""
    import jax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.kimi_k3.kernels import (
        dsa_cache,
        dsa_prefill,
        prefill_stage,
    )
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k

    selection_config = k._config().replace(dsa_prefill_select="xla")
    monkeypatch.setattr(k, "_config", lambda: selection_config)
    n, page, capacity = 8, 256, 1536
    rows = TOKENS // n
    stripe = k.window_stripe(rows)
    rng = np.random.default_rng(12)
    q_c, x, wq_t, wv_t = _query_inputs(TOKENS)
    window_keys = dsa_cache.dequantize_index_keys(
        *dsa_cache.quantize_index_keys(
            jnp.asarray(rng.standard_normal((TOKENS, 128)).astype(BF16))
        )
    )
    raw = jnp.asarray(rng.standard_normal((ARENA, 128)).astype(BF16))
    pages = dsa_prefill.index_pages(*dsa_cache.quantize_index_keys(raw), page)
    slots = jnp.concatenate((jnp.full((3, page + 32, 128), 0x7F, jnp.uint8), pages))
    order = jnp.asarray([ARENA // page, 3, 4, 5, 6, 0, 0], jnp.int32)
    segments = jnp.asarray([*OFFSETS, *LENGTHS, 0, 0, *BASES, *cached], jnp.int32)
    length = jnp.asarray([LENGTH], jnp.int32)

    def local(q_c, x):
        def in_window(values):
            gathered = jax.lax.all_gather(values, "k3_sp")
            gathered = gathered.reshape(n, rows // stripe, stripe, *values.shape[1:])
            return gathered.swapaxes(0, 1).reshape(TOKENS, *values.shape[1:])

        positions = k.sp_token_index(
            jax.lax.axis_index("k3_sp"), jnp.arange(rows), stripe, n
        )
        out = prefill_stage._dsa_selection(
            dsa.DsaConfig(index_topk=topk),
            {"dsa_wq": wq_t, "dsa_wv": wv_t},
            x,
            q_c,
            window_keys,
            slots,
            order,
            segments,
            length,
            positions,
            in_window,
            segments=COUNT,
            seq_len=TOKENS,
            capacity=capacity,
            ranks=n,
            stripe=stripe,
        )
        return out["select"], out["dense"]

    mesh = Mesh(np.asarray(jax.devices()[:n]), ("k3_sp",))
    ranked = [
        jax.device_put(a[k.sp_order(TOKENS, n)], NamedSharding(mesh, P("k3_sp")))
        for a in (q_c, x)
    ]
    words, dense = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("k3_sp"),) * 2,
            out_specs=(P(), P()),
            check_vma=False,
        )
    )(*ranked)
    candidates = np.asarray(cached) + np.asarray(LENGTHS)
    assert int(dense[0]) == (candidates <= topk).all()
    if dense[0]:
        assert not np.asarray(words).any()
        return
    arena = dsa_cache.dequantize_index_keys(*dsa_cache.quantize_index_keys(raw))
    keys = jnp.concatenate((arena, jnp.zeros((capacity - ARENA, 128)), window_keys))
    q, w = dsa_prefill.index_query(q_c, wq_t, x, wv_t)
    valid = dsa_prefill.candidate_valid(
        segments, length, count=COUNT, tokens=TOKENS, capacity=capacity
    )
    mask = dsa_prefill.select_mask(
        dsa_prefill.candidate_scores(q, w, keys), valid, topk
    )
    np.testing.assert_array_equal(
        np.asarray(words), np.asarray(dsa_prefill.pack_bits(mask))
    )
    assert np.asarray(mask).sum(1).max() == topk


@pytest.mark.parametrize("k31", [False, True])
def test_stage_group_and_final_ops_trace(monkeypatch, k31):
    """The stage's group and final ops trace for Kimi-K3 and Kimi-K3.1 on the
    stage mesh, the real MLA layer (index-key cache and slot passed through)
    among stand-in KDA, MoE, mixture and logits layers."""
    import types

    import jax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_stage

    n, window, count, capacity, page = 8, 512, 2, 4096, 256
    cycle = [(x, y, c) for x, y in k.HOST_CHIP_CYCLE for c in (0, 1)]
    cores = [
        types.SimpleNamespace(coords=(x, y, 0), core_on_chip=c) for x, y, c in cycle
    ]
    monkeypatch.setattr(
        k, "mesh_ring_tables", lambda mesh, token_ranks=None: k.stage_ring_tables(cores)
    )
    monkeypatch.setattr(
        k, "make_mixture_norm_sp", lambda mesh, **_: lambda prefix, *_: prefix
    )
    monkeypatch.setattr(
        prefill_stage,
        "make_kda_layer_stage",
        lambda mesh, **_: lambda prefix, b, r, m, f, rec, hist, *_: (
            prefix,
            rec,
            hist,
            rec,
            hist,
        ),
    )
    monkeypatch.setattr(
        prefill_stage,
        "_stage_moe",
        lambda *_: lambda x, *_: (x, jnp.zeros((n,), jnp.int32)),
    )
    monkeypatch.setattr(
        k,
        "make_prefill_logits_sp",
        lambda mesh, **_: lambda *_: jnp.zeros((n, count, 16), jnp.float32),
    )
    mesh = Mesh(np.asarray(jax.devices()[:n]), ("k3_sp",))
    sharded = NamedSharding(mesh, P("k3_sp"))

    def spec(shape, dtype=jnp.bfloat16, sharding=sharded):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)

    def scalar(values):
        return jax.ShapeDtypeStruct(
            (values,), jnp.int32, sharding=NamedSharding(mesh, P())
        )

    config = dsa.DsaConfig() if k31 else None
    rows, layers, blocks = window // n, 6, 6
    mla = {
        "m_qa": (7168, 1536),
        "m_qb": (1536, 3072),
        "m_ka": (7168, 576),
        "m_kb": (512, 3072),
        "m_gate": (7168, 12288),
        "m_qnorm": (1, 1536),
        "m_knorm": (1, 512),
        "m_o": (12288, 7168),
    }
    if k31:
        mla.update(
            dsa_wk=(7168, 128),
            dsa_knorm=(2, 128),
            dsa_wq=(1536, 4096),
            dsa_wv=(7168, 32),
        )
    mla = {name: spec((n, layers, *shape)) for name, shape in mla.items()}
    states = [spec((n, layers, window, 576))]
    index_cache = ()
    if k31:
        states += [
            spec((n, layers, window, 128), jnp.uint8),
            spec((n, layers, window), jnp.uint8),
        ]
        index_cache = (spec((n * blocks, layers, page + 32, 128), jnp.uint8), scalar(1))
    common = {
        name: spec((n, 24, 1, 7168), jnp.float32) for name in ("attn_res", "mlp_res")
    }
    common.update({name: spec((n, 24, 1, 7168)) for name in ("attn_norm", "ffn_norm")})
    arena = k.CpPrefix(
        (spec((n * blocks, 128, 2, 640)),),
        spec((n, 1 + capacity // n // page + count), jnp.int32),
        spec((n, 2 * count), jnp.int32),
    )
    prefix, snapshots = spec((n, rows, 7168)), spec((n, 8, rows, 7168))
    kda = [spec((n, 18, count, 4), jnp.float32), spec((n, 18, count, 3))] * 2
    io = {name: spec((n, 1, 128)) for name in ("output_res", "final_norm", "lm_head")}
    for stage, final in ((0, False), (3, True)):
        options = dict(
            seq_len=window,
            segments=count,
            prefix_capacity=capacity,
            slots=prefill_stage.stage_plan(stage)["slots"],
            dsa=config,
        )
        if final:
            op = prefill_stage.make_stage_final(mesh, **options)
            out = jax.eval_shape(
                op,
                prefix,
                snapshots,
                common,
                mla,
                {},
                states,
                scalar(1),
                scalar(5 * count),
                arena,
                io,
                index_cache,
            )
        else:
            op = prefill_stage.make_stage_group4(mesh, **options)
            out = jax.eval_shape(
                op,
                prefix,
                snapshots,
                common,
                {},
                mla,
                {},
                kda,
                states,
                scalar(1),
                scalar(1),
                scalar(5 * count),
                arena,
                index_cache,
            )
        assert [s.shape for s in out[-2 if final else 3]] == [s.shape for s in states]


@pytest.mark.parametrize("dtype", [BF16, np.dtype(ml_dtypes.float8_e4m3fn)])
def test_arena_order_check_finds_misplaced_tokens(dtype):
    """``check arena-order``'s host side: an arena in page token order passes,
    two tokens swapped within a page do not."""
    from tools.megakernel import __main__ as tool_main
    from tools.megakernel import check

    page_rows, packing = 128, 4 if dtype.itemsize == 1 else 2
    pages = check.arena_order_pages(page_rows, packing, dtype)
    order = np.asarray([6, 5, 2, 7, 0, 3, 6, 1, 4], np.int32)
    page = page_rows * packing
    arena = pages[order[1:]].reshape(-1, 640)[:, 512:516]
    assert check.arena_order_mismatches(arena, order, page) == 0
    arena[[page + 3, page + 9]] = arena[[page + 9, page + 3]]
    assert check.arena_order_mismatches(arena, order, page) == 2
    assert (
        tool_main.parser().parse_args(["check", "arena-order"]).check == "arena-order"
    )


def test_masked_flash_keeps_a_first_selected_block_far_below_zero(interpret):
    """``test_masked_flash_selects_in_streamed_blocks``' layout, rows 1100+
    selecting only arena rows 512..1023 (the first streamed block they see)
    at base-2 scores of -200: their probabilities against 0 would underflow,
    so the block must still count, exactly."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k

    tokens, arena_rows, rows, cached = 1536, 2048, 1400, 1500
    rng = np.random.default_rng(13)
    q = (0.05 * rng.standard_normal((1, tokens, 256))).astype(np.float32)
    q[0, 1100:] = 0
    q[0, 1100:, 0] = -200
    arena_keys = rng.standard_normal((1, 1, arena_rows, 256)).astype(np.float32)
    arena_keys[..., 512:1024, 0] = 1
    q, arena_keys = jnp.asarray(q.astype(BF16)), jnp.asarray(arena_keys.astype(BF16))
    key = jnp.asarray(rng.standard_normal((1, tokens, 256)).astype(BF16))
    value = jnp.asarray(rng.standard_normal((1, tokens, 128)).astype(BF16))
    arena_values = jnp.asarray(
        rng.standard_normal((1, 1, 128, arena_rows)).astype(BF16)
    )
    arena = (arena_keys, arena_values)
    segments = jnp.asarray([0, rows, 0, 0, cached], jnp.int32)
    length = jnp.asarray([rows], jnp.int32)
    valid = np.asarray(
        dsa_prefill.candidate_valid(
            segments, length, count=1, tokens=tokens, capacity=arena_rows
        )
    )
    chosen = valid & (rng.random(valid.shape) < 0.3)
    chosen[np.arange(rows), arena_rows + np.arange(rows)] = True
    chosen[1100:] = False
    chosen[1100:rows, 512:1024] = valid[1100:rows, 512:1024]
    got = k.mla_flash_prefill_packed(
        q,
        key,
        value,
        length,
        segments,
        count=1,
        arena=arena,
        select=dsa_prefill.pack_bits(jnp.asarray(chosen)),
        dense=jnp.asarray([0], jnp.int32),
    )
    got = np.asarray(got.astype(jnp.float32))
    reference = _reference_attention(q, key, value, arena, chosen)
    assert np.abs(got - reference).max() <= 2**-7 * np.abs(reference).max()


@pytest.mark.parametrize(
    "dots, heads, bisect, block",
    [
        ("bf16", "vpu", "full", 8),
        ("fp8", "vpu", "full", 8),
        ("bf16", "mxu", "full", 8),
        ("bf16", "vpu", "narrow", 8),
        ("bf16", "vpu", "full", 16),
        ("bf16", "vpu", "narrow", 16),
        ("bf16", "butterfly", "full", 8),
        ("bf16", "butterfly", "narrow", 16),
        ("bf16", "butterfly", "lanes", 8),
        ("bf16", "vpu", "lanes", 16),
        ("bf16", "major", "lanes", 8),
        ("bf16", "bf16", "lanes", 16),
        ("bf16", "butterfly", "lanes", 32),
        ("bf16", "major", "lanes", 32),
    ],
)
def test_select_words_is_each_rows_top_k(
    interpret, monkeypatch, dots, heads, bisect, block
):
    """``dsa_prefill_select.select_words`` (TPU interpret mode) for 32 rows of
    two sequences, each continuing a cached arena prefix of whole pages, against
    ``dsa_prefill.select_mask`` over the same candidates, under each kernel
    switch: exact with BF16 index-score products and the VPU head sums (8- or
    16-row steps, full or narrowed threshold search); with FP8 x FP8 products
    or the MXU head sums (FP32 sums in another order) at most one near-tie swap
    per row."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import (
        dsa_cache,
        dsa_prefill,
    )
    from vllm_torchtpu.megakernel.kimi_k3.kernels import (
        dsa_prefill_select as sel,
    )

    config = sel._config().replace(
        dsa_index_dots=dots,
        dsa_head_sums=heads,
        dsa_bisect=bisect,
        dsa_select_rows=block,
    )
    monkeypatch.setattr(sel, "_config", lambda: config)
    exact = dots == "bf16" and heads in ("vpu", "butterfly")
    tokens, capacity, page, topk, count = 512, 4096, 512, 64, 2
    offsets, lengths, bases, cached = (0, 256), (256, 200), (0, 2048), (1536, 512)
    rng = np.random.default_rng(21)
    # Two 16-row blocks, one per sequence; a 32-row block stays in the first
    # (the kernel's blocks never straddle sequences).
    first = 224 if block == 32 else 240
    positions = jnp.arange(first, first + 32, dtype=jnp.int32)
    q_c, x, wq_t, wv_t = _query_inputs(32)
    q, w = dsa_prefill.index_query(q_c, wq_t, x, wv_t)
    raw = jnp.asarray(rng.standard_normal((capacity, 128)).astype(BF16))
    arena_bytes, arena_exponents = dsa_cache.quantize_index_keys(raw)
    pages = dsa_prefill.index_pages(arena_bytes, arena_exponents, page)
    order = jnp.asarray([capacity // page, *range(capacity // page)], jnp.int32)
    window_keys = dsa_cache.dequantize_index_keys(
        *dsa_cache.quantize_index_keys(
            jnp.asarray(rng.standard_normal((tokens, 128)).astype(BF16))
        )
    )
    segments = jnp.asarray([*offsets, *lengths, 0, 0, *bases, *cached], jnp.int32)
    length = jnp.asarray([offsets[-1] + lengths[-1]], jnp.int32)
    words = sel.select_words(
        q,
        w,
        window_keys,
        pages,
        order,
        segments,
        length,
        positions,
        count=count,
        capacity=capacity,
        topk=topk,
    )
    keys = jnp.concatenate(
        (dsa_cache.dequantize_index_keys(arena_bytes, arena_exponents), window_keys)
    )
    valid = dsa_prefill.candidate_valid(
        segments, length, count=count, tokens=tokens, capacity=capacity, rows=positions
    )
    mask = dsa_prefill.select_mask(
        dsa_prefill.candidate_scores(q, w, keys), valid, topk
    )
    want = dsa_prefill.pack_bits(mask)
    columns, counts = sel.columns_from_words(
        words, segments, count=count, capacity=capacity, topk=topk
    )
    assert (np.asarray(counts) == topk).all()
    if exact:
        for r in range(len(positions)):
            np.testing.assert_array_equal(
                np.asarray(columns[r]), np.flatnonzero(np.asarray(mask[r]))
            )
    total = (
        bases[-1] + cached[-1]
    ) // 32  # words above the cached total stay unwritten
    got = np.concatenate(
        (
            np.asarray(words)[:, :total],
            np.asarray(words)[:, capacity // 32 : (capacity + tokens) // 32],
        ),
        axis=1,
    )
    want = np.concatenate(
        (np.asarray(want)[:, :total], np.asarray(want)[:, capacity // 32 :]), axis=1
    )
    bits = np.unpackbits(got.view(np.uint8), axis=1).sum(1)
    assert (bits == topk).all()
    differing = np.unpackbits((got ^ want).view(np.uint8), axis=1).sum(1)
    # FP32 orders other than the decoder's: near-tie swaps; BF16 head sums
    # (SM100-class): up to an eighth of the selection.
    bound = 0 if exact else 2 * topk // 8 if heads == "bf16" else 2
    assert differing.max() <= bound, differing


@pytest.mark.parametrize("topk", [256, 4096])
def test_select_words_per_lane_threshold_is_exact(interpret, monkeypatch, topk):
    """``dsa_bisect="lanes"`` over a 12288-token cached prefix (96 columns a
    lane): exact both when every lane's 32 largest keys hold the top ``topk``
    (256) and when some lane holds more of it than it keeps (4096, the full
    search)."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import (
        dsa_cache,
        dsa_prefill,
    )
    from vllm_torchtpu.megakernel.kimi_k3.kernels import (
        dsa_prefill_select as sel,
    )

    config = sel._config().replace(dsa_bisect="lanes", dsa_head_sums="butterfly")
    monkeypatch.setattr(sel, "_config", lambda: config)
    tokens, capacity, page, count = 512, 16384, 512, 1
    cached = 12288
    rng = np.random.default_rng(5)
    positions = jnp.arange(200, 216, dtype=jnp.int32)
    q_c, x, wq_t, wv_t = _query_inputs(16)
    q, w = dsa_prefill.index_query(q_c, wq_t, x, wv_t)
    raw = jnp.asarray(rng.standard_normal((capacity, 128)).astype(BF16))
    arena_bytes, arena_exponents = dsa_cache.quantize_index_keys(raw)
    pages = dsa_prefill.index_pages(arena_bytes, arena_exponents, page)
    order = jnp.asarray(
        [capacity // page, *rng.permutation(capacity // page)], jnp.int32
    )
    window_keys = dsa_cache.dequantize_index_keys(
        *dsa_cache.quantize_index_keys(
            jnp.asarray(rng.standard_normal((tokens, 128)).astype(BF16))
        )
    )
    segments = jnp.asarray([0, tokens, 0, 0, cached], jnp.int32)
    length = jnp.asarray([tokens], jnp.int32)
    words = sel.select_words(
        q,
        w,
        window_keys,
        pages,
        order,
        segments,
        length,
        positions,
        count=count,
        capacity=capacity,
        topk=topk,
    )
    held = np.asarray(order)[1:]
    arena = dsa_cache.dequantize_index_keys(
        jnp.asarray(np.asarray(pages)[held, :page].reshape(capacity, 128)),
        jnp.asarray(
            np.asarray(pages)[held, page : page + page // 128].reshape(capacity)
        ),
    )
    keys = jnp.concatenate((arena, window_keys))
    valid = dsa_prefill.candidate_valid(
        segments, length, count=count, tokens=tokens, capacity=capacity, rows=positions
    )
    scores = dsa_prefill.candidate_scores(q, w, keys)
    mask = np.asarray(dsa_prefill.select_mask(scores, valid, topk))
    chosen = np.where(mask[:, :cached], 1, 0).reshape(len(positions), -1, 128).sum(1)
    assert (chosen.max() > 32) == (topk == 4096), chosen.max()
    columns, counts = sel.columns_from_words(
        words, segments, count=count, capacity=capacity, topk=topk
    )
    assert (np.asarray(counts) == topk).all()
    for r in range(len(positions)):
        np.testing.assert_array_equal(np.asarray(columns[r]), np.flatnonzero(mask[r]))
    assert not np.asarray(words)[
        :, cached // 32 : capacity // 32
    ].any()  # above the cached total: zero


def test_select_words_capacity_buckets_select_as_sixteen_rows(interpret, monkeypatch):
    """With a small VMEM budget, select_words splits a 131072-column capacity
    into row-step buckets (32, 16 and 8 rows a step, each holding less
    context, its word stage too) and switches on the window's arena total;
    every bucket's words, the zeros above the cached total included, equal
    the one 16-row-a-step call's (butterfly head sums, lanes)."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import (
        dsa_cache,
        dsa_prefill,
    )
    from vllm_torchtpu.megakernel.kimi_k3.kernels import (
        dsa_prefill_select as sel,
    )

    tokens, capacity, page, topk = 512, 131072, 512, 256
    window = -(-tokens // sel.GROUP) * sel.GROUP
    for fit in range(
        capacity - sel.GROUP, 0, -sel.GROUP
    ):  # a budget giving three buckets
        stage = sel.stage_columns(capacity, fit, tokens)
        budget = sel.vmem_bytes(16, fit + 2 * sel.GROUP + window, tokens, page, stage)
        monkeypatch.setattr(sel, "VMEM_BUDGET", budget)
        variants = sel.row_steps(64, capacity, tokens, page, 32)
        if [v[0] for v in variants] == [32, 16, 8]:
            break
    assert [v[0] for v in variants] == [32, 16, 8], variants
    rng = np.random.default_rng(9)
    q_c, x, wq_t, wv_t = _query_inputs(64)
    q, w = dsa_prefill.index_query(q_c, wq_t, x, wv_t)
    raw = jnp.asarray(rng.standard_normal((capacity, 128)).astype(BF16))
    pages = dsa_prefill.index_pages(*dsa_cache.quantize_index_keys(raw), page)
    order = jnp.asarray(
        [capacity // page, *rng.permutation(capacity // page)], jnp.int32
    )
    window_keys = dsa_cache.dequantize_index_keys(
        *dsa_cache.quantize_index_keys(
            jnp.asarray(rng.standard_normal((tokens, 128)).astype(BF16))
        )
    )
    length = jnp.asarray([tokens], jnp.int32)
    positions = jnp.arange(160, 224, dtype=jnp.int32)
    base = sel._config()
    lows = [0] + [v[1] for v in variants[:-1]]
    for (rows, fit), low in zip(variants, lows, strict=True):
        cached = (low + fit) // 2 // page * page  # inside this bucket
        segments = jnp.asarray([0, tokens, 0, 0, cached], jnp.int32)
        words = {}
        for most, limit in ((32, budget), (16, 2**40)):
            config = base.replace(
                dsa_head_sums="butterfly", dsa_bisect="lanes", dsa_select_rows=most
            )
            monkeypatch.setattr(sel, "_config", lambda config=config: config)
            monkeypatch.setattr(sel, "VMEM_BUDGET", limit)
            words[most] = np.asarray(
                sel.select_words(
                    q,
                    w,
                    window_keys,
                    pages,
                    order,
                    segments,
                    length,
                    positions,
                    count=1,
                    capacity=capacity,
                    topk=topk,
                )
            )
        np.testing.assert_array_equal(words[32], words[16], err_msg=f"{rows} rows")
        assert not words[32][:, -(-cached // sel.GROUP) * 128 : capacity // 32].any()


def test_selected_columns_list_each_rows_selection():
    """``columns_from_words``: each row's set bits as ascending columns (the
    arena words above the cached total, unwritten by the kernel, ignored);
    ``candidate_columns``: a dense window's rows list every candidate, as
    ``dsa_prefill.candidate_valid`` admits them."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill
    from vllm_torchtpu.megakernel.kimi_k3.kernels import (
        dsa_prefill_select as sel,
    )

    tokens, capacity, topk, count = 512, 8192, 96, 2
    offsets, lengths, bases, cached = (0, 256), (256, 200), (0, 2048), (1536, 512)
    segments = jnp.asarray([*offsets, *lengths, 0, 0, *bases, *cached], jnp.int32)
    rng = np.random.default_rng(4)
    width = sel.word_columns(capacity, tokens)
    total_words = 4096 // 32  # the cached total, 2560, rounded up to a GROUP
    bits = rng.random((6, width * 32)) < 0.01
    bits[:, total_words * 32 : capacity] = True  # unwritten words: garbage
    words = np.packbits(bits.reshape(6, width, 32)[..., ::-1], axis=-1).view(">u4")
    words = jnp.asarray(words.reshape(6, width).astype(np.uint32))
    got, counts = sel.columns_from_words(
        words, segments, count=count, capacity=capacity, topk=topk
    )
    for r in range(6):
        want = np.flatnonzero(
            np.where(np.arange(width * 32) < total_words * 32, bits[r], False)
            | np.where(np.arange(width * 32) >= capacity, bits[r], False)
        )
        assert int(counts[r]) == len(want)
        np.testing.assert_array_equal(
            np.asarray(got[r, : min(len(want), topk)]), want[:topk]
        )
        assert not np.asarray(got[r, len(want) :]).any()
    length = jnp.asarray([offsets[-1] + lengths[-1]], jnp.int32)
    positions = jnp.asarray([0, 7, 255, 256, 300, 455, 456, 500], jnp.int32)
    dense = jnp.asarray([*offsets, *lengths, 0, 0, *bases, 16, 32], jnp.int32)
    got, counts = sel.candidate_columns(
        dense, length, positions, count=count, capacity=capacity, topk=512
    )
    valid = dsa_prefill.candidate_valid(
        dense, length, count=count, tokens=tokens, capacity=capacity, rows=positions
    )
    for i in range(len(positions)):
        want = np.flatnonzero(np.asarray(valid[i]))
        assert int(counts[i]) == len(want)
        np.testing.assert_array_equal(np.asarray(got[i, : len(want)]), want)


@pytest.mark.parametrize("block", [8, 16])
def test_butterfly_head_sums_are_bitwise_the_per_row_sums(interpret, block):
    """``_head_sums(butterfly=True)`` (every row's sublane reduction in one
    shared roll tree) equals the per-row reduction bit for bit."""

    from jax.experimental import pallas as pl

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill_select as sel

    rng = np.random.default_rng(block)
    queries = jnp.asarray(rng.standard_normal((block * 32, 128)).astype(BF16))
    weights = jnp.asarray(rng.standard_normal((block * 32, 128)).astype(np.float32))
    keys = jnp.asarray(rng.standard_normal((128, 128)).astype(BF16))

    def run(butterfly):
        def kernel(q_ref, w_ref, k_ref, o_ref):
            o_ref[...] = sel._head_sums(
                q_ref[...], w_ref[...], k_ref[...], block, butterfly=butterfly
            )

        return np.asarray(
            pl.pallas_call(
                kernel, out_shape=jax.ShapeDtypeStruct((block, 128), jnp.float32)
            )(queries, weights, keys)
        )

    import jax

    np.testing.assert_array_equal(run(True).view(np.int32), run(False).view(np.int32))
