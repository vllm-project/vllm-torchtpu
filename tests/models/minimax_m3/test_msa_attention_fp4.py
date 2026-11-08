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
"""MSA sparse attention on 4-bit caches (interpret mode) against the dense
reference over the dequantized cache (`msa_fp4`)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from tests.models.minimax_m3.test_msa_attention import _reference
from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4
from vllm_torchtpu.models.minimax_m3.kernels.msa_attention import msa_sparse_attention
from vllm_torchtpu.models.minimax_m3.kernels.msa_attention_lse import (
    msa_sparse_attention_lse,
    msa_sparse_attention_lse_heads,
)

HEADS = 16
HEAD_DIM = 128
PAGE = 128
PAGES = 10
TOPK = 4


def _quantized_heads(rng, kv_heads):
    """Random K/V of `kv_heads` heads quantized per (token, head): nibbles
    ``[pages, S, 2D G]``, K / V scales ``[G][pages, S]`` and the dequantized
    rows ``[G][pages, S, 2D]`` the kernel must match."""
    nib, ks, vs, deq = [], [], [], []
    for _ in range(kv_heads):
        k = jnp.asarray(rng.standard_normal((PAGES, PAGE, HEAD_DIM)), jnp.float32)
        v = jnp.asarray(rng.standard_normal((PAGES, PAGE, HEAD_DIM)), jnp.float32)
        (a, sa), (b, sb) = msa_fp4.quantize(k), msa_fp4.quantize(v)
        nib += [a, b]
        ks.append(np.asarray(sa))
        vs.append(np.asarray(sb))
        deq.append(
            np.concatenate(
                [
                    np.asarray(msa_fp4.dequantize(a, sa)),
                    np.asarray(msa_fp4.dequantize(b, sb)),
                ],
                axis=-1,
            )
        )
    return jnp.concatenate(nib, axis=-1), ks, vs, deq


def _scale_lanes(ks, vs):
    """Folded scale row ``[pages, 2 S G]``: lane ``j * S + s`` holds scale
    ``j`` of slot ``s``, ``j = 2h`` / ``2h + 1`` the K / V scales of head h."""
    return np.concatenate([x for h in range(len(ks)) for x in (ks[h], vs[h])], axis=-1)


def _fp4_cache(rng, poison_page=None, swap_kv=False):
    """One KV head: the folded uint8 cache and the dequantized rows ``[pages,
    S, 2D]``. With `poison_page`, the scales of its slots 64.. are NaN (as an
    unwritten local-page tail may hold), which the kernel must never let
    through. `swap_kv` stores V scales where K scales belong and vice versa
    (a positive control the reference must catch)."""
    nib, ks, vs, deq = _quantized_heads(rng, 1)
    lanes = _scale_lanes(vs, ks) if swap_kv else _scale_lanes(ks, vs)
    if poison_page is not None:
        lanes[poison_page, 64:PAGE] = np.nan
        lanes[poison_page, PAGE + 64 :] = np.nan
    return msa_fp4.pack_rows(nib, scales=jnp.asarray(lanes)), deq[0]


def _batch(rng, num_tokens, num_real, poison_page):
    selected = np.full((num_tokens, TOPK), -1, np.int32)
    local_page = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    others = np.array([p for p in range(PAGES) if p != poison_page])
    for t in range(num_real):
        pages = rng.choice(others, rng.integers(1, TOPK + 1), replace=False)
        selected[t, : len(pages)] = pages
        local_page[t] = pages[0]
        local_limit[t] = rng.integers(1, PAGE + 1)
    # Only token 0 reads the poisoned page, as its local page, up to slot 40.
    selected[0, 0], local_page[0], local_limit[0] = poison_page, poison_page, 40
    return selected, local_page, local_limit


def _attend(cache, q, selected, local_page, local_limit, block_q):
    return msa_sparse_attention(
        q,
        cache,
        jnp.asarray(selected),
        jnp.asarray(local_page),
        jnp.asarray(local_limit),
        sm_scale=HEAD_DIM**-0.5,
        block_q=block_q,
        interpret=True,
    )


@pytest.mark.parametrize("block_q", [16, 64])
def test_matches_dequantized_reference(block_q):
    rng = np.random.default_rng(block_q)
    poison = 3
    cache, rows = _fp4_cache(rng, poison_page=poison)
    num_tokens, num_real = 37, 33
    selected, local_page, local_limit = _batch(rng, num_tokens, num_real, poison)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    out = _attend(cache, q, selected, local_page, local_limit, block_q)
    expected = _reference(
        np.asarray(q.astype(jnp.float32)), rows, selected, local_page, local_limit
    )
    got = np.asarray(out.astype(jnp.float32))
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, atol=3e-2)


@pytest.mark.parametrize("mutation", ["swap_kv", "ones"])
def test_reference_catches_misread_scales(mutation):
    """Positive control: the same data with K and V scales swapped, or every
    scale 1, must fail the tolerance the real layout passes."""
    rng = np.random.default_rng(11)
    num_tokens, num_real = 16, 16
    selected, local_page, local_limit = _batch(rng, num_tokens, num_real, 1)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    expected = None
    for arm in ("real", mutation):
        arm_rng = np.random.default_rng(12)  # same K/V in both arms
        cache, rows = _fp4_cache(arm_rng, swap_kv=arm == "swap_kv")
        if arm == "ones":
            cache = msa_fp4.pack_rows(
                msa_fp4.unpack_rows(cache), scales=jnp.ones((PAGES, 2 * PAGE))
            )
        if expected is None:
            expected = _reference(
                np.asarray(q.astype(jnp.float32)), rows, selected, local_page, local_limit
            )
        got = np.asarray(
            _attend(cache, q, selected, local_page, local_limit, 16).astype(jnp.float32)
        )
        err = np.abs(got - expected).max()
        if arm == "real":
            assert err < 3e-2, err
        else:
            assert err > 3e-2, (mutation, err)


def test_lse_matches_dense_and_plain_kernel():
    """The DCP partial kernel: same output as the plain kernel, and the LSE of
    the dequantized scores."""
    rng = np.random.default_rng(5)
    cache, rows = _fp4_cache(rng)
    num_tokens, num_real = 40, 36
    selected, local_page, local_limit = _batch(rng, num_tokens, num_real, 2)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    table = jnp.arange(PAGES, dtype=jnp.int32)  # keys are physical pages
    common = dict(sm_scale=HEAD_DIM**-0.5, block_q=16, interpret=True)
    out, lse = msa_sparse_attention_lse(
        q,
        cache,
        jnp.asarray(selected),
        jnp.asarray(local_page),
        jnp.asarray(local_limit),
        table,
        **common,
    )
    plain = msa_sparse_attention(
        q,
        cache,
        jnp.asarray(selected),
        jnp.asarray(local_page),
        jnp.asarray(local_limit),
        **common,
    )
    np.testing.assert_allclose(
        np.asarray(out.astype(jnp.float32)), np.asarray(plain.astype(jnp.float32))
    )
    qf = np.asarray(q.astype(jnp.float32))
    for t in range(num_real):
        keys = []
        for p in selected[t][selected[t] >= 0]:
            n = local_limit[t] if p == local_page[t] else PAGE
            keys.append(rows[p, :n, :HEAD_DIM])
        s = qf[t] @ np.concatenate(keys).T * HEAD_DIM**-0.5  # [H, keys]
        m = s.max(-1)
        want = m + np.log(np.exp(s - m[:, None]).sum(-1))
        np.testing.assert_allclose(np.asarray(lse[t]), want, atol=2e-2)
    assert np.all(np.isneginf(np.asarray(lse[num_real:])))


def test_lowers_for_tpu():
    """Pallas -> Mosaic lowering of the 4-bit path runs without a TPU."""
    tokens = 256
    args = (
        jax.ShapeDtypeStruct((tokens, HEADS, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((64, PAGE // 8 + 1, 4, 2 * HEAD_DIM), jnp.uint8),
        jax.ShapeDtypeStruct((tokens, 16), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    )
    attention = functools.partial(msa_sparse_attention, sm_scale=HEAD_DIM**-0.5)
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(attention)
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("kv_heads,mutation", [(2, None), (4, None), (2, "rotate")])
def test_lse_heads_matches_each_head(kv_heads, mutation):
    """G KV heads per rank (DCP): rows [K_0 | V_0 | K_1 | V_1 ...], head h's
    K / V scales in its own column block of the folded row, one kernel for
    all heads. `rotate` (positive control) stores head h + 1's scales in
    head h's block, which the per-head reference must catch."""
    rng = np.random.default_rng(kv_heads)
    q_per_head = 4
    nib, ks, vs, deq = _quantized_heads(rng, kv_heads)
    if mutation == "rotate":
        ks, vs = ks[1:] + ks[:1], vs[1:] + vs[:1]
    cache = msa_fp4.pack_rows(nib, scales=jnp.asarray(_scale_lanes(ks, vs)))
    num_tokens, num_real = 24, 21
    selected = np.full((num_tokens, kv_heads, TOPK), -1, np.int32)
    local_page = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    for t in range(num_real):
        local_page[t] = rng.integers(PAGES)
        local_limit[t] = rng.integers(1, PAGE + 1)
        for h in range(kv_heads):
            others = [p for p in range(PAGES) if p != local_page[t]]
            more = rng.choice(others, rng.integers(0, TOPK), replace=False)
            selected[t, h, : 1 + len(more)] = [local_page[t], *more]
    q = jnp.asarray(
        rng.standard_normal((num_tokens, kv_heads * q_per_head, HEAD_DIM))
    ).astype(jnp.bfloat16)
    out, lse = msa_sparse_attention_lse_heads(
        q,
        cache,
        jnp.asarray(selected),
        jnp.asarray(local_page),
        jnp.asarray(local_limit),
        jnp.arange(PAGES, dtype=jnp.int32),
        sm_scale=HEAD_DIM**-0.5,
        block_q=16,
        interpret=True,
    )
    qf = np.asarray(q.astype(jnp.float32))
    got = np.asarray(out.astype(jnp.float32))
    assert np.isfinite(got).all()
    errs = []
    for h in range(kv_heads):
        heads = slice(h * q_per_head, (h + 1) * q_per_head)
        want = _reference(qf[:, heads], deq[h], selected[:, h], local_page, local_limit)
        errs.append(np.abs(got[:, heads] - want).max())
    if mutation is None:
        assert max(errs) < 3e-2, errs
    else:
        assert min(errs) > 3e-2, errs
    assert np.all(np.isneginf(np.asarray(lse[num_real:])))


def test_msa_forward_writes_folded_pages_and_attends(monkeypatch):
    """`msa.msa_forward` on folded 4-bit caches: the step's K/V nibbles and
    K/V scales land in the KV pages (no scale cache), and the output matches
    dense attention over the dequantized cache. Contexts fit in `topk`
    blocks, so the selection takes every block and the reference needs no
    indexer."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_indexer

    rng = np.random.default_rng(21)
    q_lens, ctx_lens = [1, 5], [200, 130]
    num_seqs, max_num_seqs, pages_per_seq = 2, 3, 3
    num_pages = max_num_seqs * pages_per_seq + 1
    num_real = sum(q_lens)
    num_tokens = num_real + 2
    heads = 4
    monkeypatch.setattr(msa_indexer, "DEFAULT_SMALL_PAGES_PER_STEP", 4)
    perm = rng.permutation(np.arange(1, num_pages))
    tables = np.zeros((max_num_seqs, pages_per_seq), np.int32)
    tables[:num_seqs] = perm[: num_seqs * pages_per_seq].reshape(num_seqs, -1)
    # Context: rotated + quantized K/V and index keys at their slots.
    f32 = np.float32
    ctx = rng.standard_normal((num_seqs, 3, 256, HEAD_DIM)).astype(f32)
    kv_nib = np.zeros((num_pages, PAGE, 2 * HEAD_DIM), np.uint8)
    kv_lanes = np.full((num_pages, 2 * PAGE), np.nan, f32)
    ix_nib = np.zeros((num_pages, PAGE, HEAD_DIM), np.uint8)
    ix_lanes = np.full((num_pages, PAGE), np.nan, f32)
    for r in range(num_seqs):
        n = ctx_lens[r]
        parts = [msa_fp4.quantize(msa_fp4.rotate_keys(jnp.asarray(ctx[r, i, :n]))) for i in range(3)]
        for j in range(n):
            page, slot = tables[r, j // PAGE], j % PAGE
            kv_nib[page, slot] = np.concatenate([np.asarray(parts[0][0][j]), np.asarray(parts[1][0][j])])
            kv_lanes[page, slot] = parts[0][1][j]
            kv_lanes[page, PAGE + slot] = parts[1][1][j]
            ix_nib[page, slot] = np.asarray(parts[2][0][j])
            ix_lanes[page, slot] = parts[2][1][j]
    kv_cache = msa_fp4.pack_rows(jnp.asarray(kv_nib), scales=jnp.asarray(kv_lanes))
    index_cache = msa_fp4.pack_rows(jnp.asarray(ix_nib), scales=jnp.asarray(ix_lanes))
    q = jnp.asarray(rng.standard_normal((num_tokens, heads, HEAD_DIM))).astype(jnp.bfloat16)
    k, v, iq = (
        jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(jnp.bfloat16)
        for _ in range(3)
    )
    ik = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM))).astype(jnp.bfloat16)
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_num_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    new_kv, new_idx, out = msa.msa_forward(
        q, k, v, iq, ik, kv_cache, index_cache,
        jnp.asarray(seq_lens), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu_q),
        jnp.asarray([num_seqs] * 3, jnp.int32),
        sm_scale=HEAD_DIM**-0.5, topk=TOPK, init_blocks=0, local_blocks=1,
        interpret=True,
    )
    assert new_kv.shape == kv_cache.shape
    nib = np.asarray(msa_fp4.unpack_rows(new_kv))
    ks, vs = (np.asarray(x[:, 0]) for x in msa_fp4.kv_scales(new_kv))
    k_nib, k_s = msa_fp4.quantize(msa_fp4.rotate_keys(k[:, 0]))
    v_nib, v_s = msa_fp4.quantize(msa_fp4.rotate_keys(v[:, 0]))
    deq = np.concatenate(
        [np.asarray(msa_fp4.dequantize(nib[..., :HEAD_DIM], ks)),
         np.asarray(msa_fp4.dequantize(nib[..., HEAD_DIM:], vs))], axis=-1,
    )  # rotated basis; NaN only at unwritten slots
    q_rot = np.asarray(msa_fp4.rotate_queries(q.astype(jnp.float32)))
    got = np.asarray(out.astype(jnp.float32))
    start = 0
    for r in range(num_seqs):
        for i in range(q_lens[r]):
            t, p = start + i, ctx_lens[r] + i
            page, slot = tables[r, p // PAGE], p % PAGE
            np.testing.assert_array_equal(nib[page, slot, :HEAD_DIM], k_nib[t])
            np.testing.assert_array_equal(nib[page, slot, HEAD_DIM:], v_nib[t])
            np.testing.assert_allclose([ks[page, slot], vs[page, slot]], [k_s[t], v_s[t]], rtol=1e-6)
            rows = np.concatenate([deq[tables[r, j // PAGE], j % PAGE][None] for j in range(p + 1)])
            s = q_rot[t] @ rows[:, :HEAD_DIM].T * HEAD_DIM**-0.5
            w = np.exp(s - s.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            want = np.asarray(msa_fp4.unrotate_output(jnp.asarray(w @ rows[:, HEAD_DIM:], jnp.float32)))
            np.testing.assert_allclose(got[t], want, atol=3e-2)
        start += q_lens[r]
    np.testing.assert_array_equal(got[num_real:], 0)
