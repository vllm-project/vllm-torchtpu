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
"""4-bit MSA caches, interpret mode: the nibble pack / unpack and int4
quantizer, and shared scale pages (TPU_MINIMAX_M3_KV4_SHARED_SCALES): a
layer that writes and reads its K/V scales at row ``row0`` of a page shared
with other layers gets the caches, candidates (indexer) and attention
outputs of the same layer over its own page, bit for bit, and leaves the
other layers' rows alone."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3 import test_msa_index_reuse as t
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention_lse,
    msa_dcp,
    msa_fp4,
    msa_write,
)


# ------------------------------------------------------- pack / quantize
def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    for width in (128, 256, 512):
        nib = jnp.asarray(rng.integers(0, 16, (5, 128, width)), jnp.uint8)
        packed = msa_fp4.pack_rows(nib)
        assert packed.shape == (5, 16, 4, width) and packed.dtype == jnp.uint8
        np.testing.assert_array_equal(msa_fp4.unpack_rows(packed), nib)
    # Folded scales (index pages): one more word row, lane = slot.
    nib = jnp.asarray(rng.integers(0, 16, (3, 128, 128)), jnp.uint8)
    scales = jnp.asarray(rng.uniform(0.01, 2.0, (3, 128)), jnp.float32)
    packed = msa_fp4.pack_rows(nib, scales)
    assert packed.shape == (3, 17, 4, 128)
    assert msa_fp4.scales_in_page(packed.shape[1], packed.dtype)
    np.testing.assert_array_equal(msa_fp4.unpack_rows(packed), nib)
    np.testing.assert_array_equal(msa_fp4.folded_scales(packed), scales)


def test_quantize_is_exact_on_grid_and_bounded():
    rng = np.random.default_rng(1)
    # Values on an int4 grid come back exactly.
    q = rng.integers(-7, 8, (64, 128)).astype(np.float32)
    q[:, 0] = 7  # amax = 7 * scale
    scale = rng.uniform(0.01, 1.0, (64, 1)).astype(np.float32)
    nib, s = msa_fp4.quantize(jnp.asarray(q * scale))
    np.testing.assert_allclose(np.asarray(s), scale[:, 0], rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(msa_fp4.dequantize(nib, s)), q * scale, rtol=1e-6, atol=0
    )
    # Any vector: error at most half a step; zero vectors stay zero.
    x = rng.standard_normal((64, 128)).astype(np.float32)
    x[3] = 0
    nib, s = msa_fp4.quantize(jnp.asarray(x))
    err = np.abs(np.asarray(msa_fp4.dequantize(nib, s)) - x)
    assert np.all(err <= np.asarray(s)[:, None] * 0.5 + 1e-6)
    assert float(s[3]) == 0 and not np.asarray(msa_fp4.dequantize(nib, s))[3].any()


def test_layers_per_scale_page():
    assert [msa_fp4.layers_per_scale_page(g) for g in (1, 2, 3, 4)] == [4, 2, 1, 1]
    view = msa_fp4.with_row0(jnp.zeros((2, 8, 128)), 4)
    leaves, tree = jax.tree_util.tree_flatten(view)
    assert len(leaves) == 1 and msa_fp4.split_scales(
        jax.tree_util.tree_unflatten(tree, leaves)
    )[1] == 4
    raw = jnp.zeros((2, 8, 128))
    assert msa_fp4.with_row0(raw, 0) is raw and msa_fp4.split_scales(raw)[1] == 0


# ------------------------------------------------------- slot-value writes
def test_slot_values_row_offset():
    """`SlotValues` at `row0` writes rows [row0, row0 + n) of the written
    slots and keeps every other row and slot."""
    rng = np.random.default_rng(2)
    pages, n, row0 = 6, 2, 4
    cache = jnp.asarray(rng.uniform(1, 2, (pages, 8, 128)), jnp.float32)
    tok_pages = np.array([3, 3, 3, 1, 5, 5], np.int32)
    slots = np.array([10, 11, 12, 127, 0, 1], np.int32)
    values = jnp.asarray(rng.uniform(-1, 0, (6, n)), jnp.float32)
    got = msa_write.write_runs(
        [msa_write.SlotValues(cache, values, row0)],
        jnp.asarray(tok_pages),
        jnp.asarray(slots),
        max_runs=8,
        interpret=True,
    )[0]
    want = np.asarray(cache).copy()
    for i, (p, s) in enumerate(zip(tok_pages, slots)):
        want[p, row0 : row0 + n, s] = np.asarray(values[i])
    np.testing.assert_array_equal(np.asarray(got), want)
    with pytest.raises(ValueError, match="past the"):
        msa_write.write_runs(
            [msa_write.SlotValues(cache, values, 7)],
            jnp.asarray(tok_pages),
            jnp.asarray(slots),
            max_runs=8,
            interpret=True,
        )


# ------------------------------------------------------- DCP layers
def _shared_page(layers, g):
    """One scale page per rank holding layer i's 2g rows at rows [2gi, 2g(i+1)),
    the remaining rows random."""
    dcp = len(layers[0]["scale"])
    rng = np.random.default_rng(9)
    out = []
    for r in range(dcp):
        page = jnp.asarray(rng.uniform(5, 6, layers[0]["scale"][r].shape), jnp.float32)
        for i, layer in enumerate(layers):
            page = page.at[:, 2 * g * i : 2 * g * (i + 1)].set(
                layer["scale"][r][:, : 2 * g]
            )
        out.append(page)
    return out


def _layers(seed, dcp, g):
    share = msa_fp4.layers_per_scale_page(g)
    rng, num_tokens, meta = t._setup(seed, dcp)
    return [t._layer(rng, num_tokens, dcp, g, "fp4") for _ in range(share)], meta


@pytest.mark.parametrize("dcp,g", [(2, 1), (4, 2)], ids=["dcp2", "dcp4"])
def test_shared_page_write_and_candidates(dcp, g):
    """Every layer of a page, in order, writes and scores with its rows of the
    shared page: caches and candidates equal its own-page run, its rows equal
    its own page's rows, and the page ends holding every layer's rows."""
    layers, meta = _layers(30 + g, dcp, g)
    shared = _shared_page(layers, g)
    for i, layer in enumerate(layers):
        own_caches, own_cands = t._full(layer, meta, dcp)
        view = dict(layer, scale=[msa_fp4.with_row0(s, 2 * g * i) for s in shared])
        caches, cands = t._full(view, meta, dcp)
        np.testing.assert_array_equal(np.asarray(cands), np.asarray(own_cands))
        for r in range(dcp):
            (kv, index, sc), (okv, oindex, osc) = caches[r], own_caches[r]
            np.testing.assert_array_equal(np.asarray(kv), np.asarray(okv))
            np.testing.assert_array_equal(np.asarray(index), np.asarray(oindex))
            page, row0 = msa_fp4.split_scales(sc)
            assert row0 == 2 * g * i
            page, before = np.asarray(page), np.asarray(shared[r])
            rows = slice(row0, row0 + 2 * g)
            np.testing.assert_array_equal(page[:, rows], np.asarray(osc)[:, : 2 * g])
            keep = np.ones(page.shape[1], bool)
            keep[rows] = False
            np.testing.assert_array_equal(page[:, keep], before[:, keep])
            shared[r] = jnp.asarray(page)


@pytest.mark.parametrize("dcp,g", [(2, 1), (4, 2)], ids=["dcp2", "dcp4"])
@pytest.mark.parametrize("path", ["small", "large", "prefill"])
def test_shared_page_attention(dcp, g, path, monkeypatch):
    """The last layer of a page attends over its rows of the shared page as
    over its own page, bit for bit: one KV head (DCP 2, the scale rows are
    static) and two (DCP 4, ``("head", row0)`` rows), decode-sized (small /
    large) and prefill-sized (page cache) programs."""
    if path == "large":
        monkeypatch.setattr(msa_dcp, "DCP_SMALL_BATCH_TOKENS", 0)
    if path == "prefill":
        monkeypatch.setattr(msa_dcp, "DCP_SMALL_BATCH_TOKENS", 0)
        monkeypatch.setattr(msa_dcp, "_MIN_CANDIDATE_TOKENS", 1)
    jax.clear_caches()
    layers, meta = _layers(40 + g, dcp, g)
    shared = _shared_page(layers, g)
    i = len(layers) - 1
    layer = layers[i]
    caches, cands = t._full(layer, meta, dcp)
    q = t._q(layer, True)
    want = t._attend(q, caches, meta, dcp, candidates=cands)
    view = [
        (kv, index, msa_fp4.with_row0(shared[r], 2 * g * i))
        for r, (kv, index, _) in enumerate(caches)
    ]
    # The written rows of this layer into the shared page (as its own write).
    view = [
        (kv, index, msa_fp4.with_row0(
            msa_fp4.pages_of(v).at[:, 2 * g * i : 2 * g * (i + 1)].set(sc[:, : 2 * g]),
            2 * g * i,
        ))
        for (kv, index, v), (_, _, sc) in zip(view, caches)
    ]  # fmt: skip
    got = t._attend(q, view, meta, dcp, candidates=cands)
    num_real = sum(t.Q_LENS)
    for (wo, wl), (go, gl) in zip(want, got):
        assert np.abs(wo[:, :num_real]).max() > 0
        np.testing.assert_array_equal(go, wo)
        np.testing.assert_array_equal(gl, wl)
    # Control: the same rows read at row 0 (another layer's) differ.
    wrong = [(kv, index, msa_fp4.pages_of(v)) for kv, index, v in view]
    alt = t._attend(q, wrong, meta, dcp, candidates=cands)
    assert any(not np.array_equal(a[0], w[0]) for a, w in zip(alt, want))


def test_heads_kernel_rows_from_offset():
    """`msa_sparse_attention_lse_heads` over a page whose rows [4, 8) hold
    the two heads' scales equals the call over a page with them at [0, 4)."""
    from tests.kernels.test_minimax_m3_msa_decode_paths import _attention_inputs

    q, kv, sel, local, limit, table, sc = _attention_inputs(
        np.random.default_rng(5), 2, 3, True
    )
    other = jnp.asarray(np.random.default_rng(6).uniform(1, 2, sc.shape), jnp.float32)
    shared = other.at[:, 4:8].set(sc[:, :4])
    kw = dict(sm_scale=t.HEAD_DIM**-0.5, block_q=8, interpret=True, head_major=True)
    want = msa_attention_lse.msa_sparse_attention_lse_heads(
        q, kv, sel, local, limit, table, scale_cache=sc, **kw
    )
    got = msa_attention_lse.msa_sparse_attention_lse_heads(
        q, kv, sel, local, limit, table, scale_cache=msa_fp4.with_row0(shared, 4), **kw
    )
    for a, b in zip(want, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="miss K/V rows"):
        msa_attention_lse.msa_sparse_attention_lse_heads(
            q, kv, sel, local, limit, table,
            scale_cache=msa_fp4.with_row0(shared, 6), **kw,
        )  # fmt: skip
