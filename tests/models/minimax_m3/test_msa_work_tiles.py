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
"""The attention work lists' per-tile page unions (`msa_attention.tile_unions`)
are exactly `jnp.unique`'s: same pages, same order, same padding, so the
attention kernel visits the same pages in the same groups and its outputs
are bit-identical."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import PACKING, pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_attention_lse,
    msa_dcp,
)

NO_PAGE = msa_attention._NO_PAGE
PAGE = 128
HEAD_DIM = 128
TOPK = 16


def _unique_unions(per_tile):
    """The previous implementation: one `jnp.unique` per row."""
    rows = per_tile.reshape(-1, per_tile.shape[-1])
    size = per_tile.shape[-1]
    out = jax.vmap(lambda x: jnp.unique(x, size=size, fill_value=NO_PAGE))(rows)
    return out.reshape(per_tile.shape)


def _sorted_only(per_tile):  # fault: no de-duplication
    return jnp.sort(per_tile, axis=-1)


def _single_sort(per_tile):  # fault: duplicates removed in place, not moved back
    ordered = jnp.sort(per_tile, axis=-1)
    repeat = jnp.concatenate(
        [jnp.zeros_like(ordered[..., :1], bool), ordered[..., 1:] == ordered[..., :-1]],
        axis=-1,
    )
    return jnp.where(repeat, NO_PAGE, ordered)


def _rows(rng):
    """Rows of 256 keys with few, many, all and no repeats, and pad rows."""
    rows = [
        rng.integers(0, 3, 256),
        rng.integers(0, 64, 256),
        rng.integers(0, 2**31 - 1, 256),
        np.full(256, NO_PAGE),
        np.full(256, 7),
        rng.permutation(256),
        np.where(rng.random(256) < 0.5, NO_PAGE, rng.integers(0, 40, 256)),
    ]
    rows += [np.where(rng.random(256) < 0.3, NO_PAGE, rng.integers(0, 600, 256)) for _ in range(25)]
    return jnp.asarray(np.stack(rows).astype(np.int32)).reshape(2, 16, 256)


def test_tile_unions_equal_unique():
    got = jax.jit(msa_attention.tile_unions)(_rows(np.random.default_rng(0)))
    want = _unique_unions(_rows(np.random.default_rng(0)))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("fault", [_sorted_only, _single_sort])
def test_tile_unions_check_catches_faults(fault):
    """Positive controls: the comparison above fails for a union that keeps
    the repeats or leaves holes between the distinct pages."""
    rows = _rows(np.random.default_rng(0))
    assert not np.array_equal(np.asarray(fault(rows)), np.asarray(_unique_unions(rows)))


def _merged_selection(rng, num_tokens, num_heads, num_seqs, dcp):
    """A merged DCP selection [T, G, topk] shaped like a served one: a few
    sequences, every token picks the first block, its own block and earlier
    blocks by score, -1 past its visible blocks; plus padding tokens."""
    q_lens = np.full(num_seqs, num_tokens // num_seqs - 8)
    seq_lens = q_lens + rng.integers(4 * num_tokens, 40 * num_tokens, num_seqs)
    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    seq_id = np.zeros(num_tokens, np.int32)
    pos = np.zeros(num_tokens, np.int32)
    valid = np.zeros(num_tokens, bool)
    selected = np.full((num_tokens, num_heads, TOPK), -1, np.int32)
    for s in range(num_seqs):
        for i, t in enumerate(range(cu_q[s], cu_q[s + 1])):
            seq_id[t], valid[t] = s, True
            pos[t] = seq_lens[s] - q_lens[s] + i
            own = pos[t] // PAGE
            for h in range(num_heads):
                pool = np.arange(own + 1)
                n = min(TOPK, own + 1)
                picks = np.concatenate([[0, own], rng.choice(pool, n, replace=False)])
                picks = list(dict.fromkeys(picks.tolist()))[:n]
                selected[t, h, : len(picks)] = picks
    return selected, seq_id, pos, valid


# (dcp, G, tokens): the served prefill buckets, a flat-work-list step
# (<= SMALL_BATCH_TOKENS) and the tiled-grid steps.
_SHAPES = [
    pytest.param(2, 1, 1024, id="dcp2-1024"),
    pytest.param(4, 2, 1024, id="dcp4-1024"),
    pytest.param(2, 1, 2048, id="dcp2-2048"),
    pytest.param(4, 2, 2048, id="dcp4-2048"),
    pytest.param(4, 2, 8192, id="dcp4-8192"),
    pytest.param(8, 4, 2048, id="dcp8-2048"),
    pytest.param(8, 4, 8192, id="dcp8-8192"),
]


@pytest.mark.parametrize("dcp,num_heads,num_tokens", _SHAPES)
def test_work_lists_equal_unique_at_served_shapes(monkeypatch, dcp, num_heads, num_tokens):
    """Every rank's work lists (`tiles` grid and flat list) of a served-like
    merged selection are the arrays the `jnp.unique` builder made."""
    rng = np.random.default_rng(dcp * 10_000 + num_tokens)
    selected, seq_id, pos, valid = _merged_selection(rng, num_tokens, num_heads, 3, dcp)
    pages_per_seq = 48 * num_tokens // PAGE
    for rank in range(dcp):
        _, keys = msa_dcp.selected_page_keys(
            jnp.asarray(selected), jnp.asarray(seq_id), jnp.asarray(pos), jnp.asarray(valid),
            topk=TOPK, block_size=PAGE, pages_per_seq=pages_per_seq,
            rank=jnp.int32(rank), dcp=dcp, merged=True,
        )
        heads = keys.transpose(1, 0, 2)
        for tiles in (True, False):
            build = functools.partial(
                msa_attention_lse._work_lists, block_q=16, pages_per_step=16, tiles=tiles
            )
            jax.clear_caches()  # a cached trace would hide the patch below
            got = jax.jit(build)(heads)
            with monkeypatch.context() as m:
                m.setattr(msa_attention, "tile_unions", _unique_unions)
                jax.clear_caches()
                want = jax.jit(build)(heads)
            jax.clear_caches()
            for g, w in zip(got, want):
                np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
            assert int(np.asarray(got[2]).max()) > 1  # tiles with several groups


@pytest.mark.parametrize(
    "dcp,num_heads,num_tokens",
    [
        pytest.param(2, 1, 1024, id="dcp2-1024"),
        pytest.param(4, 2, 2048, id="dcp4-2048"),
        pytest.param(8, 4, 2048, id="dcp8-2048"),
        pytest.param(4, 2, 64, id="dcp4-decode64"),
    ],
)
def test_attend_bit_identical_to_unique_work_lists(monkeypatch, dcp, num_heads, num_tokens):
    """`msa_attend_dcp` (interpret mode) gives the same bits with the new
    unions as with `jnp.unique`'s, prefill (flat list and tile grid) and
    decode (no work list)."""
    dtype = jnp.float8_e4m3fn
    rng = np.random.default_rng(dcp + num_tokens)
    num_seqs = 1 if num_tokens <= 64 else 3
    if num_tokens <= 64:
        selected, seq_id, pos, valid = _merged_selection(rng, 72, num_heads, 1, dcp)
        selected, seq_id, pos, valid = (x[:num_tokens] for x in (selected, seq_id, pos, valid))
    else:
        selected, seq_id, pos, valid = _merged_selection(rng, num_tokens, num_heads, num_seqs, dcp)
    pages_per_seq = 48 * max(num_tokens, 1024) // PAGE
    # The step metadata of these positions; keys map to random physical pages.
    seq_lens = np.zeros(4, np.int32)
    cu_q = np.zeros(5, np.int32)
    for s in range(num_seqs):
        mask = valid & (seq_id == s)
        seq_lens[s] = pos[mask].max() + 1
        cu_q[s + 1] = cu_q[s] + mask.sum()
    cu_q[num_seqs + 1 :] = cu_q[num_seqs]
    pages = 96
    block_tables = rng.integers(0, pages, 4 * pages_per_seq).astype(np.int32)
    kv = pack(
        rounded(rng.standard_normal((pages, PAGE, num_heads * 2 * HEAD_DIM)), dtype), dtype
    )
    q = jnp.asarray(rng.standard_normal((num_tokens, 8 * dcp, HEAD_DIM))).astype(jnp.bfloat16)
    attend = functools.partial(
        msa_dcp.msa_attend_dcp, sm_scale=HEAD_DIM**-0.5, topk=TOPK, dcp=dcp,
        merged=True, interpret=True,
    )
    for rank in range(min(dcp, 2)):
        args = (
            q, kv, jnp.asarray(selected), jnp.asarray(seq_lens),
            jnp.asarray(block_tables), jnp.asarray(cu_q),
        )
        jax.clear_caches()
        got = jax.jit(functools.partial(attend, rank=rank))(*args)
        with monkeypatch.context() as m:
            m.setattr(msa_attention, "tile_unions", _unique_unions)
            jax.clear_caches()
            want = jax.jit(functools.partial(attend, rank=rank))(*args)
        jax.clear_caches()
        for g, w in zip(got, want):
            np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
        assert np.isfinite(np.asarray(got[1])).any()


@pytest.mark.parametrize(
    "dcp,num_heads,num_tokens",
    [
        pytest.param(2, 1, 1024, id="dcp2-1024"),
        pytest.param(4, 2, 2048, id="dcp4-2048"),
        pytest.param(4, 2, 8192, id="dcp4-8192"),
        pytest.param(8, 4, 4096, id="dcp8-4096"),
        pytest.param(8, 4, 8192, id="dcp8-8192"),
    ],
)
def test_work_lists_compile_for_tpu7x_without_compaction(dcp, num_heads, num_tokens):
    """The merged attend compiles for tpu7x (deviceless AOT) at every served
    prefill shape class, and its work lists are row sorts: no scatter
    (`jnp.unique`'s compaction ran as SparseCore scatter and gathers)."""
    from jax.experimental import topologies
    from jax.sharding import AbstractDevice, AbstractMesh, SingleDeviceSharding
    from jax.sharding import use_abstract_mesh

    try:
        topology = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    sharding = SingleDeviceSharding(topology.devices[0])

    def spec(shape, dt):
        return jax.ShapeDtypeStruct(shape, dt, sharding=sharding)

    dtype = jnp.float8_e4m3fn
    packing = PACKING[dtype]
    max_seqs, pages_per_seq = 64, 1024
    i32 = jnp.int32
    args = (
        spec((num_tokens, 8 * dcp, HEAD_DIM), jnp.bfloat16),
        spec((4096, PAGE // packing, packing, 2 * HEAD_DIM * num_heads), dtype),
        spec((num_tokens, num_heads, TOPK), i32),
        spec((max_seqs,), i32),
        spec((max_seqs * pages_per_seq,), i32),
        spec((max_seqs + 1,), i32),
        spec((1,), i32),
    )

    def attend(*a):
        *rest, r = a
        return msa_dcp.msa_attend_dcp(
            *rest, sm_scale=HEAD_DIM**-0.5, topk=TOPK, rank=r, dcp=dcp, merged=True
        )

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        compiled = jax.jit(attend).lower(*args).compile()
    text = compiled.as_text()
    assert "minimax_m3_msa_sparse_attention_lse" in text
    assert " sort(" in text
    assert " scatter(" not in text
