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
"""Keys minimax_indexer_hoist_metadata and minimax_indexer_row_gather
(`msa_indexer.HOIST_METADATA` / `ROW_GATHER`) leave every output bit as it
is (interpret mode): `block_scores`, the DCP candidates of
`msa_dcp.msa_write_and_candidates_dcp` (DCP 1 and 4, one and two KV heads
per rank), and a chain of MSA layers; and the hoisted metadata is computed
once per step, not once per layer."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_dcp,
    msa_fp4,
    msa_indexer,
    msa_write,
)

PAGE = 128
HEAD_DIM = 128
# Eight requests: the cond path (`is_prefill_step` false) holds up to 512 rows.
MAX_SEQS = 8
PAGES = 72
PAGES_PER_SEQ = 8
VARIANTS = {
    "hoist": (True, False),
    "row_gather": (False, True),
    "both": (True, True),
}


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    """Start every test with both keys off and fresh jit caches: the flags
    are read at trace time, and `block_scores` is jitted."""
    monkeypatch.setattr(msa_indexer, "HOIST_METADATA", False)
    monkeypatch.setattr(msa_indexer, "ROW_GATHER", False)
    jax.clear_caches()
    yield
    jax.clear_caches()


def _set(monkeypatch, hoist, row_gather):
    monkeypatch.setattr(msa_indexer, "HOIST_METADATA", hoist)
    monkeypatch.setattr(msa_indexer, "ROW_GATHER", row_gather)
    jax.clear_caches()


def _layout(
    q_lens, ctx_lens, num_tokens, rng, pages_per_seq=PAGES_PER_SEQ, num_pages=PAGES
):
    num_seqs, num_real = len(q_lens), sum(q_lens)
    assert num_real <= num_tokens and num_seqs <= MAX_SEQS
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.add(ctx_lens, q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, pages_per_seq), np.int32)
    assert num_seqs * pages_per_seq < num_pages
    tables[:num_seqs] = rng.permutation(np.arange(1, num_pages))[
        : num_seqs * pages_per_seq
    ].reshape(num_seqs, pages_per_seq)
    return (
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
    )


def _bits(x):
    x = np.asarray(x)
    return x.view(np.int32) if x.dtype == np.float32 else x


# (q_lens, ctx_lens, num_tokens): a prefill chunk next to decode / MTP rows
# (the cond's large branch); decode rows only (its small branch); one large
# request only (no small row); eight requests, the long one 1-7 rows off the
# alignment; a prefill-sized step (no cond: `is_prefill_step`).
BATCHES = {
    "mixed": ([1, 8, 150, 4], [900, 128, 300, 700], 200),
    "decode": ([1, 8, 4, 1], [900, 128, 127, 700], 200),
    "large_only": ([200], [500], 208),
    "eight": ([8, 3, 8, 8, 133, 8, 1, 8], [100, 0, 200, 300, 40, 600, 900, 5], 192),
    "prefill_step": ([300, 230, 8], [600, 100, 50], 544),
}


def _block_scores(index_q, cache, meta, **kw):
    return np.asarray(msa_indexer.block_scores.__wrapped__(index_q, cache, *meta, **kw))


@pytest.mark.parametrize("batch", sorted(BATCHES))
@pytest.mark.parametrize(
    "dtype,small_pages_per_step",
    [
        (jnp.bfloat16, 4),  # small class transposed ([P, columns])
        (jnp.bfloat16, 128),  # token-major small class (kernel v3)
        ("fp4", 128),  # folded 4-bit pages, as served at c48
    ],
    ids=["bf16-transposed", "bf16-token-major", "fp4-token-major"],
)
def test_block_scores_bit_identical(monkeypatch, batch, dtype, small_pages_per_step):
    q_lens, ctx_lens, num_tokens = BATCHES[batch]
    rng = np.random.default_rng(len(q_lens) + num_tokens)
    meta = _layout(q_lens, ctx_lens, num_tokens, rng)
    raw = rng.standard_normal((PAGES, PAGE, HEAD_DIM))
    index_q = jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    kw = dict(
        block_q=128,
        large_block_q=64,
        pages_per_step=4,
        small_pages_per_step=small_pages_per_step,
        interpret=True,
    )
    if dtype == "fp4":
        nib, scale = msa_fp4.quantize(
            msa_fp4.rotate_keys(jnp.asarray(raw, jnp.float32))
        )
        cache = msa_fp4.pack_rows(nib, scales=scale)
        index_q = msa_fp4.rotate_queries(index_q)
        kw["folded_tile_q"] = 16
    else:
        cache = pack(rounded(raw, dtype), dtype)

    want = _bits(_block_scores(index_q, cache, meta, **kw))
    jaxpr_off = str(
        jax.make_jaxpr(
            lambda q: msa_indexer.block_scores.__wrapped__(q, cache, *meta, **kw)
        )(index_q)
    )
    for name, flags in VARIANTS.items():
        _set(monkeypatch, *flags)
        got = _bits(_block_scores(index_q, cache, meta, **kw))
        np.testing.assert_array_equal(got, want, err_msg=name)
        if name == "hoist" and batch != "prefill_step":
            # The flag reached the trace (the jit cache was not reused).
            jaxpr_on = str(
                jax.make_jaxpr(
                    lambda q: msa_indexer.block_scores.__wrapped__(
                        q, cache, *meta, **kw
                    )
                )(index_q)
            )
            assert jaxpr_on != jaxpr_off


def test_class_meta_onehot_matches_gather():
    """`_class_meta` gives the same metadata by one-hot lookups as by gathers
    (the hoisted copy uses one-hot lookups), padding requests included."""
    rng = np.random.default_rng(3)
    for q_lens, ctx_lens, num_tokens in BATCHES.values():
        seq_lens, tables, cu_q, num_seqs = _layout(q_lens, ctx_lens, num_tokens, rng)
        q_len = cu_q[1:] - cu_q[:-1]
        for in_class in (q_len > 64, (q_len <= 64) & (jnp.arange(MAX_SEQS) < num_seqs)):
            for block_q, max_tiles in ((64, 16), (8, 64)):
                kw = dict(block_q=block_q, max_tiles=max_tiles, pages_per_step=4)
                args = (num_tokens, PAGE, seq_lens, tables, cu_q, in_class)
                a = msa_indexer._class_meta(*args, onehot=False, **kw)
                b = msa_indexer._class_meta(*args, onehot=True, **kw)
                for name, x, y in zip(a._fields, a, b, strict=True):
                    np.testing.assert_array_equal(x, y, err_msg=name)


def _xla_write_rows(cache, rows, pages, slots, *, max_runs, bits=None, interpret=False):
    """`msa_write.write_rows` as an XLA scatter (covered by test_msa_write)."""
    del max_runs, bits, interpret
    packing = cache.shape[2]
    return cache.at[pages, slots // packing, slots % packing].set(rows, mode="drop")


def _dcp_inputs(rng, q_lens, ctx_lens, num_tokens, dcp, heads, dtype):
    logical_pages = -(-max(np.add(q_lens, ctx_lens)) // (PAGE * dcp))
    meta = _layout(q_lens, ctx_lens, num_tokens, rng, logical_pages, DCP_PAGES)
    meta = (*meta[:3], jnp.asarray([0, 0, len(q_lens)], jnp.int32))
    kv_cache = pack(
        np.zeros((DCP_PAGES, PAGE, 2 * HEAD_DIM * heads), np.float32), dtype
    )
    index = pack(
        rounded(rng.integers(-2, 3, (DCP_PAGES, PAGE, HEAD_DIM)), dtype), dtype
    )

    def bf16(*shape):
        # Integer-valued queries: exact dots and frequent equal scores.
        return jnp.asarray(rng.integers(-2, 3, shape)).astype(jnp.bfloat16)

    tensors = (
        bf16(num_tokens, heads, HEAD_DIM),
        bf16(num_tokens, heads, HEAD_DIM),
        bf16(num_tokens, heads, HEAD_DIM),
        bf16(num_tokens, HEAD_DIM),
    )
    return tensors, (kv_cache, index), meta


# (q_lens, ctx_lens, num_tokens) at G = 1 and 2 KV heads per rank; folded
# rows stay at most 64 * MAX_SEQS, the cond path.
DCP_PAGES = 100
DCP_BATCHES = {
    "mixed": ([1, 8, 150, 4], [1400, 128, 1000, 1200], 240),
    "decode": ([1, 8, 4, 1, 8], [1400, 128, 127, 1200, 900], 32),
    "eight": ([8, 3, 8, 8, 70, 8, 1, 8], [100, 0, 1100, 300, 40, 600, 900, 5], 128),
}


@pytest.mark.parametrize("batch", sorted(DCP_BATCHES))
@pytest.mark.parametrize("dcp,heads", [(1, 1), (4, 2)])
def test_dcp_candidates_bit_identical(monkeypatch, batch, dcp, heads):
    monkeypatch.setattr(msa_write, "write_rows", _xla_write_rows)
    q_lens, ctx_lens, num_tokens = DCP_BATCHES[batch]
    rng = np.random.default_rng(dcp * 100 + num_tokens)
    tensors, caches, meta = _dcp_inputs(
        rng, q_lens, ctx_lens, num_tokens, dcp, heads, jnp.bfloat16
    )
    assert not msa_indexer.is_prefill_step(heads * num_tokens, MAX_SEQS)
    config = dict(topk=8, init_blocks=1, local_blocks=1)

    def run():
        outs = []
        for rank in range(dcp):
            outs.append(
                msa_dcp.msa_write_and_candidates_dcp(
                    *tensors,
                    *caches,
                    *meta,
                    rank=rank,
                    dcp=dcp,
                    interpret=True,
                    **config,
                )
            )
        return [np.asarray(x) for out in outs for x in out]

    want = run()
    _set(monkeypatch, True, True)
    got = run()
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(_bits(a), _bits(b))


def _layers(index_q0, k, v, index_k, caches, meta, num_layers):
    """A chain of `msa.msa_forward` layers whose queries depend on the
    previous layer's output, each with its own caches."""
    q = k_in = jnp.tile(k, (1, 4, 1))
    index_q = index_q0
    outs = []
    for layer in range(num_layers):
        kv_cache, index_cache = caches[layer]
        kv_cache, index_cache, out = msa.msa_forward(
            q,
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            *meta,
            sm_scale=HEAD_DIM**-0.5,
            topk=4,
            init_blocks=1,
            local_blocks=1,
            interpret=True,
        )
        outs.append(out)
        # Next layer: queries from this layer's output (round to bf16).
        q = (out.astype(jnp.float32) + k_in.astype(jnp.float32)).astype(jnp.bfloat16)
        index_q = out[:, :1].astype(jnp.bfloat16)
    return outs


def _layer_inputs(num_layers, q_lens, ctx_lens, num_tokens, seed):
    rng = np.random.default_rng(seed)
    meta = _layout(q_lens, ctx_lens, num_tokens, rng)
    meta = (*meta[:3], jnp.asarray([0, 0, len(q_lens)], jnp.int32))

    def bf16(*shape):
        return jnp.asarray(rng.standard_normal(shape)).astype(jnp.bfloat16)

    def cache(width):
        rows = rng.standard_normal((PAGES, PAGE, width))
        return pack(rounded(rows, jnp.bfloat16), jnp.bfloat16)

    caches = [(cache(2 * HEAD_DIM), cache(HEAD_DIM)) for _ in range(num_layers)]
    tensors = (
        bf16(num_tokens, 1, HEAD_DIM),
        bf16(num_tokens, 1, HEAD_DIM),
        bf16(num_tokens, 1, HEAD_DIM),
        bf16(num_tokens, HEAD_DIM),
    )
    return tensors, caches, meta


def test_layer_chain_bit_identical(monkeypatch):
    """Three MSA layers (DCP 1), each layer's queries from the last one's
    output: every layer's output is bit-identical with both keys on."""
    monkeypatch.setattr(msa_write, "write_rows", _xla_write_rows)
    num_layers = 3
    tensors, caches, meta = _layer_inputs(
        num_layers, [1, 8, 150, 4], [900, 128, 300, 700], 200, seed=5
    )
    want = _layers(*tensors, caches, meta, num_layers)
    _set(monkeypatch, True, True)
    got = _layers(*tensors, caches, meta, num_layers)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(
            np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16)
        )


def _branch_metadata_ops(num_layers, monkeypatch, hoist):
    """Integer metadata ops (`searchsorted` / `_tiles` lookups) inside the
    optimized HLO's conditional branches of a jitted `num_layers` chain."""
    monkeypatch.setattr(msa_write, "write_rows", _xla_write_rows)
    _set(monkeypatch, hoist, False)
    tensors, caches, meta = _layer_inputs(
        num_layers, [1, 8, 150, 4], [900, 128, 300, 700], 200, seed=6
    )
    fn = jax.jit(lambda t, c, m: _layers(*t, c, m, num_layers))
    hlo = fn.lower(tensors, caches, meta).compile().as_text()
    # Computations called as conditional branches.
    branches = set()
    for names in re.findall(r"branch_computations=\{([^}]*)\}", hlo):
        branches.update(n.strip().lstrip("%") for n in names.split(","))
    # Count s32 ops with source lines in msa_indexer's metadata, per branch.
    count = 0
    for block in re.split(r"\n(?=\S)", hlo):
        header = block.split("\n", 1)[0]
        name = header.split(" ", 1)[0].lstrip("%")
        if name in branches:
            count += len(re.findall(r"= s32\[\d+\]", block))
    return count


def test_hoisted_metadata_leaves_the_branches(monkeypatch):
    """With the hoist on, the cond's large branch holds fewer s32 metadata
    ops per layer (the tiles, work list and columns moved out, where XLA
    CSEs them across layers)."""
    off = _branch_metadata_ops(2, monkeypatch, hoist=False)
    on = _branch_metadata_ops(2, monkeypatch, hoist=True)
    assert on < off, (on, off)
