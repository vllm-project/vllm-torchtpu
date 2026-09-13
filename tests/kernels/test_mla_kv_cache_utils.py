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
"""Insert-layout tests for the sparse-MLA KV cache writers.

Pure jnp scatter, no Pallas: these run on CPU. `test_mla_kv_cache_scatter.py`
checks the SparseCore Pallas writer against the same layout contract.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    SKIP_ROW, WORD_BYTES, KVCacheLayout, KVCacheType, SparseMLAKVCacheSpec,
    as_token_bytes, get_dst_rows, pack_tokens, update_sparse_mla_kv_cache_dcp,
    update_sparse_mla_kv_cache_jax)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel

LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 32
PAGES_PER_SEQ = 4
TOTAL_PAGES = 16

# Layouts pinned explicitly rather than read from the environment: these
# tests check the exact geometry dsa_gather is written against.
KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
NOPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.NOPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, LKV_DIM, KV_PACKING)
ROPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.ROPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, ROPE_DIM, KV_PACKING)


def _empty(spec: SparseMLAKVCacheSpec) -> jax.Array:
    return jnp.zeros(spec.shape, spec.jax_dtype)


def unpack4(words_u32: np.ndarray) -> np.ndarray:
    """Unpack uint32 words `[..., n]` into four uint8 bands `[..., 4, n]`."""
    w = np.asarray(words_u32).astype(np.uint32)
    bands = [((w >> (8 * b)) & 0xFF).astype(np.uint8) for b in range(4)]
    return np.stack(bands, axis=-2)


def token_bytes(cache: jax.Array, spec: SparseMLAKVCacheSpec) -> np.ndarray:
    """Decode any layout to `[num_pages, page_size, token_bytes]` uint8.

    Decoding is written from the layout contract, not from the writer: a
    sparsecore word `j` holds byte `b` of band `b`, so `unpack4`'s band axis
    lands directly in front of the word axis and flattens back to token-major
    with no transpose. A writer that packed in any other order fails the
    comparisons below.
    """
    rows = np.asarray(cache).reshape(spec.num_pages, spec.page_size, -1)
    if spec.layout is KVCacheLayout.SPARSECORE:
        # nope [P, S, 128] u32 and rope [P, S // 4, 128] u32.
        rows = unpack4(rows)
    # TENSORCORE nope [P, S, 4, 128] and rope [P, S // 4, 4, 128] are already
    # byte images -- the reshape above is the whole decode.
    return rows.reshape(spec.num_pages, spec.page_size, spec.token_bytes)


def _quantize_fp8(x: np.ndarray, k_scale: float) -> jax.Array:
    return jnp.asarray(x / k_scale).astype(jnp.float8_e4m3fn)


class UpdateSparseMLAKvCacheTest(parameterized.TestCase):
    """Insert-layout unit test; pure jnp scatter, runs on CPU too."""

    @parameterized.named_parameters(
        dict(testcase_name="nope_sc_rope_tc",
             nope_layout=KVCacheLayout.SPARSECORE,
             rope_layout=KVCacheLayout.TENSORCORE),
        dict(testcase_name="nope_sc_rope_sc",
             nope_layout=KVCacheLayout.SPARSECORE,
             rope_layout=KVCacheLayout.SPARSECORE),
        dict(testcase_name="nope_tc_rope_sc",
             nope_layout=KVCacheLayout.TENSORCORE,
             rope_layout=KVCacheLayout.SPARSECORE),
        dict(testcase_name="nope_tc_rope_tc",
             nope_layout=KVCacheLayout.TENSORCORE,
             rope_layout=KVCacheLayout.TENSORCORE),
    )
    def test_rows_land_at_page_slot(self, nope_layout, rope_layout):
        """Every layout combination puts a token's bytes at the same
        (page, slot); only the address arithmetic differs."""
        nope_spec = SparseMLAKVCacheSpec.create(KVCacheType.NOPE, nope_layout,
                                                TOTAL_PAGES, PAGE_SIZE,
                                                LKV_DIM, KV_PACKING)
        rope_spec = SparseMLAKVCacheSpec.create(KVCacheType.ROPE, rope_layout,
                                                TOTAL_PAGES, PAGE_SIZE,
                                                ROPE_DIM, KV_PACKING)
        rng = np.random.default_rng(7)
        q_lens, seq_lens = [5, 3], [37, 3]  # seq 0 appends across a page edge
        total = sum(q_lens)
        kv_c = _quantize_fp8(
            rng.standard_normal((total, LKV_DIM)).astype(np.float32), 1.0)
        k_pe = _quantize_fp8(
            rng.standard_normal((total, ROPE_DIM)).astype(np.float32), 1.0)
        block_tables = np.full((2, PAGES_PER_SEQ), 7, np.int32)
        block_tables[0, :2] = [5, 2]
        block_tables[1, 0] = 9

        nope_cache, rope_cache = update_sparse_mla_kv_cache_jax(
            _empty(nope_spec),
            _empty(rope_spec),
            kv_c,
            k_pe,
            jnp.asarray(seq_lens, jnp.int32),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray([0, 5, 8], jnp.int32),
            nope_spec=nope_spec,
            rope_spec=rope_spec)

        # Every layout is token-major once decoded to [page, slot, bytes].
        nope_rows = token_bytes(nope_cache, nope_spec)
        rope_rows = token_bytes(rope_cache, rope_spec)
        exp_nope = np.asarray(jax.lax.bitcast_convert_type(kv_c, jnp.uint8))
        exp_rope = np.asarray(jax.lax.bitcast_convert_type(k_pe, jnp.uint8))
        # seq 0: positions 32..36 -> page 2 slots 0..4; seq 1: page 9 slots 0..2.
        placements = [(2, i, i) for i in range(5)]
        placements += [(9, i, 5 + i) for i in range(3)]
        touched = np.zeros((TOTAL_PAGES, PAGE_SIZE), bool)
        for page, slot, token in placements:
            np.testing.assert_array_equal(nope_rows[page, slot],
                                          exp_nope[token])
            np.testing.assert_array_equal(rope_rows[page, slot, :ROPE_DIM],
                                          exp_rope[token])
            # Pad tail is never written (zero fill).
            self.assertEqual(
                int(np.count_nonzero(rope_rows[page, slot, ROPE_DIM:])), 0)
            touched[page, slot] = True
        self.assertEqual(int(np.count_nonzero(nope_rows[~touched])), 0)
        self.assertEqual(int(np.count_nonzero(rope_rows[~touched])), 0)

    def test_padded_batch_is_dropped(self):
        """Padded tokens (valid=False) must be dropped, never written."""
        rng = np.random.default_rng(42)
        total_tokens, valid_tokens = 16, 6
        kv_c = _quantize_fp8(
            rng.standard_normal((total_tokens, LKV_DIM)).astype(np.float32),
            1.0)
        k_pe = _quantize_fp8(
            rng.standard_normal((total_tokens, ROPE_DIM)).astype(np.float32),
            1.0)
        block_tables = np.full((2, PAGES_PER_SEQ), 7, np.int32)
        block_tables[0, 0] = 1
        block_tables[1, 0] = 2

        nope_cache, rope_cache = update_sparse_mla_kv_cache_jax(
            _empty(NOPE_SPEC),
            _empty(ROPE_SPEC),
            kv_c,
            k_pe,
            jnp.asarray([4, 2], jnp.int32),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray([0, 4, valid_tokens], jnp.int32),
            nope_spec=NOPE_SPEC,
            rope_spec=ROPE_SPEC)

        nope_rows = token_bytes(nope_cache, NOPE_SPEC).reshape(-1, LKV_DIM)
        rope_rows = token_bytes(rope_cache,
                                ROPE_SPEC).reshape(-1, ROPE_SPEC.token_bytes)
        # Exactly the valid tokens landed, nothing else.
        self.assertEqual(int(nope_rows.any(axis=1).sum()), valid_tokens)
        self.assertEqual(int(rope_rows.any(axis=1).sum()), valid_tokens)

    @parameterized.named_parameters(
        dict(testcase_name="nope",
             cache_type=KVCacheType.NOPE,
             head_dim=LKV_DIM),
        dict(testcase_name="rope",
             cache_type=KVCacheType.ROPE,
             head_dim=ROPE_DIM),
    )
    def test_sparsecore_words_are_little_endian_pack4(self, cache_type,
                                                      head_dim):
        """Each sparsecore word is pack4 of the token's four byte bands."""
        specs = {
            layout:
            SparseMLAKVCacheSpec.create(cache_type, layout, TOTAL_PAGES,
                                        PAGE_SIZE, head_dim, KV_PACKING)
            for layout in KVCacheLayout
        }
        is_nope = cache_type is KVCacheType.NOPE
        rng = np.random.default_rng(11)
        num_tokens = 6
        kv_c = _quantize_fp8(
            rng.standard_normal((num_tokens, LKV_DIM)).astype(np.float32), 1.0)
        k_pe = _quantize_fp8(
            rng.standard_normal((num_tokens, ROPE_DIM)).astype(np.float32),
            1.0)
        values = kv_c if is_nope else k_pe
        block_tables = np.full((1, PAGES_PER_SEQ), 3, np.int32)

        caches = {}
        for layout, spec in specs.items():
            nope_spec = spec if is_nope else NOPE_SPEC
            rope_spec = ROPE_SPEC if is_nope else spec
            nope, rope = update_sparse_mla_kv_cache_jax(
                _empty(nope_spec),
                _empty(rope_spec),
                kv_c,
                k_pe,
                jnp.asarray([num_tokens], jnp.int32),
                jnp.asarray(block_tables.reshape(-1)),
                jnp.asarray([0, num_tokens], jnp.int32),
                nope_spec=nope_spec,
                rope_spec=rope_spec)
            caches[layout] = nope if is_nope else rope

        # Reference: pack the four bands of each written token by hand.
        sc_spec = specs[KVCacheLayout.SPARSECORE]
        sc_words = np.asarray(caches[KVCacheLayout.SPARSECORE]).reshape(
            TOTAL_PAGES, PAGE_SIZE, -1)
        raw = np.asarray(jax.lax.bitcast_convert_type(values, jnp.uint8))
        padded = np.zeros((num_tokens, sc_spec.token_bytes), np.uint8)
        padded[:, :raw.shape[1]] = raw
        bands = padded.reshape(num_tokens, WORD_BYTES, -1).astype(np.uint32)
        expected = (bands[:, 0] | (bands[:, 1] << 8) | (bands[:, 2] << 16)
                    | (bands[:, 3] << 24))
        for token in range(num_tokens):
            np.testing.assert_array_equal(sc_words[3, token], expected[token])

        # ...and the two layouts hold the same bytes, differing only in how
        # the 32-bit packing is expressed.
        np.testing.assert_array_equal(
            token_bytes(caches[KVCacheLayout.SPARSECORE],
                        specs[KVCacheLayout.SPARSECORE]),
            token_bytes(caches[KVCacheLayout.TENSORCORE],
                        specs[KVCacheLayout.TENSORCORE]))


class PackingTest(parameterized.TestCase):
    """The byte-packing helpers both writers share."""

    def test_pack_tokens_round_trips_through_unpack4(self):
        raw = jnp.arange(16, dtype=jnp.uint8).reshape(1, 16)
        packed = pack_tokens(raw, token_bytes=16)
        self.assertEqual(packed.shape, (1, 4))
        # word j == bytes {j, 4 + j, 8 + j, 12 + j}, little-endian.
        np.testing.assert_array_equal(
            np.asarray(packed),
            np.array([[0x0C080400, 0x0D090501, 0x0E0A0602, 0x0F0B0703]],
                     np.uint32))
        np.testing.assert_array_equal(
            unpack4(np.asarray(packed)).reshape(1, 16), np.asarray(raw))

    def test_as_token_bytes_zero_pads_the_tail(self):
        raw = jnp.full((3, 64), 0xAB, jnp.uint8)
        out = np.asarray(
            as_token_bytes(jax.lax.bitcast_convert_type(
                raw, jnp.float8_e4m3fn),
                           token_bytes=128))
        self.assertEqual(out.shape, (3, 128))
        np.testing.assert_array_equal(out[:, :64], np.asarray(raw))
        self.assertEqual(int(np.count_nonzero(out[:, 64:])), 0)

    @parameterized.named_parameters(
        dict(testcase_name="nope",
             cache_type=KVCacheType.NOPE,
             head_dim=LKV_DIM),
        dict(testcase_name="rope",
             cache_type=KVCacheType.ROPE,
             head_dim=ROPE_DIM),
    )
    def test_spec_shape_is_a_retiling_of_token_bytes(self, cache_type,
                                                     head_dim):
        """Every layout reserves exactly `token_bytes` per token slot."""
        for layout in KVCacheLayout:
            spec = SparseMLAKVCacheSpec.create(cache_type, layout, TOTAL_PAGES,
                                               PAGE_SIZE, head_dim, KV_PACKING)
            cache = _empty(spec)
            self.assertEqual(
                cache.nbytes,
                spec.num_pages * spec.page_size * spec.token_bytes,
                msg=f"{cache_type} {layout} {spec.shape} {spec.jax_dtype}")

    def test_get_dst_rows_flattens_page_and_slot(self):
        """`dst_rows` is `page * page_size + slot`, and SKIP_ROW past the end."""
        block_tables = np.full((2, PAGES_PER_SEQ), 7, np.int32)
        block_tables[0, 1] = 5
        block_tables[1, 0] = 9
        # seq 0: len 37, 5 new tokens -> positions 32..36 on page 5 (idx 1).
        # seq 1: len 3, 3 new tokens -> positions 0..2 on page 9.
        dst_rows = np.asarray(
            get_dst_rows(num_tokens=10,
                         seq_lens=jnp.asarray([37, 3], jnp.int32),
                         block_tables=jnp.asarray(block_tables.reshape(-1)),
                         query_start_loc=jnp.asarray([0, 5, 8], jnp.int32),
                         page_size=PAGE_SIZE))
        expected = [5 * PAGE_SIZE + i for i in range(5)]
        expected += [9 * PAGE_SIZE + i for i in range(3)]
        expected += [SKIP_ROW, SKIP_ROW]  # tokens 8, 9 are batch padding
        np.testing.assert_array_equal(dst_rows, np.array(expected, np.int32))


def _interleave_placement(pos: np.ndarray, dcp_size: int, interleave_size: int,
                          page_size: int) -> tuple[np.ndarray, ...]:
    """Where the documented interleave says global position `pos` lives.

    Written from the contract in the docstring rather than from the writer, so
    the two are independent: rank `r` owns the `interleave_size`-wide chunk at
    offset `r * interleave_size` in every `dcp_size * interleave_size` cycle,
    and the positions a rank owns keep their global order inside its shard.
    Splitting that rank-local index by `page_size` is what has to come out
    equal to the (virtual page, slot) the writer computes.

    Returns `(owner, virtual_page, slot)`.
    """
    owner = (pos // interleave_size) % dcp_size
    rank_local = ((pos // (dcp_size * interleave_size)) * interleave_size +
                  pos % interleave_size)
    return owner, rank_local // page_size, rank_local % page_size


class UpdateSparseMLAKvCacheDcpTest(parameterized.TestCase):
    """Ownership masking for the DCP position-sharded writer.

    No `shard_map` and no second device: `dcp_rank` is a plain argument, so a
    DCP group is simulated by calling the writer once per rank in one process,
    each call against its own freshly-zeroed shard pair and the *same*
    replicated rows. That is exactly what each shard sees under `shard_map`,
    and it is only possible because the writer reaches for no group state.
    """

    # Pages are per rank here; the global cache is `dcp_size` times this.
    PAGES_PER_SEQ = 4

    def _inputs(self,
                dcp_size,
                interleave_size,
                seed=3,
                pad=6,
                out_of_range_ordinal=True):
        """Two sequences whose positions cover every rank several times.

        Sequence 0 is a chunked-prefill continuation -- it starts partway into
        a cycle rather than at zero -- which is where a writer that derives
        `pos` from the token index instead of `seq_lens` goes wrong.
        """
        vps = PAGE_SIZE * dcp_size  # positions per virtual page
        cycle = dcp_size * interleave_size
        seq_lens = np.asarray([3 * vps + 11, cycle + 3], np.int32)
        q_lens = np.asarray([2 * cycle + 5, cycle + 3], np.int32)
        query_start_loc = np.concatenate([[0],
                                          np.cumsum(q_lens)]).astype(np.int32)
        num_valid = int(query_start_loc[-1])

        pos = np.concatenate([
            np.arange(seq_lens[i] - q_lens[i], seq_lens[i], dtype=np.int64)
            for i in range(len(q_lens))
        ])
        seq_of = np.repeat(np.arange(len(q_lens)), q_lens)

        # Virtual page ordinals. Distinct mod TOTAL_PAGES so no two virtual
        # pages alias, and one deliberately >= TOTAL_PAGES to pin that the
        # writer resolves it into the local shard the way the readers do.
        block_tables = np.zeros((2, self.PAGES_PER_SEQ), np.int32)
        block_tables[0] = [5, 2, 7, 1]
        block_tables[1] = [9 + TOTAL_PAGES * out_of_range_ordinal, 3, 0, 4]

        rng = np.random.default_rng(seed)
        total = num_valid + pad
        kv_c = _quantize_fp8(
            rng.standard_normal((total, LKV_DIM)).astype(np.float32), 1.0)
        k_pe = _quantize_fp8(
            rng.standard_normal((total, ROPE_DIM)).astype(np.float32), 1.0)
        return dict(kv_c=kv_c,
                    k_pe=k_pe,
                    seq_lens=jnp.asarray(seq_lens),
                    block_tables=jnp.asarray(block_tables.reshape(-1)),
                    query_start_loc=jnp.asarray(query_start_loc),
                    pos=pos,
                    seq_of=seq_of,
                    num_valid=num_valid,
                    block_tables_2d=block_tables)

    def _run_group(self, inputs, dcp_size, interleave_size, nope_spec,
                   rope_spec):
        """Every rank's shards after the whole group has written."""
        return [
            update_sparse_mla_kv_cache_dcp(_empty(nope_spec),
                                           _empty(rope_spec),
                                           inputs["kv_c"],
                                           inputs["k_pe"],
                                           inputs["seq_lens"],
                                           inputs["block_tables"],
                                           inputs["query_start_loc"],
                                           jnp.int32(rank),
                                           nope_spec=nope_spec,
                                           rope_spec=rope_spec,
                                           dcp_size=dcp_size,
                                           interleave_size=interleave_size)
            for rank in range(dcp_size)
        ]

    @parameterized.named_parameters(
        dict(testcase_name="d2_i4_tc", dcp_size=2, interleave_size=4),
        dict(testcase_name="d2_i32_tc", dcp_size=2, interleave_size=32),
        dict(testcase_name="d4_i4_tc", dcp_size=4, interleave_size=4),
        dict(testcase_name="d4_i8_tc", dcp_size=4, interleave_size=8),
        dict(testcase_name="d4_i8_sc",
             dcp_size=4,
             interleave_size=8,
             layout=KVCacheLayout.SPARSECORE),
        dict(testcase_name="d8_i16_sc",
             dcp_size=8,
             interleave_size=16,
             layout=KVCacheLayout.SPARSECORE),
    )
    def test_rows_land_where_the_interleave_says(self,
                                                 dcp_size,
                                                 interleave_size,
                                                 layout=KVCacheLayout.
                                                 TENSORCORE):
        """Each row reaches exactly one rank, at the reference (page, slot).

        The count check is the other half: matching every token proves nothing
        was dropped or misplaced, and the total proves nothing was written
        twice -- together that is ownership being a partition, which is what
        the indexer's owner-filtered return leg assumes.
        """
        nope_spec = SparseMLAKVCacheSpec.create(KVCacheType.NOPE, layout,
                                                TOTAL_PAGES, PAGE_SIZE,
                                                LKV_DIM, KV_PACKING)
        rope_spec = SparseMLAKVCacheSpec.create(KVCacheType.ROPE, layout,
                                                TOTAL_PAGES, PAGE_SIZE,
                                                ROPE_DIM, KV_PACKING)
        inputs = self._inputs(dcp_size, interleave_size)
        shards = self._run_group(inputs, dcp_size, interleave_size, nope_spec,
                                 rope_spec)

        nope_rows = [token_bytes(n, nope_spec) for n, _ in shards]
        rope_rows = [token_bytes(r, rope_spec) for _, r in shards]
        exp_nope = np.asarray(
            jax.lax.bitcast_convert_type(inputs["kv_c"], jnp.uint8))
        exp_rope = np.asarray(
            jax.lax.bitcast_convert_type(inputs["k_pe"], jnp.uint8))

        owner, virtual_page, slot = _interleave_placement(
            inputs["pos"], dcp_size, interleave_size, PAGE_SIZE)
        seen_ranks = set()
        for token in range(inputs["num_valid"]):
            r = int(owner[token])
            page = int(inputs["block_tables_2d"][inputs["seq_of"][token],
                                                 virtual_page[token]] %
                       TOTAL_PAGES)
            s = int(slot[token])
            np.testing.assert_array_equal(nope_rows[r][page, s],
                                          exp_nope[token])
            np.testing.assert_array_equal(rope_rows[r][page, s, :ROPE_DIM],
                                          exp_rope[token])
            seen_ranks.add(r)
        self.assertEqual(seen_ranks, set(range(dcp_size)),
                         "the sample must exercise every rank")

        written = sum(
            int(rows.reshape(-1, rows.shape[-1]).any(axis=1).sum())
            for rows in nope_rows)
        self.assertEqual(written, inputs["num_valid"])

    @parameterized.named_parameters(
        dict(testcase_name="i4", interleave_size=4),
        dict(testcase_name="i32", interleave_size=32),
    )
    def test_a_group_of_one_is_the_plain_writer(self, interleave_size):
        """At `dcp_size=1` the mask is vacuous, so the twins must agree.

        Bit-for-bit, for any legal interleave -- the sharded path collapses to
        the same (page, slot) arithmetic rather than to something that merely
        holds the same tokens. Against the pure-XLA golden, because that is the
        twin the DCP writer still is: it has yet to be routed through the
        SparseCore scatter kernel the unsharded writer now uses.

        Every ordinal is kept in range, because that is the one place the two
        genuinely differ: `block_tables` holds *virtual* ordinals here, so an
        ordinal past the end of the shard is wrapped into a live page rather
        than dropped as out of bounds the way the unsharded writer drops it.
        """
        inputs = self._inputs(1, interleave_size, out_of_range_ordinal=False)
        args = (_empty(NOPE_SPEC), _empty(ROPE_SPEC), inputs["kv_c"],
                inputs["k_pe"], inputs["seq_lens"], inputs["block_tables"],
                inputs["query_start_loc"])
        plain = update_sparse_mla_kv_cache_jax(*args,
                                               nope_spec=NOPE_SPEC,
                                               rope_spec=ROPE_SPEC)
        sharded = update_sparse_mla_kv_cache_dcp(
            *args,
            jnp.int32(0),
            nope_spec=NOPE_SPEC,
            rope_spec=ROPE_SPEC,
            dcp_size=1,
            interleave_size=interleave_size)
        for got, want in zip(sharded, plain):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_padded_rows_are_dropped_on_every_rank(self):
        """A padded row belongs to no position, so no rank may claim it.

        `owner` is computed from `pos`, which is garbage for a padded row, so
        without the validity term some rank would find itself the owner.
        """
        dcp_size, interleave_size = 4, 8
        inputs = self._inputs(dcp_size, interleave_size, pad=11)
        shards = self._run_group(inputs, dcp_size, interleave_size, NOPE_SPEC,
                                 ROPE_SPEC)
        written = sum(
            int(
                token_bytes(n, NOPE_SPEC).reshape(-1, LKV_DIM).any(
                    axis=1).sum()) for n, _ in shards)
        self.assertEqual(written, inputs["num_valid"])

    def test_an_interleave_that_splits_a_rope_tile_is_rejected(self):
        """A rope tile is `WORD_BYTES` tokens; it cannot span two ranks."""
        inputs = self._inputs(2, 4)
        with self.assertRaisesRegex(ValueError, "WORD_BYTES"):
            self._run_group(inputs, 2, WORD_BYTES + 2, NOPE_SPEC, ROPE_SPEC)

    def test_an_interleave_that_straddles_a_page_is_rejected(self):
        """The guard that makes the reference placement above correct.

        `owner = (pos // interleave) % dcp_size` only agrees with the writer's
        within-virtual-page arithmetic when the interleave divides the page,
        so this is load-bearing rather than tidiness.
        """
        self.assertTrue(PAGE_SIZE % 12)
        inputs = self._inputs(2, 4)
        with self.assertRaisesRegex(ValueError, "must divide"):
            self._run_group(inputs, 2, 12, NOPE_SPEC, ROPE_SPEC)

    def test_a_cache_that_does_not_match_its_spec_is_rejected(self):
        """Both writers validate identically; the DCP one used to check less."""
        inputs = self._inputs(2, 4)
        with self.assertRaisesRegex(AssertionError, "nope cache"):
            update_sparse_mla_kv_cache_dcp(jnp.zeros(NOPE_SPEC.shape,
                                                     jnp.bfloat16),
                                           _empty(ROPE_SPEC),
                                           inputs["kv_c"],
                                           inputs["k_pe"],
                                           inputs["seq_lens"],
                                           inputs["block_tables"],
                                           inputs["query_start_loc"],
                                           jnp.int32(0),
                                           nope_spec=NOPE_SPEC,
                                           rope_spec=ROPE_SPEC,
                                           dcp_size=2,
                                           interleave_size=4)
