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
"""GLM5_MKD_STACK_C: stacked [n_layers, num_blocks, ...] KV caches on the DCP path.

Pure jnp, no Pallas: these run on CPU. The DCP writers are simulated one rank
at a time (`dcp_rank` is a plain argument), the same way
`test_mla_kv_cache_utils.py` does. `test_mla_kv_stack_dcp_tpu.py` checks the
real ops (SparseCore writer, sparse attention kernel, indexer) on TPU.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    get_dcp_page_and_slot,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    OOB_PAGE,
    SPARSECORE_MAX_TENSOR_BYTES,
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
    glm5_stack_group_sizes,
    glm5_stacked_kv_kinds,
    merge_kv_stack,
    split_kv_stack,
    stack_page_table_dcp,
    update_sparse_mla_kv_cache_dcp_jax,
)

LKV_DIM = 512
ROPE_DIM = 64
KV_PACKING = 4  # fp8


def _random_like(shape, dtype, seed):
    """Nonzero prior content: a write landing in the wrong row shows up."""
    rng = np.random.default_rng(seed)
    if jnp.dtype(dtype) == jnp.uint32:
        return jnp.asarray(rng.integers(0, 2**32, shape, dtype=np.uint64), jnp.uint32)
    return jnp.asarray(rng.integers(0, 256, shape, dtype=np.int64), jnp.uint8)


class MergeSplitTest(parameterized.TestCase):
    def test_merge_is_row_major_page_pool(self):
        stack = jnp.arange(3 * 5 * 4 * 8, dtype=jnp.uint32).reshape(3, 5, 4, 8)
        for row in range(3):
            merged = merge_kv_stack(stack, (row, 5))
            self.assertEqual(merged.shape, (15, 4, 8))
            for page in range(5):
                np.testing.assert_array_equal(
                    np.asarray(merged[row * 5 + page]), np.asarray(stack[row, page])
                )
            np.testing.assert_array_equal(
                np.asarray(split_kv_stack(merged, stack.shape)), np.asarray(stack)
            )

    def test_merge_rejects_a_foreign_stack(self):
        stack = jnp.zeros((3, 5, 4, 8), jnp.uint8)
        with self.assertRaisesRegex(ValueError, "kv_stack"):
            merge_kv_stack(stack, (0, 4))  # wrong num_blocks
        with self.assertRaisesRegex(ValueError, "kv_stack"):
            merge_kv_stack(stack, (3, 5))  # row out of range


class StackPageTableTest(parameterized.TestCase):
    @parameterized.parameters((1, 7), (3, 7), (21, 16), (78, 5))
    def test_resolves_to_the_per_layer_page_in_this_row(self, n_layers, num_blocks):
        """`id % (n * nb)` of the rewritten table == row * nb + `id % nb`.

        That is the resolution every DCP reader and writer applies; the table
        holds in-range ids, ids >= num_blocks and negative padding.
        """
        rng = np.random.default_rng(n_layers)
        table = np.concatenate(
            [
                rng.integers(0, num_blocks, 32),
                rng.integers(num_blocks, 10 * num_blocks, 8),
                [-1, -2, -num_blocks, 0, num_blocks - 1, num_blocks],
            ]
        ).astype(np.int32)
        for row in range(n_layers):
            stacked = np.asarray(
                stack_page_table_dcp(jnp.asarray(table), (row, num_blocks))
            )
            self.assertTrue(
                (
                    (stacked >= row * num_blocks) & (stacked < (row + 1) * num_blocks)
                ).all()
            )
            resolved = np.mod(stacked, n_layers * num_blocks)
            np.testing.assert_array_equal(
                resolved, row * num_blocks + np.mod(table, num_blocks)
            )


def _dcp_inputs(dcp_size, interleave_size, page_size, pages_per_seq, num_blocks):
    """Two sequences: a chunked-prefill continuation and a decode/verify-sized
    tail, positions covering every rank; block table with an id >= num_blocks."""
    vps = page_size * dcp_size
    cycle = dcp_size * interleave_size
    seq_lens = np.asarray([2 * vps + 11, cycle + 3, 0], np.int32)
    q_lens = np.asarray([2 * cycle + 5, 4, 0], np.int32)
    qsl = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    total = int(qsl[-1]) + 6  # batch padding rows
    table = np.zeros((3, pages_per_seq), np.int32)
    perm = np.random.default_rng(5).permutation(num_blocks)
    table[0] = perm[:pages_per_seq]
    table[1] = perm[pages_per_seq : 2 * pages_per_seq]
    table[1, 0] += num_blocks  # a virtual id past the pool: resolved by `%`
    rng = np.random.default_rng(11)
    kv_c = jnp.asarray(rng.standard_normal((total, LKV_DIM)), jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    k_pe = jnp.asarray(rng.standard_normal((total, ROPE_DIM)), jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    return dict(
        kv_c=kv_c,
        k_pe=k_pe,
        seq_lens=jnp.asarray(seq_lens),
        block_tables=jnp.asarray(table.reshape(-1)),
        query_start_loc=jnp.asarray(qsl),
    )


class StackedDcpWriterTest(parameterized.TestCase):
    """`update_sparse_mla_kv_cache_dcp` on the merged stack writes exactly what
    the per-layer writer writes, into this layer's row, and nothing else."""

    @parameterized.named_parameters(
        dict(testcase_name="d4_i4_sc", dcp_size=4, layout=KVCacheLayout.SPARSECORE),
        dict(testcase_name="d8_i4_sc", dcp_size=8, layout=KVCacheLayout.SPARSECORE),
        dict(testcase_name="d4_i4_tc", dcp_size=4, layout=KVCacheLayout.TENSORCORE),
    )
    def test_stacked_matches_per_layer(self, dcp_size, layout):
        n_layers, num_blocks, page_size, interleave = 3, 12, 32, 4
        nope_spec = SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE, layout, num_blocks, page_size, LKV_DIM, KV_PACKING
        )
        rope_spec = SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE, layout, num_blocks, page_size, ROPE_DIM, KV_PACKING
        )
        merged_nope_spec = dataclasses.replace(
            nope_spec, num_pages=n_layers * num_blocks
        )
        merged_rope_spec = dataclasses.replace(
            rope_spec, num_pages=n_layers * num_blocks
        )
        inputs = _dcp_inputs(dcp_size, interleave, page_size, 4, num_blocks)
        for rank in range(dcp_size):
            nope_stack = _random_like(
                (n_layers, *nope_spec.shape), nope_spec.jax_dtype, rank
            )
            rope_stack = _random_like(
                (n_layers, *rope_spec.shape), rope_spec.jax_dtype, 100 + rank
            )
            for row in range(n_layers):
                kv_stack = (row, num_blocks)
                want_nope, want_rope = update_sparse_mla_kv_cache_dcp_jax(
                    nope_stack[row],
                    rope_stack[row],
                    inputs["kv_c"],
                    inputs["k_pe"],
                    inputs["seq_lens"],
                    inputs["block_tables"],
                    inputs["query_start_loc"],
                    jnp.int32(rank),
                    nope_spec=nope_spec,
                    rope_spec=rope_spec,
                    dcp_size=dcp_size,
                    interleave_size=interleave,
                )
                got_nope, got_rope = update_sparse_mla_kv_cache_dcp_jax(
                    merge_kv_stack(nope_stack, kv_stack),
                    merge_kv_stack(rope_stack, kv_stack),
                    inputs["kv_c"],
                    inputs["k_pe"],
                    inputs["seq_lens"],
                    stack_page_table_dcp(inputs["block_tables"], kv_stack),
                    inputs["query_start_loc"],
                    jnp.int32(rank),
                    nope_spec=merged_nope_spec,
                    rope_spec=merged_rope_spec,
                    dcp_size=dcp_size,
                    interleave_size=interleave,
                )
                got_nope = np.asarray(split_kv_stack(got_nope, nope_stack.shape))
                got_rope = np.asarray(split_kv_stack(got_rope, rope_stack.shape))
                # The served form: the per-layer table + kv_stack (pages
                # resolved per layer, then offset to the row) writes the same
                # bytes as the rewritten table, everywhere in the stack.
                kw_nope, kw_rope = update_sparse_mla_kv_cache_dcp_jax(
                    merge_kv_stack(nope_stack, kv_stack),
                    merge_kv_stack(rope_stack, kv_stack),
                    inputs["kv_c"],
                    inputs["k_pe"],
                    inputs["seq_lens"],
                    inputs["block_tables"],
                    inputs["query_start_loc"],
                    jnp.int32(rank),
                    nope_spec=merged_nope_spec,
                    rope_spec=merged_rope_spec,
                    dcp_size=dcp_size,
                    interleave_size=interleave,
                    kv_stack=kv_stack,
                )
                np.testing.assert_array_equal(
                    np.asarray(split_kv_stack(kw_nope, nope_stack.shape)), got_nope
                )
                np.testing.assert_array_equal(
                    np.asarray(split_kv_stack(kw_rope, rope_stack.shape)), got_rope
                )
                self.assertFalse(
                    np.array_equal(np.asarray(want_nope), np.asarray(nope_stack[row])),
                    "degenerate: this rank wrote nothing",
                )
                np.testing.assert_array_equal(got_nope[row], np.asarray(want_nope))
                np.testing.assert_array_equal(got_rope[row], np.asarray(want_rope))
                for other in range(n_layers):
                    if other != row:
                        np.testing.assert_array_equal(
                            got_nope[other], np.asarray(nope_stack[other])
                        )
                        np.testing.assert_array_equal(
                            got_rope[other], np.asarray(rope_stack[other])
                        )


class StackedIndexerPageSlotTest(parameterized.TestCase):
    """The indexer writer's resolver (`get_dcp_page_and_slot`) on the merged
    stack: same slot, page offset by row * num_blocks, OOB rows stay OOB."""

    @parameterized.parameters(4, 8, 16)
    def test_offset_pages(self, dcp_size):
        n_layers, num_blocks, page_size, interleave = 21, 12, 256, 4
        inputs = _dcp_inputs(dcp_size, interleave, page_size, 4, num_blocks)
        num_tokens = inputs["kv_c"].shape[0]
        for rank in range(dcp_size):
            want_page, want_slot = get_dcp_page_and_slot(
                num_tokens,
                inputs["seq_lens"],
                inputs["block_tables"],
                inputs["query_start_loc"],
                jnp.int32(rank),
                page_size=page_size,
                num_pages=num_blocks,
                dcp_size=dcp_size,
                interleave_size=interleave,
            )
            want_page, want_slot = np.asarray(want_page), np.asarray(want_slot)
            valid = want_page != OOB_PAGE
            for row in (0, 7, n_layers - 1):
                page, slot = get_dcp_page_and_slot(
                    num_tokens,
                    inputs["seq_lens"],
                    stack_page_table_dcp(inputs["block_tables"], (row, num_blocks)),
                    inputs["query_start_loc"],
                    jnp.int32(rank),
                    page_size=page_size,
                    num_pages=n_layers * num_blocks,
                    dcp_size=dcp_size,
                    interleave_size=interleave,
                )
                page, slot = np.asarray(page), np.asarray(slot)
                np.testing.assert_array_equal(page[~valid], want_page[~valid])
                np.testing.assert_array_equal(
                    page[valid], want_page[valid] + row * num_blocks
                )
                np.testing.assert_array_equal(slot[valid], want_slot[valid])


class StackKindsEnvTest(parameterized.TestCase):
    def test_parse(self):
        self.assertEqual(glm5_stacked_kv_kinds(""), frozenset())
        self.assertEqual(glm5_stacked_kv_kinds("0"), frozenset())
        self.assertEqual(glm5_stacked_kv_kinds("1"), frozenset({"nope", "rope", "idx"}))
        self.assertEqual(glm5_stacked_kv_kinds("idx"), frozenset({"idx"}))
        self.assertEqual(
            glm5_stacked_kv_kinds("nope,rope"), frozenset({"nope", "rope"})
        )
        with self.assertRaisesRegex(ValueError, "together"):
            glm5_stacked_kv_kinds("nope")
        with self.assertRaisesRegex(ValueError, "unknown"):
            glm5_stacked_kv_kinds("kv")


class StackGroupSizesTest(parameterized.TestCase):
    def test_glm52_full_depth_nope_splits_in_two(self):
        # 78 layers x 2085 pages x 128 KiB = 20.8 GB > the 16 GiB SC limit.
        per_layer = 2085 * 256 * 128 * 4
        self.assertEqual(glm5_stack_group_sizes(78, per_layer), [39, 39])
        # rope (32 KiB pages) and the 21 indexer layers fit in one tensor.
        self.assertEqual(glm5_stack_group_sizes(78, 2085 * 64 * 128 * 4), [78])
        self.assertEqual(glm5_stack_group_sizes(21, 2085 * 33 * 4 * 256), [21])

    @parameterized.parameters((78, 7, 100), (10, 3, 9), (5, 1, 1), (21, 5, 49))
    def test_fewest_balanced_groups_under_the_limit(self, n, per, limit):
        sizes = glm5_stack_group_sizes(n, per, limit)
        self.assertEqual(sum(sizes), n)
        self.assertLessEqual(max(sizes) * per, limit)
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        if len(sizes) > 1:  # one fewer group would not fit
            self.assertGreater(-(-n // (len(sizes) - 1)) * per, limit)

    def test_default_limit_and_oversized_layer(self):
        self.assertEqual(
            glm5_stack_group_sizes(3, SPARSECORE_MAX_TENSOR_BYTES // 3), [3]
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            glm5_stack_group_sizes(2, 11, 10)


class StackedPageSlotNoRewriteTest(parameterized.TestCase):
    """`_get_dcp_page_and_slot(..., kv_stack=)` on the per-layer table resolves
    every token to the page the rewritten table (`stack_page_table_dcp`) does on
    the merged pool -- OOB rows included -- and its pre-offset part is the
    per-layer computation itself (identical in every row: what lets XLA CSE it
    across layers). A wrong row (positive control) lands elsewhere."""

    @parameterized.parameters(4, 8, 16)
    def test_matches_rewritten_table(self, dcp_size):
        from vllm_torchtpu.kernels.mla.kv_cache_utils import _get_dcp_page_and_slot

        n_layers, num_blocks, page_size, interleave = 5, 12, 32, 4
        spec = SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE,
            KVCacheLayout.SPARSECORE,
            num_blocks,
            page_size,
            LKV_DIM,
            KV_PACKING,
        )
        merged = dataclasses.replace(spec, num_pages=n_layers * num_blocks)
        inputs = _dcp_inputs(dcp_size, interleave, page_size, 4, num_blocks)
        num_tokens = inputs["kv_c"].shape[0]
        args = (inputs["seq_lens"], inputs["block_tables"], inputs["query_start_loc"])
        kw = dict(dcp_size=dcp_size, interleave_size=interleave)
        for rank in range(dcp_size):
            base_page, base_slot = _get_dcp_page_and_slot(
                num_tokens, *args, jnp.int32(rank), nope_spec=spec, **kw
            )
            base_page = np.asarray(base_page)
            valid = base_page != OOB_PAGE
            self.assertTrue(valid.any() and (~valid).any(), "degenerate ownership")
            for row in range(n_layers):
                want = _get_dcp_page_and_slot(
                    num_tokens,
                    inputs["seq_lens"],
                    stack_page_table_dcp(inputs["block_tables"], (row, num_blocks)),
                    inputs["query_start_loc"],
                    jnp.int32(rank),
                    nope_spec=merged,
                    **kw,
                )
                got = _get_dcp_page_and_slot(
                    num_tokens,
                    *args,
                    jnp.int32(rank),
                    nope_spec=merged,
                    kv_stack=(row, num_blocks),
                    **kw,
                )
                np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want[0]))
                np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))
                np.testing.assert_array_equal(
                    np.asarray(got[0])[valid], base_page[valid] + row * num_blocks
                )
                np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(base_slot))
                wrong = _get_dcp_page_and_slot(
                    num_tokens,
                    *args,
                    jnp.int32(rank),
                    nope_spec=merged,
                    kv_stack=((row + 1) % n_layers, num_blocks),
                    **kw,
                )
                self.assertFalse(
                    np.array_equal(np.asarray(wrong[0]), np.asarray(want[0])),
                    "positive control: a wrong stack row must move the pages",
                )


class StackedGatherIndicesTest(parameterized.TestCase):
    """The uncapped sparse-MLA row prep on the per-layer geometry + the stack
    row offset (`sparse_ragged_paged_attention(kv_stack=)`) gathers, for every
    valid top-k slot, the row the rewritten table gathers on the merged pool;
    padding slots stay inside this layer's row of the stack, at the per-layer
    padding rows shifted by the row offset."""

    def test_offset_rows(self):
        from jax.experimental.pallas import tpu as pltpu

        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
            self._offset_rows()

    def _offset_rows(self):
        from vllm_torchtpu.kernels.mla.sparse.kernel import prepare_gather_indices

        n_layers, num_blocks, page_size, pps = 5, 12, 32, 4
        rng = np.random.default_rng(3)
        table = rng.permutation(num_blocks)[: 3 * pps].astype(np.int32)
        q_lens = np.asarray([40, 4, 3], np.int32)
        qsl = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
        topk = rng.integers(0, pps * page_size, (48, 64)).astype(np.int32)
        topk[rng.random(topk.shape) < 0.3] = -1
        topk = jnp.asarray(topk)
        local = jnp.mod(jnp.asarray(table), num_blocks)
        _, base = prepare_gather_indices(
            topk, local, qsl, page_size=page_size, total_num_pages=num_blocks
        )
        pad = np.asarray(topk) == -1
        for row in range(n_layers):
            off = row * num_blocks * page_size
            _, want = prepare_gather_indices(
                topk,
                jnp.mod(
                    stack_page_table_dcp(jnp.asarray(table), (row, num_blocks)),
                    n_layers * num_blocks,
                ),
                qsl,
                page_size=page_size,
                total_num_pages=n_layers * num_blocks,
            )
            got = np.asarray(base) + off
            np.testing.assert_array_equal(got[~pad], np.asarray(want)[~pad])
            self.assertTrue(((got >= off) & (got < off + num_blocks * page_size)).all())


def _jaxpr_cone_hashes(closed_jaxpr, eqns):
    """A structural hash of each eqn's whole input cone (primitive, params and
    inputs, recursively; jaxpr inputs by position, constants by value). Two
    eqns hash equal iff they compute the same thing from the same inputs --
    what XLA's CSE merges."""
    import hashlib

    jaxpr = closed_jaxpr.jaxpr
    producer = {}
    for e in jaxpr.eqns:
        for k, o in enumerate(e.outvars):
            producer[o] = (e, k)
    names = {v: f"in{i}" for i, v in enumerate(jaxpr.invars)}
    for v, c in zip(jaxpr.constvars, closed_jaxpr.consts):
        c = np.asarray(c)
        names[v] = "c" + hashlib.md5(c.tobytes() + str(c.shape).encode()).hexdigest()
    memo = {}

    def canon(v):
        if hasattr(v, "val"):  # Literal
            return f"lit({v.val!r})"
        if v in names:
            return names[v]
        if v not in memo:
            e, k = producer[v]
            params = sorted((a, str(b)) for a, b in e.params.items())
            ins = ",".join(canon(i) for i in e.invars)
            memo[v] = hashlib.md5(
                f"{e.primitive.name}{params}({ins})#{k}".encode()
            ).hexdigest()
        return memo[v]

    return [canon(e.outvars[0]) for e in eqns]


class StackedWindowPrepCSETest(parameterized.TestCase):
    """Every capped top-k -> row prep of `sparse_ragged_paged_attention` under
    `kv_stack` -- plain window, live-group (TPU_DCP_MLA_MIXED_LIVE), mixed fused
    (TPU_DCP_MLA_MIXED_FUSED_PREP), fused decode (TPU_DCP_MLA_FUSED_PREP, with
    TPU_MLA_KV_WRITE_IN_GATHER's kv_write) -- is the SAME computation as on a
    per-layer cache, for every stack row: one structural hash over per-layer,
    row 0, row 1 and row 2 sharing a top-k and a page table (so XLA CSEs an
    IndexShare layer's prep with its indexer layer's). The row offset reaches
    the SparseCore gather instead (`row_offset` = row * num_blocks * page
    size, none on row 0 or per-layer caches), and kv_write is handed over
    unchanged (per-layer rows; the gather offsets them). Traced only (the
    gather is stubbed: SparseCore needs a TPU). Positive control: a stack-wide
    page table (the pre-fix form) makes the preps distinct."""

    NB, L, PS, TOPK, CAP = 12, 3, 256, 2048, 512

    def _trace(self, num_tokens, cfg, kv_write, rewrite_table=False):
        from unittest import mock

        import jax
        from jax.experimental.pallas import tpu as pltpu

        from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
        from vllm_torchtpu.kernels.mla.sparse import native_sc_gather

        calls = []

        def stub(nope, rope, indices, *, out_size, kv_write=None, row_offset=None, **_):
            calls.append((row_offset, kv_write))
            outs = (
                jnp.zeros((out_size, 128), jnp.uint32),
                jnp.zeros((out_size // native_sc_gather.ROPE_PACKING, 128), jnp.uint32),
            )
            return outs + ((nope, rope) if kv_write is not None else ())

        nb, ps = self.NB, self.PS
        rows = (None, 0, 1, 2)
        attn = sparse_mla.sparse_ragged_paged_attention
        attn = getattr(attn, "__wrapped__", attn)

        def fn(q, q_pe, n1, r1, n_l, r_l, topk, table, cu, dist, write):
            outs = []
            for row in rows:
                stack = None if row is None else (row, nb)
                pages = table
                if rewrite_table and stack is not None:
                    pages = stack_page_table_dcp(table, stack)
                outs.append(
                    attn(
                        q,
                        n1 if stack is None else n_l,
                        r1 if stack is None else r_l,
                        topk,
                        pages,
                        cu,
                        dist,
                        q_pe=q_pe,
                        return_lse=True,
                        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
                        is_dcp_mode=True,
                        topk_cap=self.CAP,
                        small_chunk_batch_size=4,
                        small_chunk_tokens=32,
                        lse_mask_empty=True,
                        lse_lane_dense=True,
                        gather_and_attention_chunk_size=512 if num_tokens > 32 else 256,
                        kv_write=write if kv_write else None,
                        kv_stack=stack,
                        **cfg,
                    )
                )
            return outs

        s = jax.ShapeDtypeStruct
        args = (
            s((num_tokens, 64, 512), jnp.bfloat16),
            s((num_tokens, 64, 64), jnp.bfloat16),
            s((nb, ps, 128), jnp.uint32),
            s((nb, ps // 4, 128), jnp.uint32),
            s((self.L * nb, ps, 128), jnp.uint32),
            s((self.L * nb, ps // 4, 128), jnp.uint32),
            s((num_tokens, self.TOPK), jnp.int32),
            s((8 * 4,), jnp.int32),
            s((9,), jnp.int32),
            s((3,), jnp.int32),
            s((3 * 16, 128), jnp.uint32),
        )
        with (
            mock.patch.object(native_sc_gather, "dsa_gather_native_sc", stub),
            mock.patch.object(pltpu, "get_tpu_info", lambda: None),
        ):
            closed = jax.make_jaxpr(fn)(*args)
        preps = [
            e
            for e in closed.jaxpr.eqns
            if e.primitive.name == "pallas_call"
            and str(e.params.get("name", "")).startswith("capped_window")
        ]
        return closed, preps, calls

    @parameterized.named_parameters(
        ("plain_window", 1024, {}, False, "capped_window_indices"),
        (
            "live_group",
            1024,
            {"mixed_live_gather": True, "mixed_live_widths": (256, 384)},
            False,
            "capped_window_indices_live",
        ),
        (
            "mixed_fused_prep",
            1024,
            {
                "mixed_live_gather": True,
                "mixed_fused_prep": True,
                "mixed_live_widths": (256, 384),
            },
            False,
            "capped_window_indices_live_fused",
        ),
        (
            "fused_decode_kv_write",
            16,
            {"fused_window_prep": True},
            True,
            "capped_window_indices_fused",
        ),
    )
    def test_prep_is_per_layer_and_offset_rides_the_gather(
        self, num_tokens, cfg, kv_write, prep_name
    ):
        closed, preps, calls = self._trace(num_tokens, cfg, kv_write)
        self.assertLen(preps, 4)  # one prep per call: the path under test ran
        self.assertEqual([e.params["name"] for e in preps], [prep_name] * 4)
        self.assertLen(set(_jaxpr_cone_hashes(closed, preps)), 1)
        per_call = len(calls) // 4
        layer_rows = self.NB * self.PS
        want = (
            [None] * 2 * per_call
            + [layer_rows] * per_call
            + [2 * layer_rows] * per_call
        )
        self.assertEqual([c[0] for c in calls], want)
        self.assertTrue(all((c[1] is not None) == kv_write for c in calls))

        # Positive control: the stack-wide page table makes every row's prep
        # its own computation (what kept XLA from CSE-ing them).
        closed, preps, _ = self._trace(num_tokens, cfg, kv_write, rewrite_table=True)
        self.assertLen(set(_jaxpr_cone_hashes(closed, preps)), 4)


class StackedWindowRowsTest(parameterized.TestCase):
    """`_window_physical_rows` (the translation every live / fused window prep
    runs) on the per-layer table and geometry, plus the stack row offset, is
    the row the stack-wide table translates to on the merged pool for every
    valid slot; padding slots stay inside this layer's row of the stack (the
    per-layer padding rows shifted). A wrong row offset (positive control)
    misses."""

    def test_offset_rows(self):
        from vllm_torchtpu.kernels.mla.sparse.kernel import _window_physical_rows

        n_layers, num_blocks, page_size, pps, chunk = 5, 12, 32, 128, 128
        rng = np.random.default_rng(11)
        table = rng.integers(0, num_blocks, (40, pps)).astype(np.int32)
        idx = rng.integers(0, pps * page_size, (40, 256)).astype(np.int32)
        idx[rng.random(idx.shape) < 0.3] = -1
        flat = np.arange(idx.size, dtype=np.int32).reshape(idx.shape)
        assert pps % chunk == 0
        kw = dict(page_size=page_size, pages_per_seq=pps)
        base = np.asarray(
            _window_physical_rows(
                jnp.asarray(idx),
                jnp.asarray(table),
                jnp.asarray(flat),
                total_num_pages=num_blocks,
                **kw,
            )
        )
        pad = idx == -1
        for row in range(n_layers):
            off = row * num_blocks * page_size
            want = np.asarray(
                _window_physical_rows(
                    jnp.asarray(idx),
                    stack_page_table_dcp(jnp.asarray(table), (row, num_blocks)),
                    jnp.asarray(flat),
                    total_num_pages=n_layers * num_blocks,
                    **kw,
                )
            )
            got = base + off
            np.testing.assert_array_equal(got[~pad], want[~pad])
            self.assertTrue(((got >= off) & (got < off + num_blocks * page_size)).all())
            wrong = base + ((row + 1) % n_layers) * num_blocks * page_size
            self.assertFalse(np.array_equal(wrong[~pad], want[~pad]))
