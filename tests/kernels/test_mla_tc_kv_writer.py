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
"""Correctness tests for the TensorCore sparse-MLA KV cache writer.

`tc_scatter.scatter_caches` must leave both whole DCP shards byte-identical to
`update_sparse_mla_kv_cache_dcp_jax`, on every rank. The caches start as
random words, not zeros, so a stray write, a dropped write and a rope
read-modify-write that loses its neighbours all show up.

The kernel tests run on TPU, or on CPU in the Pallas TPU interpreter. The
dispatch tests go through `kv_cache_utils` and its SparseCore fallback, so
they need a TPU.
"""

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.mla import kv_cache_utils, tc_scatter
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)

LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
KV_PACKING = 4
DCP_SIZE = 16
INTERLEAVE = 4
NUM_PAGES = 64  # per rank
PAGES_PER_SEQ = 3  # virtual pages of PAGE_SIZE * DCP_SIZE positions
VIRTUAL_PAGE = PAGE_SIZE * DCP_SIZE
CYCLE = DCP_SIZE * INTERLEAVE
FLAG = "TPU_MLA_TC_KV_WRITER"


def _rank_chunk(rank, cycle_index):
    """The INTERLEAVE positions `rank` owns in the given interleave cycle."""
    return cycle_index * CYCLE + rank * INTERLEAVE


# name -> ([(first new position, new tokens)] per sequence, bucket tokens).
# Positions count from 0; each sequence's `seq_len` is first + new tokens.
_CELLS = {
    # Random decode positions, one token per sequence.
    "decode_t16": (
        [
            (int(p), 1)
            for p in np.random.default_rng(7).integers(
                0, PAGES_PER_SEQ * VIRTUAL_PAGE, 16
            )
        ],
        16,
    ),
    # Five real tokens in a 16-token bucket; the tail is garbage to drop.
    "padded_bucket": (
        [
            (int(p), 1)
            for p in np.random.default_rng(8).integers(
                0, PAGES_PER_SEQ * VIRTUAL_PAGE, 5
            )
        ],
        16,
    ),
    # MTP-sized steps that start mid-chunk, so each one straddles two ranks
    # and lands two tokens or more in one rope row.
    "straddle_chunk_t32": (
        [
            (_rank_chunk(r, c) + off, 4)
            for r, c, off in zip(
                (0, 3, 7, 15, 9, 11, 2, 14),
                (0, 5, 17, 63, 70, 100, 129, 191),
                (1, 2, 3, 1, 2, 3, 2, 1),
            )
        ],
        32,
    ),
    # Physical-page and virtual-page boundaries (local slot 255 -> 0 of the
    # next block-table entry), a sequence's first token, and the largest step
    # the TC writer takes.
    "page_boundary_tmax": (
        [(VIRTUAL_PAGE - 4, 8), (2 * VIRTUAL_PAGE - 4, 8), (0, 8), (2000, 8)],
        tc_scatter.MAX_TOKENS,
    ),
    # Every token on rank 5 (whole rope rows), so the other ranks own none.
    "all_owned_rank5": (
        [(_rank_chunk(5, c), 4) for c in (0, 9, 64, 150)],
        16,
    ),
    # Nothing real in the bucket at all.
    "all_padding": ([(100, 0)], 16),
}
_CONTROL_CELLS = ("decode_t16", "straddle_chunk_t32")


def _specs(layout=KVCacheLayout.SPARSECORE):
    return (
        SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE, layout, NUM_PAGES, PAGE_SIZE, LKV_DIM, KV_PACKING
        ),
        SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE, layout, NUM_PAGES, PAGE_SIZE, ROPE_DIM, KV_PACKING
        ),
    )


def _inputs(seqs, num_tokens, seed=0, wrap_ordinals=True):
    """Addressing, rows and random-word caches for one cell."""
    rng = np.random.default_rng(seed)
    q_lens = np.asarray([n for _, n in seqs], np.int32)
    seq_lens = np.asarray([p + n for p, n in seqs], np.int32)
    seq_lens = np.maximum(seq_lens, 1)
    query_start_loc = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    num_valid = int(query_start_loc[-1])
    assert num_valid <= num_tokens

    # Virtual page ordinals, distinct mod NUM_PAGES and shuffled; half of them
    # past NUM_PAGES, which the DCP writer must resolve into the local shard.
    ordinals = rng.permutation(NUM_PAGES)[: len(seqs) * PAGES_PER_SEQ]
    if wrap_ordinals:
        ordinals = ordinals + NUM_PAGES * rng.integers(0, 2, ordinals.shape)
    block_tables = ordinals.astype(np.int32)

    # Raw bytes, not sampled floats, generated at the bucket size: the writer
    # is byte transport, and a float8 concatenate on TPU canonicalizes NaN.
    kv_c_u8 = rng.integers(0, 256, (num_tokens, LKV_DIM), np.uint8)
    k_pe_u8 = rng.integers(0, 256, (num_tokens, ROPE_DIM), np.uint8)
    nope_spec, rope_spec = _specs()
    caches = tuple(
        jnp.asarray(rng.integers(0, 2**32, s.shape, np.uint32), s.jax_dtype)
        for s in (nope_spec, rope_spec)
    )
    return dict(
        caches=caches,
        kv_c_normed=jax.lax.bitcast_convert_type(
            jnp.asarray(kv_c_u8), jnp.float8_e4m3fn
        ),
        k_pe=jax.lax.bitcast_convert_type(jnp.asarray(k_pe_u8), jnp.float8_e4m3fn),
        seq_lens=jnp.asarray(seq_lens),
        block_tables=jnp.asarray(block_tables),
        query_start_loc=jnp.asarray(query_start_loc),
        num_valid=num_valid,
    )


def _args(inp):
    return (
        *inp["caches"],
        inp["kv_c_normed"],
        inp["k_pe"],
        inp["seq_lens"],
        inp["block_tables"],
        inp["query_start_loc"],
    )


def _golden(inp, rank):
    nope_spec, rope_spec = _specs()
    return kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax(
        *_args(inp),
        jnp.int32(rank),
        nope_spec=nope_spec,
        rope_spec=rope_spec,
        dcp_size=DCP_SIZE,
        interleave_size=INTERLEAVE,
    )


def _dst_rows(inp, rank):
    """This rank's destination rows, as `_scatter_caches` derives them."""
    nope_spec, _ = _specs()
    page, slot = kv_cache_utils._get_dcp_page_and_slot(
        inp["kv_c_normed"].shape[0],
        inp["seq_lens"],
        inp["block_tables"],
        inp["query_start_loc"],
        jnp.int32(rank),
        nope_spec=nope_spec,
        dcp_size=DCP_SIZE,
        interleave_size=INTERLEAVE,
    )
    return kv_cache_utils._dst_rows(page, slot, PAGE_SIZE)


_tc_scatter_caches = jax.jit(
    tc_scatter.scatter_caches,
    static_argnames=("nope_spec", "rope_spec", "interpret", "_fault"),
)


def _tc_write(inp, dst_rows, interpret, fault=None):
    nope_spec, rope_spec = _specs()
    return _tc_scatter_caches(
        *inp["caches"],
        inp["kv_c_normed"],
        inp["k_pe"],
        dst_rows,
        nope_spec=nope_spec,
        rope_spec=rope_spec,
        interpret=interpret,
        _fault=fault,
    )


def _differs(got, want):
    return any(
        not np.array_equal(np.asarray(g), np.asarray(w)) for g, w in zip(got, want)
    )


class TcKvWriterTest(parameterized.TestCase):
    """The kernel alone, against the golden, every rank of a DCP16/il4 group."""

    def setUp(self):
        super().setUp()
        # The interpreter runs the same kernel body on CPU, DMAs and all.
        self.interpret = jax.devices()[0].platform != "tpu"

    @parameterized.named_parameters(*((name, name) for name in _CELLS))
    def test_matches_the_golden_on_every_rank(self, cell):
        seqs, num_tokens = _CELLS[cell]
        inp = _inputs(seqs, num_tokens)
        owned = 0
        for rank in range(DCP_SIZE):
            dst_rows = _dst_rows(inp, rank)
            owned += int(jnp.sum(dst_rows >= 0))
            got = _tc_write(inp, dst_rows, self.interpret)
            want = _golden(inp, rank)
            for name, g, w in zip(("nope", "rope"), got, want):
                np.testing.assert_array_equal(
                    np.asarray(g),
                    np.asarray(w),
                    err_msg=f"{cell} rank {rank}: {name} cache differs from golden",
                )
        # Ownership is a partition: every real token on exactly one rank.
        self.assertEqual(owned, inp["num_valid"])

    @parameterized.named_parameters(
        dict(testcase_name=f"{fault}_{cell}", fault=fault, cell=cell)
        for fault in ("wrong_row", "skip_rope_read", "shift_lane_group")
        for cell in _CONTROL_CELLS
    )
    def test_positive_control_fails(self, fault, cell):
        """Each fault must turn every rank that owns a token red: the
        comparison above is only evidence if it can fail."""
        seqs, num_tokens = _CELLS[cell]
        inp = _inputs(seqs, num_tokens)
        owning, caught = 0, 0
        for rank in range(DCP_SIZE):
            dst_rows = _dst_rows(inp, rank)
            if not bool(jnp.any(dst_rows >= 0)):
                continue
            owning += 1
            kernel_fault = None
            if fault == "wrong_row":
                dst_rows = jnp.where(dst_rows >= 0, dst_rows + 1, dst_rows)
            else:
                kernel_fault = fault
            got = _tc_write(inp, dst_rows, self.interpret, kernel_fault)
            caught += _differs(got, _golden(inp, rank))
        self.assertGreater(owning, 1)
        self.assertEqual(caught, owning, f"{fault} missed on {owning - caught} ranks")


class TcKvWriterDispatchTest(parameterized.TestCase):
    """`TPU_MLA_TC_KV_WRITER` routing inside `kv_cache_utils`."""

    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform != "tpu":
            self.skipTest("the SparseCore fallback requires a TPU.")

    def _dcp(self, inp, rank):
        nope_spec, rope_spec = _specs()
        return kv_cache_utils.update_sparse_mla_kv_cache_dcp(
            *_args(inp),
            jnp.int32(rank),
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=DCP_SIZE,
            interleave_size=INTERLEAVE,
        )

    def _run(self, inp, flag):
        env = {FLAG: "1"} if flag else {}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(
                tc_scatter, "scatter_caches", wraps=tc_scatter.scatter_caches
            ) as spy,
        ):
            if not flag:
                os.environ.pop(FLAG, None)
            out = [self._dcp(inp, rank) for rank in range(DCP_SIZE)]
        return out, spy.call_count

    @parameterized.named_parameters(
        ("decode_t16", "decode_t16"), ("page_boundary_tmax", "page_boundary_tmax")
    )
    def test_flag_on_takes_the_tc_writer(self, cell):
        seqs, num_tokens = _CELLS[cell]
        inp = _inputs(seqs, num_tokens)
        out, calls = self._run(inp, flag=True)
        self.assertEqual(calls, DCP_SIZE)
        for rank, got in enumerate(out):
            self.assertFalse(_differs(got, _golden(inp, rank)), f"rank {rank}")

    def test_flag_off_keeps_the_sparsecore_writer(self):
        seqs, num_tokens = _CELLS["decode_t16"]
        inp = _inputs(seqs, num_tokens)
        out, calls = self._run(inp, flag=False)
        self.assertEqual(calls, 0)
        for rank, got in enumerate(out):
            self.assertFalse(_differs(got, _golden(inp, rank)), f"rank {rank}")

    def test_above_max_tokens_falls_back_to_the_sparsecore_writer(self):
        """A prefill-sized step keeps today's path, flag or no flag."""
        num_tokens = 2 * tc_scatter.MAX_TOKENS
        seqs = [(_rank_chunk(r, 3 * r) + 1, 8) for r in range(8)]
        inp = _inputs(seqs, num_tokens)
        on, calls = self._run(inp, flag=True)
        self.assertEqual(calls, 0)
        off, _ = self._run(inp, flag=False)
        for rank in range(DCP_SIZE):
            self.assertFalse(_differs(on[rank], off[rank]), f"rank {rank}")
            self.assertFalse(_differs(on[rank], _golden(inp, rank)), f"rank {rank}")

    def test_unsharded_writer_with_the_flag_matches_the_jnp_writer(self):
        """`update_sparse_mla_kv_cache` shares `_scatter_caches`, so the flag
        reaches it too."""
        # Unsharded pages: keep every position inside the block table, and
        # every ordinal a real page.
        seqs = [
            (p % (PAGES_PER_SEQ * PAGE_SIZE), n) for p, n in _CELLS["decode_t16"][0]
        ]
        inp = _inputs(seqs, 16, wrap_ordinals=False)
        nope_spec, rope_spec = _specs()
        with mock.patch.dict(os.environ, {FLAG: "1"}):
            got = kv_cache_utils.update_sparse_mla_kv_cache(
                *_args(inp), nope_spec=nope_spec, rope_spec=rope_spec
            )
        want = kv_cache_utils.update_sparse_mla_kv_cache_jax(
            *_args(inp), nope_spec=nope_spec, rope_spec=rope_spec
        )
        self.assertFalse(_differs(got, want))


class TcKvWriterContractTest(parameterized.TestCase):
    """Host-side dispatch rules; no device work, so these run anywhere."""

    def test_supports_decode_steps_into_sparsecore_caches(self):
        nope_spec, rope_spec = _specs()
        self.assertTrue(tc_scatter.supports(nope_spec, rope_spec, 1))
        self.assertTrue(
            tc_scatter.supports(nope_spec, rope_spec, tc_scatter.MAX_TOKENS)
        )
        self.assertFalse(tc_scatter.supports(nope_spec, rope_spec, 0))
        self.assertFalse(
            tc_scatter.supports(nope_spec, rope_spec, tc_scatter.MAX_TOKENS + 1)
        )

    @parameterized.named_parameters(
        ("nope_tc", KVCacheLayout.TENSORCORE, KVCacheLayout.SPARSECORE),
        ("rope_tc", KVCacheLayout.SPARSECORE, KVCacheLayout.TENSORCORE),
        ("both_tc", KVCacheLayout.TENSORCORE, KVCacheLayout.TENSORCORE),
    )
    def test_tensorcore_layouts_are_not_supported(self, nope_layout, rope_layout):
        nope_spec, _ = _specs(nope_layout)
        _, rope_spec = _specs(rope_layout)
        self.assertFalse(tc_scatter.supports(nope_spec, rope_spec, 16))

    def test_flag_defaults_off(self):
        from vllm_torchtpu import envs

        with mock.patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            self.assertFalse(envs.TPU_MLA_TC_KV_WRITER)
