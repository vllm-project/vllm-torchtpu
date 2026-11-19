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
"""TPU_DCP_MLA_MIXED_FUSED_PREP_SMALL and TPU_MLA_GATHER_EARLY_DMA_LIVE_GROUP.

The 64-token decode bucket is a live-group call of one 64-token chunk. Before
these flags its window-index preparation ran as separate XLA ops (kv_lens,
`live_group_table`, `prepare_capped_window_live`), because the fused prep
`prepare_capped_window_live_fused` required chunks of whole 128-token rows.
The claim of both flags is bit identity with that incumbent:

* kernel level: the fused prep at chunk 64 returns the same kv_lens, group
  table, gather bound and window rows as the three separate ops, on ragged
  steps with and without overflow past the 512-slot window; the injected
  faults "kv", "off" and "win1" each fail that comparison;
* attention level: `sparse_ragged_paged_attention` on a 64-token bucket gives
  the same output and lse bits with the fused prep as without it;
* DCP level: `sparse_mla_attention_dcp` with the KV write in the gather gives
  the same caches, partial outputs and lse on every rank with each flag, and a
  fused prep run with the "off" fault differs.
"""

import functools
import os
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

sys.path.insert(0, os.path.dirname(__file__))

import test_mla_decode_prologue as prologue  # noqa: E402
import test_mla_native_sc_attention as nsa  # noqa: E402

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla  # noqa: E402
from vllm_torchtpu.kernels.mla.sparse import native_sc_gather  # noqa: E402

pytestmark = pytest.mark.tpu_test

SMALL_FLAG = "TPU_DCP_MLA_MIXED_FUSED_PREP_SMALL"
EARLY_FLAG = "TPU_MLA_GATHER_EARLY_DMA_LIVE_GROUP"
_Base = nsa.SparseMlaCacheLayoutsTest


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("the fused prep and the SparseCore gather need a TPU")


def test_flags_default_off():
    from vllm_torchtpu import envs

    with mock.patch.dict(os.environ):
        os.environ.pop(SMALL_FLAG, None)
        os.environ.pop(EARLY_FLAG, None)
        assert not envs.TPU_DCP_MLA_MIXED_FUSED_PREP_SMALL
        assert not envs.TPU_MLA_GATHER_EARLY_DMA_LIVE_GROUP


def test_supported_takes_only_one_small_chunk():
    """Chunk 64 is accepted only with `small_chunk` and only as the whole
    call; 128-token chunks are accepted as before."""
    ok = sparse_mla.mixed_fused_prep_supported
    assert not ok(64, 2048, 64, 16, 512)
    assert ok(64, 2048, 64, 16, 512, small_chunk=True)
    assert not ok(128, 2048, 64, 16, 512, small_chunk=True)  # two chunks
    assert not ok(64, 2048, 64, 4, 512, small_chunk=True)  # live group refuses bz 4
    assert ok(256, 2048, 128, 16, 512)


class FusedPrepSmallTest(parameterized.TestCase):
    # The case builders and the bitwise check of the 128-token fused prep test
    # (`test_capped_window_live_fused_matches_live`); its chunk is
    # min(256, num_tokens), so 64 tokens give one 64-token chunk.
    _live_fused_case = _Base._live_fused_case
    _live_fused_check = _Base._live_fused_check
    _split_q_lse_case = _Base._split_q_lse_case

    @parameterized.product(real=(64, 49, 17), overflow=(False, True))
    def test_fused_prep_chunk64_matches_live(self, real, overflow):
        """kv_lens, group table, gather bound and every window the chunk's
        gather reads, bit for bit against the three separate ops."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("Mosaic kernel requires a TPU")
        self._live_fused_check(64, real, overflow)

    def test_fused_prep_chunk64_faults_caught(self):
        """Positive controls: each injected fault fails the bitwise check."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("Mosaic kernel requires a TPU")
        for fault in ("kv", "off", "win1"):
            with self.assertRaises(AssertionError, msg=fault):
                self._live_fused_check(64, 64, True, fault=fault)

    def test_attention_bucket64_bit_identical(self):
        """The 64-token bucket through `sparse_ragged_paged_attention` (raw
        top-k, in-kernel mask, live gather, widths, small-bucket body levers):
        the fused prep is taken and every real row's output and lse match."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        query_tokens, seq_q = 64, (4,) * 16
        rng = np.random.default_rng(20261119)
        problem, _, _, counts, page_table, cu_q_lens, distribution = (
            self._split_q_lse_case(query_tokens, seq_q, rng)
        )
        live = np.arange(2048)[None, :] < counts[:, None]
        topk_raw = jnp.where(jnp.asarray(live), problem["topk"], -1)
        q_nope = problem["q"][..., : nsa.NOPE_DIM]
        q_pe = problem["q"][..., nsa.NOPE_DIM :]

        def run(topk_in, **kw):
            out, lse = sparse_mla.sparse_ragged_paged_attention(
                q_nope,
                problem["sparsecore_nope"],
                problem["sparsecore_rope"],
                topk_in,
                page_table,
                cu_q_lens,
                distribution,
                sm_scale=0.125,
                k_scale=0.75,
                return_lse=True,
                cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
                gather_and_attention_chunk_size=512,
                topk_cap=512,
                small_chunk_batch_size=4,
                small_chunk_tokens=32,
                q_pe=q_pe,
                lse_lane_dense=True,
                lse_mask_empty=True,
                mixed_live_gather=True,
                mixed_live_widths=(256, 384),
                mixed_body_levers=sparse_mla.MIXED_BODY_DEFAULT,
                mixed_body_small_levers=sparse_mla.MIXED_BODY_SMALL_DEFAULT,
                **kw,
            )
            return out[..., : nsa.NOPE_DIM], lse

        def bits(x):
            return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint16))

        ref, ref_lse = jax.block_until_ready(run(topk_raw, mixed_fused_prep=True))
        with mock.patch.object(
            sparse_mla,
            "prepare_capped_window_live_fused",
            wraps=sparse_mla.prepare_capped_window_live_fused,
        ) as spy:
            out, lse = jax.block_until_ready(
                run(topk_raw, mixed_fused_prep=True, mixed_fused_prep_small=True)
            )
        self.assertEqual(spy.call_count, 1)
        real = int(sum(seq_q))
        np.testing.assert_array_equal(bits(out)[:real], bits(ref)[:real])
        np.testing.assert_array_equal(
            np.asarray(lse)[:real].view(np.uint32),
            np.asarray(ref_lse)[:real].view(np.uint32),
        )
        # Positive control: one moved live entry of the longest row moves it.
        row = int(np.argmax(counts[:real]))
        moved = topk_raw.at[row, 0].set((topk_raw[row, 0] + 1) % 8192)
        bad, _ = jax.block_until_ready(
            run(moved, mixed_fused_prep=True, mixed_fused_prep_small=True)
        )
        self.assertFalse(np.array_equal(bits(bad)[row], bits(ref)[row]))


# --- DCP: sparse_mla_attention_dcp with the KV write in the 64-token gather ---


def _dcp_size():
    return min(8, len(jax.devices()))


def _dcp(dcp, extra_env, prep_fault=None):
    """One `prologue._dcp_run` with the bucket-64 step, the KV write in the
    gather at 64 tokens, the fused prep enabled for 128-token chunks, and
    `extra_env`. Returns (results, fused prep calls, gather calls with
    early_dma)."""
    env = dict(
        prologue.PROD_ENV,
        TPU_MLA_DECODE_PROLOGUE="1",
        TPU_MLA_KV_WRITE_IN_GATHER="1",
        TPU_MLA_KV_WRITE_IN_GATHER_MAX_TOKENS="64",
        TPU_DCP_MLA_MIXED_FUSED_PREP="1",
        TPU_MLA_GATHER_EARLY_DMA="1",
        **extra_env,
    )
    prep = sparse_mla.prepare_capped_window_live_fused
    if prep_fault is not None:
        prep = functools.partial(prep, _fault=prep_fault)
    jax.clear_caches()  # the spies below count traces
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(
            sparse_mla, "prepare_capped_window_live_fused", wraps=prep
        ) as prep_spy,
        mock.patch.object(
            native_sc_gather,
            "dsa_gather_native_sc",
            wraps=native_sc_gather.dsa_gather_native_sc,
        ) as gather_spy,
    ):
        for k in (SMALL_FLAG, EARLY_FLAG):
            if k not in extra_env:
                os.environ.pop(k, None)
        res = prologue._dcp_run(dcp, candidate=True, bucket64=True)
        early = [
            c.kwargs["out_size"]
            for c in gather_spy.call_args_list
            if c.kwargs.get("early_dma")
        ]
    return res, prep_spy.call_count, early


def _assert_same(ref, got):
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope", "out", "lse"):
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")


def test_dcp_bucket64_fused_prep_and_early_dma_bit_identical():
    """Each flag on the 64-token step: the fused prep is traced once (the
    bucket-64 step) and, with the early-DMA flag, the 64-token gather starts
    its DMAs early; caches, partials and lse stay bit-identical. Fused prep
    with the "off" fault differs (positive control)."""
    _require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    out64 = 64 * 2048
    ref, ref_prep, ref_early = _dcp(dcp, {})
    assert ref_prep == 0, ref_prep
    assert out64 not in ref_early, ref_early
    got, got_prep, got_early = _dcp(dcp, {SMALL_FLAG: "1"})
    assert got_prep == 1, got_prep
    assert out64 not in got_early, got_early
    _assert_same(ref, got)
    got, got_prep, got_early = _dcp(dcp, {SMALL_FLAG: "1", EARLY_FLAG: "1"})
    assert got_prep == 1, got_prep
    assert out64 in got_early, got_early
    _assert_same(ref, got)
    got, _, got_early = _dcp(dcp, {EARLY_FLAG: "1"})
    assert out64 in got_early, got_early
    _assert_same(ref, got)
    bad, bad_prep, _ = _dcp(dcp, {SMALL_FLAG: "1"}, prep_fault="off")
    assert bad_prep == 1
    diff = [
        not np.array_equal(a[k], b[k])
        for a, b in zip(ref, bad)
        for k in ("out", "lse")
    ]
    assert any(diff), "fused prep fault 'off' not caught"
