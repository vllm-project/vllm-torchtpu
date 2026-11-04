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
"""Tests for the DCP sparse MLA path with one preparation kernel.

`dcp_sparse_mla_attention` runs `prepare_dcp_sparse_mla_inputs` (a TensorCore
Pallas kernel), then per chunk the SparseCore gather and the attention kernel.

* The preparation kernel and the attention kernel run on CPU in Pallas TPU
  interpret mode. The SparseCore gather cannot run there, so the CPU tests
  replace it with a NumPy emulation that reads the same index order and applies
  the same ROPE byte transform (`native_sc_gather.transpose_quad`); staging rows
  outside the live bound are filled with fp8 NaN bytes, so a kernel that reads
  past its rows fails the test.
* On a TPU the whole path is compared with `sparse_ragged_paged_attention`.

Every comparison has a positive control that must fail.
"""

import contextlib
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_gather

NOPE = 512
ROPE = 64
PAGE = 256
# fp8 e4m3fn NaN in every byte: what an unwritten staging row may hold.
NAN_WORD = np.uint32(0x7F7F7F7F)


def _on_tpu() -> bool:
    return jax.devices()[0].platform == "tpu"


def _interpret():
    """Interpret mode on CPU; a no-op context on TPU."""
    return contextlib.nullcontext() if _on_tpu() else pltpu.force_tpu_interpret_mode()


class _FakeSparseCore:
    num_cores = 2
    num_subcores = 16


class _FakeTpuInfo:
    sparse_core = _FakeSparseCore


def _emulated_gather(
    nope_cache,
    rope_cache,
    indices,
    *,
    out_size,
    atoms_per_batch=16,
    index_offset=0,
    live_vector=None,
    live_offset=0,
    **unused,
):
    """NumPy model of `dsa_gather_native_sc` with a precomputed live vector.

    Staging row f holds cache row `indices[index_offset + f]` for the first
    `live batches * atoms_per_batch * 64` rows and NaN bytes after them. ROPE
    row r holds the four tokens 4r..4r+3 after `transpose_quad`, followed by
    zero words, as the SparseCore kernel writes it.
    """
    del unused
    nope_rows = nope_cache.reshape(-1, 128)
    rope_words = rope_cache.size // nope_rows.shape[0]
    rope_rows = rope_cache.reshape(-1, rope_words)
    idx = indices.reshape(-1)[index_offset : index_offset + out_size]
    live_rows = live_vector[live_offset] * atoms_per_batch * 64
    row = jnp.arange(out_size)
    nope = jnp.where((row < live_rows)[:, None], nope_rows[idx], NAN_WORD)
    quads = rope_rows[idx].reshape(out_size // 4, 4, rope_words)
    bands = native_sc_gather.transpose_quad([quads[:, t] for t in range(4)])
    zeros = jnp.zeros((out_size // 4, 128 - 4 * rope_words), jnp.uint32)
    rope = jnp.concatenate([*bands, zeros], axis=-1)
    rope = jnp.where((row[: out_size // 4] * 4 < live_rows)[:, None], rope, NAN_WORD)
    return nope, rope


def _problem(rng, *, num_tokens, counts, num_heads=8, topk=512, num_pages=48, seqs=2):
    """Paged fp8 caches (SparseCore layout) and rank-local top-k rows.

    Token t belongs to sequence `t * seqs // num_tokens` (equal split); its
    top-k row holds `counts[t]` distinct local indices, then -1.
    """
    pages_per_seq = 8
    f8 = jnp.float8_e4m3fn
    rows = num_pages * PAGE
    nope_src = jnp.asarray(rng.normal(size=(rows, NOPE)), jnp.float32).astype(f8)
    rope_src = jnp.asarray(rng.normal(size=(rows, ROPE)), jnp.float32).astype(f8)
    nope_cache = kv_cache_utils.pack_tokens(nope_src, NOPE).reshape(
        num_pages, PAGE, 128
    )
    rope_cache = kv_cache_utils.pack_tokens(rope_src, ROPE).reshape(
        num_pages, PAGE // 8, 128
    )
    # Virtual page ordinals; `% num_pages` maps them to physical pages.
    block_tables = rng.permutation(num_pages * 3)[: seqs * pages_per_seq].astype(
        np.int32
    )
    seq_of = np.arange(num_tokens) * seqs // num_tokens
    cu = np.searchsorted(seq_of, np.arange(seqs + 1)).astype(np.int32)
    topk_idx = np.full((num_tokens, topk), -1, np.int32)
    for t in range(num_tokens):
        topk_idx[t, : counts[t]] = rng.choice(
            pages_per_seq * PAGE, counts[t], replace=False
        )
    q_nope = jnp.asarray(rng.normal(size=(num_tokens, num_heads, NOPE)), jnp.bfloat16)
    q_pe = jnp.asarray(rng.normal(size=(num_tokens, num_heads, ROPE)), jnp.bfloat16)
    return dict(
        q_nope=q_nope,
        q_pe=q_pe,
        nope_cache=nope_cache,
        rope_cache=rope_cache,
        topk=topk_idx,
        block_tables=block_tables,
        cu_q_lens=cu,
        distribution=np.array([0, seqs, seqs], np.int32),
        nope_src=np.asarray(nope_src.astype(jnp.float32), np.float64),
        rope_src=np.asarray(rope_src.astype(jnp.float32), np.float64),
        seq_of=seq_of,
        num_pages=num_pages,
    )


def _physical(p, topk_idx, block_tables=None):
    """Physical row per slot (-1 stays -1)."""
    bt = p["block_tables"] if block_tables is None else block_tables
    bt = bt.reshape(len(p["cu_q_lens"]) - 1, -1) % p["num_pages"]
    safe = np.maximum(topk_idx, 0)
    rows = bt[p["seq_of"][:, None], safe // PAGE] * PAGE + safe % PAGE
    return np.where(topk_idx >= 0, rows, -1)


def _reference(p, sm_scale, k_scale, block_tables=None):
    """float64 attention per token; out 0 and lse -inf for an empty row."""
    phys = _physical(p, p["topk"], block_tables)
    qn = np.asarray(p["q_nope"].astype(jnp.float32), np.float64)
    qp = np.asarray(p["q_pe"].astype(jnp.float32), np.float64)
    out = np.zeros(qn.shape)
    lse = np.full(qn.shape[:2], -np.inf)
    for t in range(qn.shape[0]):
        r = phys[t][phys[t] >= 0]
        if r.size == 0:
            continue
        kn, kr = p["nope_src"][r], p["rope_src"][r]
        s = (qn[t] @ kn.T + qp[t] @ kr.T) * sm_scale * k_scale
        m = s.max(-1, keepdims=True)
        e = np.exp(s - m)
        out[t] = e @ kn * k_scale / e.sum(-1, keepdims=True)
        lse[t] = (m + np.log(e.sum(-1, keepdims=True)))[:, 0]
    return out, lse


def _hash_rows(token, col, topk, num_pages):
    """The `-1` scatter of `_capped_window_indices_kernel`, int32 wraparound."""
    flat = (token * topk + col).astype(np.int32)
    with np.errstate(over="ignore"):
        page = (flat * np.int32(104729)) % np.int32(num_pages)
        offset = (flat * np.int32(15485863)) % np.int32(PAGE)
    return page * PAGE + offset


class DcpPrepKernelTest(parameterized.TestCase):
    @parameterized.parameters(
        # (tokens, chunk, cap, attention step, real tokens): a decode bucket,
        # two prefill chunks with overflow rows, and a ragged tail.
        (16, 16, 256, 4, 8),
        (256, 128, 256, 16, 256),
        (32, 32, 128, 1, 21),
    )
    def test_matches_reference(self, num_tokens, chunk, cap, step, real):
        rng = np.random.default_rng(num_tokens * 7 + cap)
        topk = 512
        counts = rng.integers(0, cap + 1, size=num_tokens)
        counts[:4] = [0, 1, cap, min(topk, 3 * cap + 1)]
        counts[real:] = 0
        p = _problem(rng, num_tokens=num_tokens, counts=counts, topk=topk)
        p["cu_q_lens"][-1] = real
        p["cu_q_lens"] = np.minimum(p["cu_q_lens"], real)

        def run(block_tables):
            with (
                _interpret(),
                mock.patch.object(pltpu, "get_tpu_info", lambda: _FakeTpuInfo),
            ):
                return jax.jit(
                    lambda tk, bt, cu, d: sparse_mla.prepare_dcp_sparse_mla_inputs(
                        tk,
                        bt,
                        cu,
                        d,
                        num_heads=8,
                        chunk_size=chunk,
                        topk_cap=cap,
                        page_size=PAGE,
                        total_num_pages=p["num_pages"],
                        attention_batch=step,
                        gather_batch_rows=64,
                        num_owners=32,
                    )
                )(p["topk"], block_tables, p["cu_q_lens"], p["distribution"])

        idx, kv_lens, lse, live = (np.asarray(x) for x in run(p["block_tables"]))
        np.testing.assert_array_equal(kv_lens.reshape(-1), counts)
        self.assertTrue(np.all(np.isneginf(lse)))

        windows = topk // cap
        phys = _physical(p, p["topk"])
        token = np.arange(num_tokens)[:, None]
        col = np.arange(topk)[None, :]
        want = np.where(phys >= 0, phys, _hash_rows(token, col, topk, p["num_pages"]))
        # [chunk c, window w, token t, slot j] -> tile order
        # [t // 8, j // 128, t % 8, j % 128].
        want = want.reshape(
            num_tokens // chunk, chunk // 8, 8, windows, cap // 128, 128
        )
        want = want.transpose(0, 3, 1, 4, 2, 5)
        got = idx.reshape(want.shape)
        live_tile = live.reshape(-1, 8, 128)
        for c in range(num_tokens // chunk):
            lt = int(np.clip(real - c * chunk, 0, chunk))
            lt = -(-lt // step) * step
            seg = counts[c * chunk : c * chunk + lt]
            lw = int(
                np.clip(
                    (np.maximum(seg, 1).max() + cap - 1) // cap if lt else 1, 1, windows
                )
            )
            np.testing.assert_array_equal(got[c, :lw], want[c, :lw])
            num_live = (lw - 1) * chunk * cap + (-(-lt // 8) * 8) * cap
            live_b = min(-(-num_live // 64), chunk * topk // 64)
            self.assertEqual(live_tile[c, 0, 0], live_b)
            self.assertEqual(live_tile[c, 0, 16], -(-live_b // 32))
        # Positive control: a shifted page table must change the rows.
        shifted = np.roll(p["block_tables"], 1)
        got_bad = np.asarray(run(shifted)[0]).reshape(want.shape)
        self.assertFalse(np.array_equal(got_bad[:, :1], want[:, :1]))


class DcpSparseMlaAttentionTest(parameterized.TestCase):
    @parameterized.parameters(
        dict(step=1),
        dict(step=4),
        dict(step=4, bias_mask=True, exact_division=True),
        dict(step=4, split_dots=True),
        dict(step=4, bias_mask=True, exact_division=True, split_dots=True),
    )
    def test_matches_float64_reference(self, step, **options):
        """Decode bucket with rows owning 0, 1, cap, cap + 1 and topk slots."""
        rng = np.random.default_rng(20261104 + step)
        num_tokens, topk, cap = 16, 512, 256
        counts = np.array([0, 1, 256, 257, 512, 7, 100, 200] + [0] * 8)
        p = _problem(rng, num_tokens=num_tokens, counts=counts, topk=topk, seqs=2)
        p["cu_q_lens"] = np.array([0, 4, 8], np.int32)
        p["seq_of"] = np.minimum(np.arange(num_tokens) // 4, 1)
        sm_scale, k_scale = 0.0625, 0.75

        def run(block_tables):
            gather = mock.patch.object(
                native_sc_gather, "dsa_gather_native_sc", _emulated_gather
            )
            info = mock.patch.object(pltpu, "get_tpu_info", lambda: _FakeTpuInfo)
            ctx = contextlib.nullcontext() if _on_tpu() else contextlib.ExitStack()
            with ctx as stack, _interpret():
                if stack is not None:
                    stack.enter_context(gather)
                    stack.enter_context(info)
                out, lse = sparse_mla.dcp_sparse_mla_attention(
                    p["q_nope"],
                    p["q_pe"],
                    p["nope_cache"],
                    p["rope_cache"],
                    jnp.asarray(p["topk"]),
                    jnp.asarray(block_tables),
                    jnp.asarray(p["cu_q_lens"]),
                    jnp.asarray(p["distribution"]),
                    sm_scale=sm_scale,
                    k_scale=k_scale,
                    topk_cap=cap,
                    chunk_size=256,
                    small_chunk_batch_size=step,
                    **options,
                )
            return np.asarray(out.astype(jnp.float32)), np.asarray(lse)

        ref_out, ref_lse = _reference(p, sm_scale, k_scale)
        out, lse = run(p["block_tables"])
        real = np.arange(num_tokens) < 8
        owned = real & (counts > 0)
        scale = np.sqrt(np.mean(ref_out[owned] ** 2))

        def error(o, lse_value):
            return (
                np.abs(o[owned] - ref_out[owned]).max() / scale,
                np.abs(lse_value[owned] - ref_lse[owned]).max(),
            )

        out_err, lse_err = error(out, lse)
        self.assertLess(out_err, 0.1)
        self.assertLess(lse_err, 1e-3)
        empty = real & (counts == 0)
        self.assertTrue(np.all(np.isneginf(lse[empty])))
        self.assertTrue(np.all(out[empty] == 0))
        self.assertTrue(np.all(np.isfinite(out[real])))
        # Positive control: a shifted page table must fail the same checks.
        bad_out, bad_lse = run(np.roll(p["block_tables"], 1))
        bad_out_err, bad_lse_err = error(bad_out, bad_lse)
        self.assertTrue(bad_out_err >= 0.1 or bad_lse_err >= 1e-3)

    def test_matches_sparse_ragged_paged_attention_on_tpu(self):
        """Same output and lse as the generic capped split-q path, real gather."""
        if not _on_tpu():
            self.skipTest("SparseCore gather requires a TPU")
        rng = np.random.default_rng(7)
        num_tokens, topk, cap = 512, 2048, 256
        counts = rng.integers(1, 200, size=num_tokens)
        counts[:6] = [1, 255, 256, 257, 700, 2048]
        p = _problem(
            rng, num_tokens=num_tokens, counts=counts, topk=topk, num_pages=64, seqs=2
        )
        common = dict(sm_scale=0.0625, k_scale=1.0)
        topk_idx = jnp.asarray(p["topk"])
        args = (
            jnp.asarray(p["block_tables"]),
            jnp.asarray(p["cu_q_lens"]),
            jnp.asarray(p["distribution"]),
        )
        ref_out, ref_lse = sparse_mla.sparse_ragged_paged_attention(
            p["q_nope"],
            p["nope_cache"],
            p["rope_cache"],
            topk_idx,
            args[0] % p["num_pages"],
            *args[1:],
            return_lse=True,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            topk_cap=cap,
            gather_and_attention_chunk_size=256,
            q_pe=p["q_pe"],
            lse_lane_dense=True,
            **common,
        )
        out, lse = sparse_mla.dcp_sparse_mla_attention(
            p["q_nope"],
            p["q_pe"],
            p["nope_cache"],
            p["rope_cache"],
            topk_idx,
            *args,
            topk_cap=cap,
            chunk_size=256,
            **common,
        )
        np.testing.assert_allclose(
            np.asarray(out, np.float32),
            np.asarray(ref_out, np.float32),
            atol=2e-2,
            rtol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(lse), np.asarray(ref_lse), atol=1e-4, rtol=1e-4
        )
        # Positive control: the comparison can fail.
        self.assertGreater(
            np.abs(
                np.asarray(out, np.float32) - np.asarray(p["q_nope"], np.float32)
            ).max(),
            2e-2,
        )


if __name__ == "__main__":
    absltest.main()
