# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness for the all-gather PCP attention port.

Adapted from tpu-inference's tests/layers/common/test_pcp_attention_interface.py
to drive `sharded_pcp_allgather_ragged_paged_attention` on this repo's pcp-only
mesh. The global KV cache here is sharded on the BLOCK dim (rank r owns blocks
[r*npages, (r+1)*npages)), rather than on the page_size dim as in tpu-inference;
the per-rank tensor the kernel sees is the same either way.

Cache striping for THIS path is round-robin per token (interleave=1): global
token g lives on rank g % pcp at local slot g // pcp. (The streaming PCP path
uses interleave == page_size instead -- the two are not cache-compatible.)
"""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import Mesh

from vllm_torchtpu.layers.common.attention_interface import (
    PCP_AXIS_NAME, sharded_pcp_allgather_ragged_paged_attention)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    merge_kv, ref_ragged_paged_attention)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)

PAGE = 16          # per-rank block_size
MAX_SEQ = 8
NQ, NKV, HD = 8, 2, 128
DTYPE = jnp.float32
SM_SCALE = HD**-0.5


def _row_perm(pcp):
    """Rank r owns chunk r (head) and chunk 2P-1-r (tail)."""
    return [c for r in range(pcp) for c in (r, 2 * pcp - 1 - r)]


def _inv_row(pcp):
    inv = np.empty(2 * pcp, np.int64)
    inv[_row_perm(pcp)] = np.arange(2 * pcp)
    return inv


def _to_rank_order(x, pcp, C):
    x = np.asarray(x)
    return jnp.asarray(
        x.reshape(2 * pcp, C, *x.shape[1:])[_row_perm(pcp)].reshape(x.shape))


def _pcp_meta(pcp, C, num_current):
    """Per-rank fused current-phase metadata: cu = [0, C, C+tail_real],
    q_pos_offsets = [head_off, tail_off]."""
    two_p = 2 * pcp
    cu = np.zeros((pcp, MAX_SEQ + 1), np.int32)
    qpos = np.zeros((pcp, MAX_SEQ), np.int32)
    for r in range(pcp):
        tail_off = (two_p - 1 - r) * C
        tail_real = int(np.clip(num_current - tail_off, 0, C))
        cu[r, 1] = C
        cu[r, 2:] = C + tail_real
        qpos[r, 0] = r * C
        qpos[r, 1] = tail_off
    return jnp.asarray(cu), jnp.asarray(qpos)


class PcpAllGatherTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

    @property
    def _kvp(self):
        return get_dtype_packing(DTYPE)

    @property
    def _nkv2(self):
        return align_to(2 * NKV, self._kvp)

    def _cache_dims(self):
        return (self._nkv2 // self._kvp, self._kvp, align_to(HD, 128))

    def _rand(self, rng, shape):
        return jnp.array(rng.random(size=shape,
                                    dtype=np.float32)).astype(DTYPE)

    def _ref_cache(self, k, v, ntok, npages):
        kv = merge_kv(k, v)
        pad = cdiv(ntok, PAGE) * PAGE - ntok
        kv = jnp.pad(kv, ((0, pad), (0, 0), (0, 0), (0, 0)),
                     constant_values=jnp.nan)
        kv = kv.reshape(-1, PAGE, *self._cache_dims())
        cache = jnp.full((npages, PAGE, *self._cache_dims()), jnp.nan, DTYPE)
        return cache.at[:kv.shape[0]].set(kv)

    def _strided_cache(self, k, v, ntok, pcp, npages):
        """GLOBAL pcp cache, sharded on the BLOCK dim: rank r owns blocks
        [r*npages, (r+1)*npages). Rank r holds its g % pcp round-robin share
        at local slot g // pcp."""
        dims = self._cache_dims()
        shards = []
        for r in range(pcp):
            idx = np.arange(r, ntok, pcp)
            kv = np.asarray(merge_kv(k[idx], v[idx])) if len(idx) else None
            shard = np.full((npages, PAGE, *dims), np.nan, np.float32)
            if kv is not None:
                n = kv.shape[0]
                kv = np.pad(kv, ((0, cdiv(n, PAGE) * PAGE - n), (0, 0), (0, 0),
                                 (0, 0)),
                            constant_values=np.nan)
                kv = kv.reshape(-1, PAGE, *dims)
                shard[:kv.shape[0]] = kv
            shards.append(shard)
        return jnp.asarray(np.concatenate(shards, axis=0), DTYPE)

    def _page_indices(self, pps):
        """Both fused seqs are the SAME request; the writing seq is the tail,
        so seq 1 must carry a COPY of the request's pages."""
        pi = np.arange(pps, dtype=np.int32)
        out = np.zeros(MAX_SEQ * pps, np.int32)
        out[:pps] = pi
        out[pps:2 * pps] = pi
        return jnp.asarray(out)

    def _mesh(self, pcp):
        return Mesh(np.array(jax.devices()[:pcp]).reshape(pcp),
                    (PCP_AXIS_NAME, ))

    def _run(self, pcp, L, num_current, padded_s):
        C = padded_s // (2 * pcp)
        kv_total = L + num_current
        rng = np.random.default_rng(4)

        k_prev = self._rand(rng, (L, NKV, HD))
        v_prev = self._rand(rng, (L, NKV, HD))
        q_cur = self._rand(rng, (num_current, NQ, HD))
        k_cur = self._rand(rng, (num_current, NKV, HD))
        v_cur = self._rand(rng, (num_current, NKV, HD))

        # reference: plain full-causal prefill over the whole context
        ref_pps = cdiv(kv_total, PAGE)
        ref_pi = jnp.pad(jnp.arange(ref_pps, dtype=jnp.int32),
                         (0, MAX_SEQ * ref_pps - ref_pps))
        exp, _ = ref_ragged_paged_attention(
            q_cur, k_cur, v_cur,
            self._ref_cache(k_prev, v_prev, L, ref_pps),
            jnp.pad(jnp.array([kv_total], jnp.int32), (0, MAX_SEQ - 1)),
            ref_pi,
            jnp.pad(jnp.array([0, num_current], jnp.int32), (0, MAX_SEQ - 1)),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=SM_SCALE)
        exp = np.asarray(exp[:num_current])

        def pad_and_rank_order(x, width):
            buf = np.zeros((padded_s, width, HD), np.float32)
            buf[:num_current] = np.asarray(x, np.float32)
            return _to_rank_order(jnp.asarray(buf, DTYPE), pcp, C)

        q = pad_and_rank_order(q_cur, NQ)
        k = pad_and_rank_order(k_cur, NKV)
        v = pad_and_rank_order(v_cur, NKV)

        pps = cdiv(cdiv(kv_total, pcp), PAGE)
        npages = max(pps, 1)
        cache = self._strided_cache(k_prev, v_prev, L, pcp, npages)

        pad1 = lambda xs: jnp.pad(jnp.array(xs, jnp.int32),
                                  (0, MAX_SEQ - len(xs)))
        kv_lens = pad1([kv_total, kv_total])
        kv_cache_lens = pad1([L, L])
        cu_q_lens, q_pos_offsets = _pcp_meta(pcp, C, num_current)
        distribution = jnp.array([0, 0, 2], jnp.int32)

        out, new_cache = sharded_pcp_allgather_ragged_paged_attention(
            self._mesh(pcp), q, k, v, cache, kv_lens,
            self._page_indices(pps), cu_q_lens, distribution, kv_cache_lens,
            q_pos_offsets,
            sm_scale=SM_SCALE, update_kv_cache=True, use_causal_mask=True)
        return np.asarray(out), np.asarray(new_cache), exp, C, npages

    def _assert_matches(self, out, exp, pcp, C, num_current):
        inv = _inv_row(pcp)
        rows = np.array(
            [inv[g // C] * C + (g % C) for g in range(num_current)])
        got = out[rows]
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertGreater(float(np.abs(got).max()), 0.0)
        self.assertAllClose(got, exp, atol=2e-2, rtol=2e-2)

    @parameterized.product(pcp=[2, 4])
    def test_chunked_prefill(self, pcp):
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, S = 128, 128
        out, _, exp, C, _ = self._run(pcp, L, S, S)
        self._assert_matches(out, exp, pcp, C, S)

    @parameterized.product(pcp=[2, 4])
    def test_partial_tail(self, pcp):
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, S, padded_s = 128, 100, 128
        out, _, exp, C, _ = self._run(pcp, L, S, padded_s)
        self._assert_matches(out, exp, pcp, C, S)

    @parameterized.product(pcp=[2, 4])
    def test_first_chunk_no_cache(self, pcp):
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        S = 128
        out, _, exp, C, _ = self._run(pcp, 0, S, S)
        self._assert_matches(out, exp, pcp, C, S)

    @parameterized.product(pcp=[2, 4])
    def test_kv_cache_write(self, pcp):
        """Current KV must land strided: global token g at rank g % pcp, local
        slot g // pcp -> global block r*npages + (local // PAGE), col local % PAGE."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, S = 128, 128
        rng = np.random.default_rng(4)  # same seed as _run
        _ = self._rand(rng, (L, NKV, HD)), self._rand(rng, (L, NKV, HD))
        _ = self._rand(rng, (S, NQ, HD))
        k_cur, v_cur = self._rand(rng, (S, NKV, HD)), self._rand(rng,
                                                                 (S, NKV, HD))
        ref = np.asarray(merge_kv(k_cur, v_cur))

        _, cache, _, _, npages = self._run(pcp, L, S, S)
        for i in range(S):
            g = L + i
            r, local = g % pcp, g // pcp
            page, off = local // PAGE, local % PAGE
            got = cache[r * npages + page, off]
            self.assertAllClose(got, ref[i], atol=2e-2, rtol=2e-2)


    # ---- the branch production actually takes -------------------------------
    # The wrapper picks between two current-phase KV layouts:
    #   remap_kv=True  (C >= PAGE and C % PAGE == 0): rank-ordered ag(k_l) with
    #                  the in-kernel token->rank offset remap.
    #   remap_kv=False (otherwise): to_token_order(k_l), pcp_chunk_size=None.
    # Every test above has C = 32 or 16 against PAGE = 16, so all of them take
    # the True branch. The server buckets the PER-RANK token count against a
    # 256-token block size, so a ~1000-token prompt at pcp=8 gives C = 64 and
    # takes the False branch -- previously with no coverage at all.

    @parameterized.product(pcp=[2, 4])
    def test_chunked_prefill_chunk_below_page(self, pcp):
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, S = 128, 32
        out, _, exp, C, _ = self._run(pcp, L, S, S)
        self.assertLess(C, PAGE)  # to_token_order branch
        self._assert_matches(out, exp, pcp, C, S)

    @parameterized.product(pcp=[2, 4])
    def test_first_chunk_no_cache_chunk_below_page(self, pcp):
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        S = 32
        out, _, exp, C, _ = self._run(pcp, 0, S, S)
        self.assertLess(C, PAGE)
        self._assert_matches(out, exp, pcp, C, S)

    @parameterized.product(pcp=[2, 4])
    def test_kv_cache_write_chunk_below_page(self, pcp):
        """Same striding contract as test_kv_cache_write, on the branch the
        server uses. If the current phase feeds token-ordered KV but the cache
        write still assumes rank order, this is where it shows up."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, S = 128, 32
        rng = np.random.default_rng(4)  # same seed and draw order as _run
        _ = self._rand(rng, (L, NKV, HD)), self._rand(rng, (L, NKV, HD))
        _ = self._rand(rng, (S, NQ, HD))
        k_cur, v_cur = self._rand(rng, (S, NKV, HD)), self._rand(rng,
                                                                 (S, NKV, HD))
        ref = np.asarray(merge_kv(k_cur, v_cur))

        _, cache, _, C, npages = self._run(pcp, L, S, S)
        self.assertLess(C, PAGE)
        for i in range(S):
            g = L + i
            r, local = g % pcp, g // pcp
            page, off = local // PAGE, local % PAGE
            got = cache[r * npages + page, off]
            self.assertAllClose(got, ref[i], atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
