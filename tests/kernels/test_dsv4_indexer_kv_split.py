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
"""streamindex_topk kv_split: N emulated ranks' shares merged == the unsplit
top-k (same indices, same order), at DSv4-Pro served shapes (SEQ_ALONG_LANE,
page 256 compressed tokens, 1024 pages per sequence, k=1024, 64 heads, ratio 4,
rc7 block-1024 block sizes). Also times one core: unsplit vs one share + merge.

    python3 tests/kernels/test_dsv4_indexer_kv_split.py [--time]
"""

import argparse
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    convert_cache_to_seq_along_lane,
    merge_kv_split_topk,
    streamindex_topk,
)

PAGE = 256  # compressed tokens per page at --block-size 1024, ratio 4
PAGES_PER_SEQ = 1024  # max-model-len 1M
D = 128
H = 64
K = 1024
RATIO = 4
MAX_SEQS = 4
KV_PAGES = (32, 64, 32)
Q_BLOCKS = (1, 32, 32)
VQ = 7  # 1 + 6 speculative tokens

pytestmark = [pytest.mark.tpu_test]


def _pack_pages(num_pages, seed):
    """Random keys -> fp8 + ue8m0 SAL records (valid encodings, no NaN)."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.normal(key, (num_pages, PAGE, D), jnp.float32)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.clip(jnp.max(jnp.abs(keys), -1, keepdims=True), 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q8 = (keys / scale).astype(jnp.float8_e4m3fn)
    sc8 = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
    rec = jnp.concatenate(
        [
            jax.lax.bitcast_convert_type(q8, jnp.uint8),
            jax.lax.bitcast_convert_type(sc8, jnp.uint8)[..., None],
        ],
        axis=-1,
    )
    width = 256
    rec = jnp.pad(rec, ((0, 0), (0, 0), (0, width - rec.shape[-1])))
    has = rec.reshape(num_pages, PAGE // 4, 4, width)
    return convert_cache_to_seq_along_lane(has, D)


def _case(seqs, T, seed=0):
    """seqs: list of (q_len, seq_len, kind) kind in {'verify','chunk'};
    verify windows first (static pass), then chunks (mixed pass)."""
    rng = np.random.default_rng(seed)
    n = len(seqs)
    q_lens = [s[0] for s in seqs]
    seq_lens = [s[1] for s in seqs]
    n_verify = sum(1 for s in seqs if s[2] == "verify")
    pages_needed = [-(-(sl // RATIO) // PAGE) for sl in seq_lens]
    total = sum(pages_needed) + 8
    perm = rng.permutation(total).astype(np.int32)
    table = np.zeros((MAX_SEQS, PAGES_PER_SEQ), np.int32)
    off = 0
    for i, p in enumerate(pages_needed):
        table[i, :p] = perm[off : off + p]
        off += p
    cache = _pack_pages(total, seed)
    q = jax.random.normal(jax.random.PRNGKey(seed + 1), (T, H, D), jnp.float32)
    q = q.astype(jnp.float8_e4m3fn)
    w = jnp.asarray(rng.uniform(-1.0, 1.0, (T, H)).astype(np.float32))
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1 : n + 1] = np.cumsum(q_lens)
    cu[n + 1 :] = cu[n]
    sl = np.zeros(MAX_SEQS, np.int32)
    sl[:n] = seq_lens
    dist = np.array([0, n_verify, n], np.int32)
    return dict(
        q=q,
        indexer_weights=w,
        cache_kv=cache,
        seq_lens=jnp.asarray(sl),
        page_indices=jnp.asarray(table.reshape(-1)),
        cu_q_lens=jnp.asarray(cu),
        distribution=jnp.asarray(dist),
    )


CHUNK = None
DIAG = False


def _kw(**extra):
    return dict(
        chunk_tokens=CHUNK,
        k=K,
        compression_ratio=RATIO,
        num_kv_pages_per_block=KV_PAGES,
        num_queries_per_block=Q_BLOCKS,
        decode_req_batch_size=4,
        kv_layout=KVLayout.SEQ_ALONG_LANE,
        uniform_q_len=VQ,
        **extra,
    )


REF_CHUNK = None


@jax.jit
def _unsplit(inp):
    # The harness's unchunked b=2048 SparseCore top-k overflows SparseCore
    # memory here (it fits in the served program); chunking does not change
    # the selection (TestChunkedPipelining), so the reference may chunk.
    kw = _kw()
    if REF_CHUNK and inp["q"].shape[0] > REF_CHUNK:
        kw["chunk_tokens"] = REF_CHUNK
    return streamindex_topk(**inp, **kw)


def _share(inp, rank, nsplit):
    idx, sc = streamindex_topk(
        **inp, **_kw(kv_split=nsplit, kv_split_rank=rank)
    )
    return jnp.concatenate([idx, sc], axis=1)


_share_jit = jax.jit(_share, static_argnums=(2,))
_merge_jit = jax.jit(merge_kv_split_topk, static_argnums=(1,))

CASES = {
    # 2 verify windows + one 2000-token chunk at 100k (the served T2048 mixed step)
    "T2048_v2_c2000@100k": ([(VQ, 100_007, "verify"), (VQ, 300_007, "verify"),
                             (2000, 102_352, "chunk")], 2048),
    "T2048_c2000@300k": ([(2000, 302_032, "chunk")], 2048),
    "T2048_c2000@fresh": ([(2000, 2000, "chunk")], 2048),
    "T512_v1_c500@100k": ([(VQ, 90_007, "verify"), (500, 100_852, "chunk")], 512),
    "T256_c250@500k": ([(250, 500_250, "chunk")], 256),
    # c16 TP4 mnbt 512: ~495-token chunk at 31.7k / 127k + 2 verify windows
    "T512_v2_c495@32k": ([(VQ, 40_007, "verify"), (VQ, 90_007, "verify"),
                          (495, 32_263, "chunk")], 512),
    "T512_v2_c495@127k": ([(VQ, 40_007, "verify"), (VQ, 90_007, "verify"),
                           (495, 127_495, "chunk")], 512),
}


def run_case(name, nsplit=8, time_it=False, iters=20):
    seqs, T = CASES[name]
    inp = _case(seqs, T)
    ref = np.asarray(_unsplit(inp))
    shares = [_share_jit(inp, jnp.int32(r), nsplit) for r in range(nsplit)]
    merged = np.asarray(_merge_jit(jnp.stack(shares), K))
    n_tok = int(np.asarray(inp["cu_q_lens"])[len(seqs)])
    bad = int(np.sum(np.any(ref[:n_tok] != merged[:n_tok], axis=1)))
    res = {"case": name, "rows": n_tok, "mismatch_rows": bad}
    if bad:
        r0 = int(np.argmax(np.any(ref[:n_tok] != merged[:n_tok], axis=1)))
        res["first_bad_row"] = r0
        res["ref"] = ref[r0, :8].tolist()
        res["got"] = merged[r0, :8].tolist()
        res["same_set"] = bool(set(ref[r0].tolist()) == set(merged[r0].tolist()))
    if time_it:
        def t(f):
            # Back-to-back async dispatches, one block: device time per call
            # once it exceeds the dispatch time.
            f().block_until_ready()
            xs = []
            for _ in range(3):
                t0 = time.perf_counter()
                outs = [f() for _ in range(iters)]
                outs[-1].block_until_ready()
                xs.append((time.perf_counter() - t0) * 1e3 / iters)
            return round(min(xs), 3)
        stacked = jnp.stack(shares)
        res["unsplit_ms"] = t(lambda: _unsplit(inp))
        res["share_r0_ms"] = t(lambda: _share_jit(inp, jnp.int32(0), nsplit))
        blk = stacked[:, : T // nsplit]
        res["merge_ms"] = t(lambda: _merge_jit(blk, K))
        if DIAG:
            from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
            R, TT, _ = blk.shape
            cand = blk.reshape(R, TT, 2, K).transpose(2, 1, 0, 3).reshape(2, TT, R * K)
            bits = jax.block_until_ready(cand[1])
            sc_full = jax.jit(lambda b: sparsecore_topk(b, K, row_lengths=jnp.full((TT,), R * K, jnp.int32)))
            sc_zero = jax.jit(lambda b: sparsecore_topk(b, K, row_lengths=jnp.zeros((TT,), jnp.int32)))
            res["sc_full_ms"] = t(lambda: sc_full(bits))
            res["sc_zero_ms"] = t(lambda: sc_zero(bits))
            tr = jax.jit(lambda x: x.reshape(R, TT, 2, K).transpose(2, 1, 0, 3).reshape(2, TT, R * K))
            res["transpose_ms"] = t(lambda: tr(blk))
    print(res, flush=True)
    return res


@pytest.mark.parametrize("name", list(CASES))
def test_kv_split_matches_unsplit(name):
    assert run_case(name)["mismatch_rows"] == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--nsplit", type=int, default=8)
    ap.add_argument("--ref-chunk", type=int, default=0)
    a = ap.parse_args()
    CHUNK = a.chunk or None
    DIAG = a.diag
    REF_CHUNK = a.ref_chunk or None
    bad = 0
    for c in a.cases.split(","):
        try:
            bad += run_case(c, nsplit=a.nsplit, time_it=a.time)["mismatch_rows"] > 0
        except Exception as e:  # noqa
            print({"case": c, "error": repr(e)[:2000]}, flush=True)
            bad += 1
    sys.exit(1 if bad else 0)
