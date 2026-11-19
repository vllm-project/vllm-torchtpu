# SPDX-License-Identifier: Apache-2.0
"""q38 L2 O1s-HOIST port 2: VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT against today's decode indexer.

Today the decode shared-stream indexer (pooled_qsa ``_select_blocks`` decode branch) calls StreamIndex
with ``enable_early_exit=True``: one ``lax.cond`` per layer (``max(seq_lens) // ratio <= k``) that lists
every visible group on a fast path instead of running the kernel. At served contexts (tens to
hundreds of thousands of tokens against k = 512 groups = 2048 tokens) it always takes the kernel path,
and the cond costs 3.2-3.4 us of exclusive time per layer, 0.040-0.041 ms per step (M, T85m c16 / c32
rank 0). MM's compressed-DCP decode indexer runs without it (int-5 ``_dcp_topk`` pops it).

The real StreamIndex kernel runs in TPU interpret mode on CPU with the JAX reference top-k for the
SparseCore selector (test_qwen4_exp_shared_stream_indexer's harness):
  * plumbing: the flag reaches StreamIndex as enable_early_exit=False, and only then;
  * served-like contexts (some row beyond the budget): the cond takes the kernel path today, so flag on
    and off are bit-identical;
  * short contexts (every row inside the budget, the fast path today), at the production budget
    (block_topk 512, contexts <= 2048 tokens) and at the harness budget: the kernel path selects the
    same set of groups per row (the order may differ; the attention consumes the set, and the
    SparseCore top-k's order is unspecified anyway). One summary line per case for the reader.
"""

import functools

import numpy as np
import pytest
from test_qwen4_exp_shared_stream_indexer import (  # noqa: F401 (interpret is a fixture)
    BLOCK_TOPK,
    RATIO,
    ReferenceTopK,
    inputs,
    interpret,
    streamindex,
)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

NO_EXIT = "VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT"


_MISSING = object()


@pytest.fixture
def envset(monkeypatch):
    """Set envs flags for one test, order-proof and leak-free: os.environ (monkeypatch) AND the module attribute,
    restored on teardown by deleting it (or putting back an attribute that was already there). Never
    monkeypatch.setattr(envs, ...): its undo writes the old value back as a frozen module attribute that shadows
    os.environ for every later test in the session (j-3307bbed / j-cab937b3)."""
    saved = {}

    def set_(name, value):
        if name not in saved:
            saved[name] = envs.__dict__.get(name, _MISSING)
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))
        envs.__dict__[name] = envs.environment_variables[name]()

    yield set_
    for name, old in saved.items():
        if old is _MISSING:
            envs.__dict__.pop(name, None)
        else:
            envs.__dict__[name] = old


# (q_lens per request slot, context per slot, tokens, block_topk). Tables hold 8 x 512 = 4096 tokens; the decode path
# needs tokens <= request slots and the live rows (sum of q_lens) <= tokens.
CASES = {
    # Production budget: 512 groups = 2048 tokens. Every row inside it (fast path today).
    "short_k512": ([4, 4, 3, 1] + [0] * 8, [2044, 1203, 7, 640] + [0] * 8, 12, 512),
    # The last row of the first request just past it (2052 // 4 = 513 groups): the kernel path today.
    "edge_k512": ([4, 4] + [0] * 6, [2048, 90] + [0] * 6, 8, 512),
    "served_like_k512": ([4, 4, 4, 4] + [0] * 12, [4000, 2049, 3001, 517] + [0] * 12, 16, 512),
    # Harness budget (16 groups = 64 tokens): short, and the mixed cases of the shared-stream suite.
    "short_k16": ([4, 3, 4] + [0] * 9, [5, 30, 60] + [0] * 9, 12, BLOCK_TOPK),
    "verify_k3": ([4, 4, 4] + [0] * 9, [3001, 2048, 1203] + [0] * 9, 12, BLOCK_TOPK),
    "short_and_long": ([4, 3] + [0] * 6, [5, 3000] + [0] * 6, 8, BLOCK_TOPK),
}
for _q, _c, _t, _k in CASES.values():
    assert len(_q) == len(_c) and sum(_q) <= _t <= len(_q), (_q, _c, _t)


def _spy(monkeypatch):
    seen = []
    real = functools.partial(streamindex.streamindex_topk, topk_backend=ReferenceTopK())

    def spy(*args, **kwargs):
        seen.append(kwargs.get("enable_early_exit"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", spy)
    return seen


@pytest.mark.parametrize("kv_block", [256, 1024])
@pytest.mark.parametrize("case", sorted(CASES))
def test_no_early_exit_selects_the_same_groups(monkeypatch, interpret, envset, case, kv_block):
    q_lens, context, tokens, k = CASES[case]
    args = inputs(len(case), q_lens, context, tokens, "integer")
    seen = _spy(monkeypatch)
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    for name in ("VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA",
                 "VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY",
                 "VLLM_TPU_QSA_INDEXER_HOIST_META", "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK"):
        envset(name, None)
    envset("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", 4)
    envset("VLLM_TPU_QSA_DECODE_KV_BLOCK", kv_block)

    def select():
        return np.asarray(pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=k))

    envset(NO_EXIT, None)
    off = select()
    assert seen and all(v is True for v in seen), seen
    seen.clear()
    envset(NO_EXIT, 1)
    on = select()
    assert seen and all(v is False for v in seen), seen

    lengths = np.asarray(args["lengths"])
    fast_today = int(lengths.max()) // RATIO <= k
    assert off.shape == on.shape
    for row in range(len(lengths)):
        a, b = off[row], on[row]
        assert np.array_equal(np.sort(a), np.sort(b)), f"{case} row {row}: selected groups differ"
        count = min(k, int(lengths[row]) // RATIO)
        live = b[b >= 0]
        assert len(live) == count and len(np.unique(live)) == count
        assert np.all(live < max(1, int(lengths[row]) // RATIO))
    exact = bool(np.array_equal(off, on))
    if not fast_today:
        assert exact, f"{case}: the kernel path today and with the flag must be bit-identical"
    print(f"Q38_L2_NOEXIT_KERNEL {case} kv_block={kv_block} fast_path_today={fast_today} exact={exact}")
