# SPDX-License-Identifier: Apache-2.0
"""Bitwise + timing check of VLLM_TPU_QSA_PREFILL_INDEXER_SMALL_CHUNK, one v7x chip.

``python tests/shapetune/run.py smallchunk``. For every small prefill indexer step
(at most 1024 rows per rank) the served default (one call, or two 512-row chunks)
runs first; then SMALL_CHUNK=128 and 256. The ``select_blocks`` output is hashed
per config: the line's ``note`` carries ``ORDER_DIFF`` / ``SET_DIFF`` if a chunked
config differs from the default. Timing is the median us per call.

Buckets (one TP8 rank, as ``bench_indexer``):

* replicated mixed steps (SP off for <= 1024 tokens): 24 verify requests x 4 rows
  plus one prefill request carrying the rest of the bucket, ``max_query_len=4``,
  so ``select_blocks`` takes the ragged prefill branch. Buckets 256 / 512 / 1024.
* SP token shard: the last rank's ``chunk / 8`` rows of a 2K / 4K / 8K chunk
  (``bench_indexer.prefill_inputs``).

Contexts 65536 and 131072. Env: ST_SC_CTX (comma list) overrides the contexts.
"""

import functools
import os
import time

import numpy as np

import bench_indexer as bi

SMALL = (128, 256)
DECODE_REQS = 24  # served max_num_seqs at c32
CTXS = tuple(int(c) for c in os.environ.get("ST_SC_CTX", "65536,131072").split(","))
DEFAULT = {**bi.PRE_DEFAULT, "VLLM_TPU_QSA_PREFILL_INDEXER_SMALL_CHUNK": 0}


def mixed_inputs(bucket, ctx, seed=0):
    """24 verify requests (contexts around ``ctx``) + one prefill of the rest."""
    rng = np.random.default_rng(seed)
    dec_rows = DECODE_REQS * bi.MTP_ROWS
    pre_rows = bucket - dec_rows
    assert pre_rows > 0, bucket
    context = np.zeros(bi.SLOTS, np.int32)
    context[:DECODE_REQS] = rng.integers(ctx // 2, ctx + ctx // 2, size=DECODE_REQS)
    context[DECODE_REQS] = ctx
    table = np.zeros((bi.SLOTS, bi.TABLE_COLS), np.int32)
    for r in range(DECODE_REQS):
        table[r] = bi._table_row(rng, int(context[r]) + bi.MTP_ROWS)
    table[DECODE_REQS] = bi._table_row(rng, ctx + pre_rows)
    qlens = np.zeros(bi.SLOTS, np.int32)
    qlens[:DECODE_REQS] = bi.MTP_ROWS
    qlens[DECODE_REQS] = pre_rows
    starts = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    req = np.repeat(np.arange(bi.SLOTS, dtype=np.int32), qlens)
    within = np.arange(bucket, dtype=np.int32) - starts[req]
    lengths = context[req] + within + 1
    return dict(
        query=rng.standard_normal((bucket, bi.HEADS, bi.HEAD_DIM)).astype(np.float32),
        lengths=lengths.astype(np.int32),
        table=table,
        req=req,
        starts=starts,
        context=context,
    ), bi.MTP_ROWS


def configs(rows):
    out = [{}]  # served default first: the digest reference
    for small in SMALL:
        if small < rows and rows % small == 0:
            out.append({"VLLM_TPU_QSA_PREFILL_INDEXER_SMALL_CHUNK": small})
    return out


def buckets():
    out = []
    for ctx in CTXS:
        for bucket in (256, 512, 1024):
            out.append(({"phase": "replicated", "bucket": bucket, "ctx": ctx},
                        functools.partial(mixed_inputs, bucket, ctx), configs(bucket)))
        for chunk in (2048, 4096, 8192):
            out.append(({"phase": "sp", "chunk": chunk, "ctx": ctx},
                        functools.partial(bi.prefill_inputs, chunk, ctx),
                        configs(chunk // bi.TP)))
    return out


def main(emit):
    runner = bi.Runner(emit)
    for bucket, inputs, cfgs in buckets():
        print(f"smallchunk bucket {bucket}: {len(cfgs)} configs "
              f"(t={time.time() - runner.t0:.0f}s)", flush=True)
        try:
            runner.run_bucket(bucket, inputs(), cfgs, DEFAULT)
        except Exception as e:  # noqa: BLE001
            emit.error(bucket, {}, e)
