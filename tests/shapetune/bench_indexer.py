# SPDX-License-Identifier: Apache-2.0
"""Shape sweep of the QSA indexer (StreamIndex top-k) tile knobs, one v7x chip.

Models one TP8 rank's served ``pooled_qsa.select_blocks`` call:

* decode/verify (target layers, MTP K=3): ``pooled_qsa_dcp_step`` ->
  ``_pooled_qsa`` -> ``select_blocks(iq, pool, lengths, compressed_table, req,
  starts, context, ratio=4, block_topk=512, max_query_len=4)`` with no
  shard/dcp (main-KV DCP only splits the attention; VLLM_TPU_QSA_COMPRESSED_DCP
  and VLLM_TPU_QSA_DECODE_TOKEN_SHARD are off). ``rows`` tokens =
  ``rows / 4`` requests of 4 tokens -> the shared-stream path
  (``_select_shared_stream``), mixed pass only (VLLM_TPU_QSA_INDEXER_MIXED_ONLY).
* prefill (SP graph, VLLM_TPU_QSA_INDEXER_TOKEN_SHARD): each rank runs
  ``select_blocks`` on its ``chunk / 8`` token rows with ``max_query_len=1``
  and ``token_shard_step``'s re-based starts/context -> the ragged prefill
  indexer. We model the last rank (widest context: prefix + 7/8 chunk).

Indexer: 4 replicated heads x 128, BF16 keys in the FP8 unified pool
(``[pages, 512, 1, 4, 256]`` carrier, low/high bytes in head slots 0::2/1::2),
page 512, compressed slot 0, table width max_model_len / 512.

Knobs are env vars read at trace time. Config dicts are the env overrides;
``_CHUNK`` (decode only) is a pseudo-knob that caps ``decode_query_chunk_size``
(the shared-stream window) by monkeypatching ``pooled_qsa``.

Env: ST_IDX_POOL_PAGES (2048), ST_IDX_SLOTS (prefill request slots, 32),
ST_IDX_MAX_MODEL_LEN (262144), ST_IDX_BUDGET_S (2400: after this, only
default configs run), ST_IDX_ONLY=decode|prefill.
"""

import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import common  # noqa: F401  (SERVED_ENV before vllm_torchtpu imports)
from common import bench, digest, env

RATIO, BLOCK_TOPK, PAGE, WIDTH = 4, 512, 512, 256
HEADS, HEAD_DIM = 4, 128
TP = 8
MTP_ROWS = 4  # K=3 verify: 4 tokens per request
MAX_MODEL_LEN = int(os.environ.get("ST_IDX_MAX_MODEL_LEN", "262144"))
TABLE_COLS = MAX_MODEL_LEN // PAGE
POOL_PAGES = int(os.environ.get("ST_IDX_POOL_PAGES", "2048"))
SLOTS = int(os.environ.get("ST_IDX_SLOTS", "32"))
BUDGET_S = float(os.environ.get("ST_IDX_BUDGET_S", "2400"))
ONLY = os.environ.get("ST_IDX_ONLY", "")

DEC_DEFAULT = {"VLLM_TPU_QSA_DECODE_KV_BLOCK": 3328, "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": 4}
PRE_DEFAULT = {"VLLM_TPU_QSA_PREFILL_INDEXER_BKV": 1024, "VLLM_TPU_QSA_PREFILL_INDEXER_BQ": 128,
               "VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK": 0}


# ----------------------------------------------------------------- inputs


@functools.cache
def _pool_builder(pages):
    @jax.jit
    def build(key):
        keys = jax.random.normal(key, (pages, PAGE, 2, WIDTH), jnp.bfloat16)
        u = jax.lax.bitcast_convert_type(keys, jnp.uint16)
        lo = (u & 255).astype(jnp.uint8)
        hi = (u >> 8).astype(jnp.uint8)
        raw = jnp.stack([lo, hi], axis=3).reshape(pages, PAGE, 1, 4, WIDTH)
        return jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn)

    return build


def make_pool(pages=None):
    """BF16 index keys in the FP8 carrier, built on device (no host copy)."""
    return _pool_builder(pages or POOL_PAGES)(jax.random.key(0))


def _table_row(rng, tokens):
    row = np.zeros(TABLE_COLS, np.int32)
    cols = -(-tokens // PAGE)
    assert cols <= min(TABLE_COLS, POOL_PAGES), (tokens, POOL_PAGES)
    row[:cols] = rng.permutation(POOL_PAGES)[:cols]
    return row


def decode_inputs(rows, ctx_lo, ctx_hi, seed=0):
    """``rows / 4`` verify requests, contexts uniform in [ctx_lo, ctx_hi)."""
    reqs = rows // MTP_ROWS
    rng = np.random.default_rng(seed)
    context = rng.integers(ctx_lo, ctx_hi, size=reqs).astype(np.int32)
    table = np.stack([_table_row(rng, c + MTP_ROWS) for c in context])
    req = np.repeat(np.arange(reqs, dtype=np.int32), MTP_ROWS)
    lengths = context[req] + np.tile(np.arange(1, MTP_ROWS + 1, dtype=np.int32), reqs)
    return dict(
        query=rng.standard_normal((rows, HEADS, HEAD_DIM)).astype(np.float32),
        lengths=lengths.astype(np.int32),
        table=table,
        req=req,
        starts=(np.arange(reqs + 1) * MTP_ROWS).astype(np.int32),
        context=context,
    ), MTP_ROWS


def prefill_inputs(chunk, ctx, seed=0):
    """Last TP rank's token shard of one ``chunk``-token prefill on ``ctx``."""
    width = chunk // TP
    rank = TP - 1
    rng = np.random.default_rng(seed)
    context = np.zeros(SLOTS, np.int32)
    context[0] = ctx + rank * width  # token_shard_step: rows before the shard
    table = np.zeros((SLOTS, TABLE_COLS), np.int32)
    table[0] = _table_row(rng, ctx + chunk)
    starts = np.full(SLOTS + 1, width, np.int32)
    starts[0] = 0
    return dict(
        query=rng.standard_normal((width, HEADS, HEAD_DIM)).astype(np.float32),
        lengths=(context[0] + np.arange(1, width + 1)).astype(np.int32),
        table=table,
        req=np.zeros(width, np.int32),
        starts=starts,
        context=context,
    ), 1


# ------------------------------------------------------------------ bench


class Case:
    """Device carry for one bucket; rebuilt if a failed run ate the donation."""

    def __init__(self, host, max_query_len, pool):
        self.host, self.mql = host, max_query_len
        self.carry = self._carry(pool)

    def _carry(self, pool):
        h = self.host
        return dict(
            pool=pool,
            query=jnp.asarray(h["query"], jnp.bfloat16),
            lengths=jnp.asarray(h["lengths"]),
            table=jnp.asarray(h["table"]),
            req=jnp.asarray(h["req"]),
            starts=jnp.asarray(h["starts"]),
            context=jnp.asarray(h["context"]),
            sel=jnp.zeros((len(h["query"]), BLOCK_TOPK), jnp.int32),
            acc=jnp.zeros((), jnp.float32),
        )

    def ensure(self):
        leaves = jax.tree_util.tree_leaves(self.carry)
        if any(x.is_deleted() for x in leaves):
            pool = self.carry["pool"]
            self.carry = self._carry(make_pool() if pool.is_deleted() else pool)

    def body(self, c):
        from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

        # Never true; makes every call depend on the carry (no hoisting).
        bump = (c["acc"] > 1e30).astype(jnp.int32)
        sel = pooled_qsa.select_blocks(
            c["query"], c["pool"], c["lengths"], c["table"], c["req"], c["starts"],
            c["context"] + bump, ratio=RATIO, block_topk=BLOCK_TOPK,
            max_query_len=self.mql,
        )
        acc = c["acc"] + (sel[0, 0] == -7).astype(jnp.float32)
        return dict(c, sel=sel, acc=acc)


def _chunk_cap(cap):
    """Context manager capping the decode shared-stream window at ``cap``."""
    from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

    orig = pooled_qsa.decode_query_chunk_size

    class _Patch:
        def __enter__(self):
            if cap:
                pooled_qsa.decode_query_chunk_size = lambda p: min(orig(p), cap)

        def __exit__(self, *exc):
            pooled_qsa.decode_query_chunk_size = orig

    return _Patch()


class Runner:
    def __init__(self, emit):
        self.emit = emit
        self.t0 = time.time()
        self.pool = None

    def over_budget(self):
        return time.time() - self.t0 > BUDGET_S

    def run_bucket(self, bucket, host_and_mql, configs, defaults):
        if self.pool is None or self.pool.is_deleted():
            self.pool = make_pool()
        host, mql = host_and_mql
        case = Case(host, mql, self.pool)
        ref = None
        for i, cfg in enumerate(configs):
            default = i == 0
            if not default and self.over_budget():
                self.emit(bucket, cfg, None, note="SKIP budget")
                continue
            knobs = {**defaults, **cfg}
            cap = knobs.pop("_CHUNK", 0)
            case.ensure()
            try:
                with env(**knobs), _chunk_cap(cap):
                    us, case.carry = bench(case.body, case.carry)
                sel = np.asarray(jax.device_get(case.carry["sel"]))
                dg = digest(sel)
                sdg = digest(np.sort(sel, axis=-1))
                note = f"set={sdg}"
                if ref is None and default:
                    ref = (dg, sdg)
                elif ref is not None and dg != ref[0]:
                    note += " ORDER_DIFF" if sdg == ref[1] else " SET_DIFF"
                self.emit(bucket, cfg, round(us, 2), dg, default=default, note=note)
            except Exception as e:  # noqa: BLE001
                self.emit.error(bucket, cfg, e)
            self.pool = case.carry["pool"]


# ---------------------------------------------------------------- sweeps


def _dec(kv, qb, **extra):
    return {"VLLM_TPU_QSA_DECODE_KV_BLOCK": kv, "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": qb, **extra}


def decode_configs(rows):
    """Served default first. KV blocks: multiples of 128 (and of 32 keys/page)."""
    base = [_dec(3328, 4), _dec(1664, 4), _dec(6656, 4), _dec(3328, 2), _dec(3328, 8)]
    if rows >= 64:
        base += [_dec(2560, 4), _dec(4992, 4), _dec(6656, 8)]
    return base


def prefill_configs(chunk, wide):
    """Served default first. CHUNK must divide the rank's rows (chunk / 8) and
    be a multiple of 8; 0 = default (512 if it divides the rows, else one call);
    CHUNK = rows is one call, the same as None."""
    rows = chunk // TP

    def c(bkv=1024, bq=128, ch=0):
        return {"VLLM_TPU_QSA_PREFILL_INDEXER_BKV": bkv,
                "VLLM_TPU_QSA_PREFILL_INDEXER_BQ": bq,
                "VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK": ch}

    out = [c(), c(bkv=512), c(bkv=2048), c(bkv=4096), c(bq=256)]
    if wide:
        out += [c(bq=64), c(bkv=2048, bq=256)]
    default_chunk = 512 if rows % 512 == 0 else rows
    if chunk >= 8192 and not wide:
        out = [out[0], out[2], out[4]]
    chunks = (rows // 4, rows // 2, rows) if wide else (rows,)
    for ch in chunks:
        if ch >= 64 and ch % 8 == 0 and ch != default_chunk:
            out.append(c(ch=ch))
    return out


def buckets():
    """(bucket, inputs thunk, configs, defaults) in priority order."""
    out = []
    if ONLY != "prefill":
        for rows in (128, 96, 64):
            out.append(({"phase": "decode", "rows": rows, "ctx": 100000},
                        functools.partial(decode_inputs, rows, 60000, 140000),
                        decode_configs(rows), DEC_DEFAULT))
    if ONLY != "decode":
        for chunk in (1024, 2048, 4096):
            out.append(({"phase": "prefill", "chunk": chunk, "ctx": 131072},
                        functools.partial(prefill_inputs, chunk, 131072),
                        prefill_configs(chunk, wide=True), PRE_DEFAULT))
    if ONLY != "prefill":
        for rows in (32, 16, 8, 4):
            out.append(({"phase": "decode", "rows": rows, "ctx": 100000},
                        functools.partial(decode_inputs, rows, 60000, 140000),
                        decode_configs(rows), DEC_DEFAULT))
        out.append(({"phase": "decode", "rows": 32, "ctx": 200000},
                    functools.partial(decode_inputs, 32, 190000, 210000),
                    decode_configs(32), DEC_DEFAULT))
    if ONLY != "decode":
        for ctx in (65536, 204800):
            for chunk in (1024, 2048, 4096):
                out.append(({"phase": "prefill", "chunk": chunk, "ctx": ctx},
                            functools.partial(prefill_inputs, chunk, ctx),
                            prefill_configs(chunk, wide=False), PRE_DEFAULT))
        for ctx in (65536, 131072, 204800):
            out.append(({"phase": "prefill", "chunk": 8192, "ctx": ctx},
                        functools.partial(prefill_inputs, 8192, ctx),
                        prefill_configs(8192, wide=False), PRE_DEFAULT))
    return out


def main(emit):
    runner = Runner(emit)
    for bucket, inputs, configs, defaults in buckets():
        print(f"indexer bucket {bucket}: {len(configs)} configs "
              f"(t={time.time() - runner.t0:.0f}s)", flush=True)
        try:
            runner.run_bucket(bucket, inputs(), configs, defaults)
        except Exception as e:  # noqa: BLE001
            emit.error(bucket, {}, e)
