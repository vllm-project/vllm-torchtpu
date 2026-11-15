# SPDX-License-Identifier: Apache-2.0
"""TPU check of VLLM_TPU_QSA_DECODE_INDEXER_REQ_SHARD: the request-sharded
decode indexer against today's replicated one, on all devices of the slice.

Per bucket (``rows / 4`` verify requests, one context band) and under one
``shard_map`` over every device (one device = one TP rank):

  whole     every rank runs ``pooled_qsa.select_blocks`` on all rows (today)
  sharded   every rank runs ``pooled_qsa.token_shard_selection`` on its
            1/TP of the rows (row scatter: token_shard_step + slices + the
            requests_within_shards cond), then the selection all-gather
  scatter   ``sharded`` with the indexer replaced by a constant: the row
            scatter + cond + all-gather overhead alone

Bitwise: the gathered selection must equal ``whole`` on every rank (same ids
in the same order); "set" compares sorted rows. Prints ST lines (kernel=reqshard).
Env: ST_RS_ROWS (48,64,96), ST_RS_CTX (108000,200000).
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import common  # noqa: F401  (SERVED_ENV before vllm_torchtpu imports)
from common import bench, digest

import bench_indexer as bi

ROWS = [int(x) for x in os.environ.get("ST_RS_ROWS", "48,64,96").split(",")]
CTXS = [int(x) for x in os.environ.get("ST_RS_CTX", "108000,200000").split(",")]


def main(emit):
    from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

    devs = np.array(jax.devices())
    tp = len(devs)
    mesh = Mesh(devs, ("x",))
    rep = NamedSharding(mesh, P())
    print("reqshard devices", tp, flush=True)

    def select(query, *, pool, table, lengths, req, starts, context, max_query_len):
        return pooled_qsa.select_blocks(
            query, pool, lengths, table, req, starts, context,
            ratio=bi.RATIO, block_topk=bi.BLOCK_TOPK, max_query_len=max_query_len,
        )

    def make(arm):
        def per_rank(pool, query, lengths, table, req, starts, context, acc):
            bump = (acc > 1e30).astype(jnp.int32)
            context = context + bump
            if arm == "whole":
                sel = select(query, pool=pool, table=table, lengths=lengths, req=req,
                             starts=starts, context=context, max_query_len=bi.MTP_ROWS)
            else:
                index = jax.lax.axis_index("x").astype(jnp.int32).reshape(1)
                if arm == "scatter":
                    def fn(q, *, lengths, req, starts, context, max_query_len):
                        return jnp.zeros((len(q), bi.BLOCK_TOPK), jnp.int32) + context[0] * 0
                else:
                    def fn(q, **kw):
                        return select(q, pool=pool, table=table, **kw)
                own = pooled_qsa.token_shard_selection(
                    fn, query, lengths, req, starts, context, bi.MTP_ROWS, (index, tp)
                )
                sel = jax.lax.all_gather(own, "x", tiled=True)
            return pool, sel, acc + (sel[0, 0] == -7).astype(jnp.float32)

        f = jax.shard_map(per_rank, mesh=mesh, in_specs=(P(),) * 8,
                          out_specs=(P(), P(), P()), check_vma=False)
        return f

    pool = None
    for ctx in CTXS:
        for rows in ROWS:
            need = (rows // 4) * (-(-int(ctx * 1.1 + 8) // bi.PAGE))
            pages = max(2048, need + 16)
            if pool is None or pool.shape[0] != pages:
                pool = None
                pool = jax.device_put(bi.make_pool(pages), rep)
            host, _ = bi.decode_inputs(rows, int(ctx * 0.9), int(ctx * 1.1), seed=rows)
            ops = [jax.device_put(jnp.asarray(host[k]), rep) for k in
                   ("query", "lengths", "table", "req", "starts", "context")]
            ops[0] = ops[0].astype(jnp.bfloat16)
            bucket = {"rows": rows, "ctx": ctx, "tp": tp}
            ref = None
            for arm in ("whole", "sharded", "scatter"):
                f = make(arm)
                try:
                    one = jax.jit(f)
                    pool, sel, _ = one(pool, *ops, jnp.float32(0))
                    sel = np.asarray(jax.device_get(sel))
                    note = ""
                    if arm == "whole":
                        ref = sel
                    elif arm == "sharded":
                        same = np.array_equal(sel, ref)
                        same_set = np.array_equal(np.sort(sel, -1), np.sort(ref, -1))
                        note = f"bitwise={same} set={same_set}"

                    def body(c, f=f):
                        p, s, a = f(c[0], *c[1:7], c[7])
                        return (p, *c[1:7], a)

                    carry = (pool, *ops, jnp.float32(0))
                    us, carry = bench(body, carry)
                    pool = carry[0]
                    ops = list(carry[1:7])
                    emit(bucket, {"arm": arm}, round(us, 2), digest(sel), note=note or None)
                except Exception as e:  # noqa: BLE001
                    emit.error(bucket, {"arm": arm}, e)
                    if pool.is_deleted():
                        pool = jax.device_put(bi.make_pool(pages), rep)
                    ops = [o if not o.is_deleted() else None for o in ops]
                    if any(o is None for o in ops):
                        ops = [jax.device_put(jnp.asarray(host[k]), rep) for k in
                               ("query", "lengths", "table", "req", "starts", "context")]
                        ops[0] = ops[0].astype(jnp.bfloat16)
