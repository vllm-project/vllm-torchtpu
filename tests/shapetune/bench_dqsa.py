# SPDX-License-Identifier: Apache-2.0
"""Removal-arm attribution of the stock QSA decode layer (c32, b48-b96, DCP-2).

One TP8 rank's served stock decode QSA path, split into the parts the c32
decode budget prices, each timed per layer call at the served shapes
(``rows / 4`` requests of 4 MTP verify rows, one context band per bucket):

  idx_full     ``pooled_qsa.select_blocks`` decode path (StreamIndex score +
               SparseCore top-k), served env (shared stream, query block 4)
  idx_notopk   the same with the SparseCore top-k replaced by a slice of the
               score bits (removal arm: score cost alone)
  idx_shard8   ``select_blocks`` on ceil(reqs / 8) requests (one rank's share
               of a request-sharded indexer) ...
  ag_sel       ... plus the all-gather of the selections over all devices
  rpad_h3/h6   ``qsa_rpa.qsa_attention(dcp=(0, 2))``, 3 or 6 q heads per rank
  rpad_half    h3 with half the selected groups (per-row DMA vs fixed test)
  rpad_union   h3 with one selection per request (all 4 rows identical: the
               upper bound of a per-request union fetch)
  combine      ``qsa_rpa.dcp_combine_own`` on the pair's partials
  ag_outlse    all-gather of (out f32, lse) over a 2-device group

Env: ST_DQSA_ROWS (48,64,96), ST_DQSA_CTX (108000), ST_DQSA_ARMS (all).
Output: "ST {json}" lines (kernel=dqsa) and $CDK_OUTPUT_DIR/shapetune_dqsa.jsonl.
"""

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np

import common  # noqa: F401  (SERVED_ENV before vllm_torchtpu imports)
from common import bench, try_config

import bench_indexer as bi
import bench_rpad as br

ROWS = [int(x) for x in os.environ.get("ST_DQSA_ROWS", "48,64,96").split(",")]
CTXS = [int(x) for x in os.environ.get("ST_DQSA_CTX", "108000").split(",")]
ARMS = os.environ.get("ST_DQSA_ARMS", "").split(",") if os.environ.get("ST_DQSA_ARMS") else None
TP = 8


def want(arm):
    return ARMS is None or arm in ARMS


# ------------------------------------------------------------------ indexer


@dataclasses.dataclass(frozen=True)
class _NoTopK:
    """Removal arm: keep the score buffer, skip the SparseCore selection."""

    overlaps_scoring = False
    bounded_copy: bool = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        out = score_bits[:, :k]
        return (out, out) if return_scores else out


def _idx_case(rows, ctx, pool, seed):
    host, mql = bi.decode_inputs(rows, int(ctx * 0.9), int(ctx * 1.1), seed=seed)
    return bi.Case(host, mql, pool)


def run_indexer(emit, rows, ctx, state):
    from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

    need = (rows // 4) * (-(-int(ctx * 1.1 + 8) // bi.PAGE))
    pages = max(2048, need + 16)
    if state.get("idx_pages") != pages:
        state.pop("idx_pool", None)
        jax.clear_caches()
        bi.POOL_PAGES = pages
        state["idx_pool"] = bi.make_pool(pages)
        state["idx_pages"] = pages
    bucket = {"rows": rows, "ctx": ctx}

    def timed(case):
        us, case.carry = bench(case.body, case.carry)
        state["idx_pool"] = case.carry["pool"]
        return us, None

    if want("idx_full"):
        case = _idx_case(rows, ctx, state["idx_pool"], 0)
        try_config(emit, bucket, {"arm": "idx_full"}, lambda: timed(case))
    if want("idx_notopk"):
        orig = pooled_qsa.SparseCoreTopK
        pooled_qsa.SparseCoreTopK = _NoTopK
        try:
            case = _idx_case(rows, ctx, state["idx_pool"], 0)
            try_config(emit, bucket, {"arm": "idx_notopk"}, lambda: timed(case))
        finally:
            pooled_qsa.SparseCoreTopK = orig
    if want("idx_shard8"):
        sub = 4 * max(1, -(-(rows // 4) // TP))
        case = _idx_case(sub, ctx, state["idx_pool"], 1)
        try_config(emit, bucket, {"arm": "idx_shard8", "rows_per_rank": sub}, lambda: timed(case))
        if want("ag_sel"):
            try_config(emit, bucket, {"arm": "ag_sel", "rows_per_rank": sub},
                       lambda: (all_gather_us((sub, bi.BLOCK_TOPK), jnp.int32, TP), None))


# ---------------------------------------------------------------- attention


def _rpad_inputs(rows, ctx, mode, heads, seed):
    rng = np.random.default_rng(seed)
    br.CTX_LO, br.CTX_HI = int(ctx * 0.9), int(ctx * 1.1)
    q, g, blocks, lengths, table, starts = br.build(rng, rows)
    blocks = np.array(blocks)
    if mode == "half":
        blocks[..., br.BUDGET // 2:] = -1
    elif mode == "union":
        for r in range(rows // 4):
            blocks[:, 4 * r + 1: 4 * r + 4] = blocks[:, 4 * r: 4 * r + 1]
    if heads != br.HEADS:
        q = jax.random.normal(jax.random.key(1), (br.NL, rows, heads, br.HD), jnp.bfloat16)
        g = jax.random.normal(jax.random.key(2), (br.NL, rows, heads, br.HD), jnp.bfloat16)
    return q, g, jnp.asarray(blocks), lengths, table, starts


def run_rpad(emit, rows, ctx, state):
    if state.get("rpad_cache") is None or state["rpad_cache"].is_deleted():
        state["rpad_cache"] = br.make_cache()
    bucket = {"rows": rows, "ctx": ctx}
    for arm, mode, heads in (("rpad_h3", "full", 3), ("rpad_h6", "full", 6),
                             ("rpad_half", "half", 3), ("rpad_union", "union", 3)):
        if not want(arm):
            continue
        inputs = _rpad_inputs(rows, ctx, mode, heads, 0)

        def run():
            _, body = br.make_step(*inputs)
            us, (cache, _) = bench(body, (state["rpad_cache"], jnp.float32(0)), iters=10)
            state["rpad_cache"] = cache
            return us / br.NL, None

        try_config(emit, bucket, {"arm": arm, "heads": heads}, run)
        if state["rpad_cache"].is_deleted():
            state["rpad_cache"] = br.make_cache()


# ------------------------------------------------------- combine / collectives


def run_combine(emit, rows, ctx, state):
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    bucket = {"rows": rows, "ctx": ctx}
    heads = 3
    if want("combine"):
        k = jax.random.key(5)
        carry = (
            jax.random.normal(k, (2, rows, heads, br.HD), jnp.float32),
            jax.random.normal(k, (2, rows, heads), jnp.float32),
            jax.random.normal(k, (rows, heads, br.HD), jnp.bfloat16),
            jnp.zeros((rows, heads, br.HD), jnp.bfloat16),
        )

        def body(c):
            out_pair, lse_pair, gate, acc = c
            o = qsa_rpa.dcp_combine_own(out_pair, lse_pair, gate)
            # feed back so XLA cannot hoist
            return out_pair, lse_pair, gate + (o * 0).astype(gate.dtype), acc + o

        try_config(emit, bucket, {"arm": "combine"}, lambda: (bench(body, carry)[0], None))
    if want("ag_outlse"):
        # 6 gathered heads (the pair's q heads) of out f32 [T, 6, D] + lse.
        try_config(emit, bucket, {"arm": "ag_outlse"},
                   lambda: (all_gather_us((rows, 2 * heads, br.HD + 1), jnp.float32, 2), None))


def _ag_fn(shape, dtype, n):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devs = np.array(jax.devices()[:n])
    mesh = Mesh(devs, ("x",))

    def body(c):
        x, acc = c
        y = jax.lax.all_gather(x, "x", tiled=True)
        return x + (y[: x.shape[0]] * 0).astype(x.dtype), acc + y[0].astype(jnp.float32).sum()

    f = jax.shard_map(lambda x, a: body((x, a)), mesh=mesh,
                      in_specs=(P("x"), P()), out_specs=(P("x"), P()), check_vma=False)
    full = (shape[0] * n, *shape[1:])
    x = jax.device_put(jnp.zeros(full, dtype), NamedSharding(mesh, P("x")))
    a = jax.device_put(jnp.zeros((), jnp.float32), NamedSharding(mesh, P()))
    return f, x, a


def all_gather_us(shape, dtype, n):
    if len(jax.devices()) < n:
        raise RuntimeError(f"need {n} devices, have {len(jax.devices())}")
    f, x, a = _ag_fn(tuple(shape), dtype, n)
    us, _ = bench(lambda c: f(*c), (x, a))
    return us


def main(emit):
    state = {}
    print("devices", len(jax.devices()), jax.devices()[0].device_kind, flush=True)
    for ctx in CTXS:
        for rows in ROWS:
            print(f"dqsa rows={rows} ctx={ctx}", flush=True)
            for fn in (run_indexer, run_rpad, run_combine):
                try:
                    fn(emit, rows, ctx, state)
                except Exception as e:  # noqa: BLE001
                    emit.error({"rows": rows, "ctx": ctx}, {"stage": fn.__name__}, e)
