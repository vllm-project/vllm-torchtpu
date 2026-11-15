"""StreamIdxTC step time with VLLM_TPU_QSA_INDEXER_DENSE_STEP off and on.

The served decode indexer's scoring kernel alone (mixed pass, per-token
scores, 4 query rows a block, 3,328-key blocks) at the served geometry: FP8
unified pool of 512-token pages read as BF16 keys 256 wide, max_model_len
262144, compressed-DCP rank 0 of 4 on the expanded kernel-page table (``dcp``)
or on the pool-page table (``dcpp``, VLLM_TPU_QSA_DCP_POOL_PAGE_TABLE), or the
replicated call on the pool-page table (``rep``). Each case times ITERS back-to-back calls (wall
clock, median of 5) for R requests of 4 rows at one context; the step count
of a call is the sum over requests of their KV blocks, so the slope over
contexts is the time a (request, block) step and the intercept the fixed
part. One xprof trace per case goes to $CDK_OUTPUT_DIR/streamidx-bench/xprof
and, with BENCH_LLO=1, the kernels' LLO bundles to .../llo.tgz.

usage: python3 -u scripts/bench_qsa_indexer_dense_step.py
"""

import os
import sys
import tarfile
from pathlib import Path

OUT = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp")) / "streamidx-bench"
OUT.mkdir(parents=True, exist_ok=True)
DUMP = "/tmp/streamidx_llo"
if os.environ.get("BENCH_LLO") == "1":
    os.makedirs(DUMP, exist_ok=True)
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "")
        + f" --xla_jf_dump_to={DUMP} --xla_jf_dump_llo_text=true"
    ).strip()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataclasses  # noqa: E402
import functools  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing  # noqa: E402

sit = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)

RATIO, PAGE, DCP, SLOT, KV_BLOCK, KEYS = 4, 512, 4, 3, 3328, 32
MAX_MODEL_LEN, POOL_PAGES, ITERS = 262144, 4096, 20


@dataclasses.dataclass(frozen=True)
class _FirstColumns:
    overlaps_scoring = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        out = score_bits[:, :k]
        return (out, out) if return_scores else out


def make_pool():
    k1, k2 = jax.random.split(jax.random.key(0))
    pages = POOL_PAGES * 32
    keys = jax.random.normal(k1, (pages, 16, 2, 256), jnp.bfloat16)
    bits = jax.lax.bitcast_convert_type(keys, jnp.uint16)
    raw = jnp.stack(
        [(bits & 0xFF).astype(jnp.uint8), (bits >> 8).astype(jnp.uint8)], axis=3
    ).reshape(pages, 16, 4, 256)
    raw = jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn)
    return raw.reshape(POOL_PAGES, PAGE, 1, 4, 256)


def local_kv(seq, rank):
    kv = seq // RATIO
    if rank is None:
        return kv
    ic = PAGE // RATIO
    full, rem = kv // (DCP * ic), kv % (DCP * ic)
    return full * ic + np.clip(rem - rank * ic, 0, ic)


def case(pool, mode, context, requests, flag, rng, trace_dir, kv_block=KV_BLOCK,
         null_tail=False, tail_skip=0):
    rank = None if mode == "rep" else 0
    rows = 4 * requests
    span = PAGE * (DCP if rank is not None else 1)
    cols = MAX_MODEL_LEN // span
    table = np.stack(
        [rng.choice(np.arange(1, POOL_PAGES), size=cols, replace=False) for _ in range(requests)]
    ).astype(np.int32)
    if null_tail:
        # As vLLM's block tables: entries past the request's pages hold the
        # null block 0, so a partly filled last block reads pool page 0.
        table[:, -(-(context + 4) // span):] = 0
    if rank is None:
        pages = table
        expansion = pooled_qsa._compressed_expansion(4, 32, SLOT)
        extra = {}
    elif mode == "dcpp":
        pages = np.pad(table, ((0, 0), (0, (-table.shape[1]) % (kv_block // KEYS // 4))))
        expansion = pooled_qsa._compressed_expansion(4, 32, SLOT)
        extra = dict(cp_size=DCP, cp_rank=jnp.int32(0), interleave_size=PAGE)
    else:
        local = (table[..., None] * 32 + SLOT * 4 + np.arange(4)).reshape(requests, -1)
        pages = np.pad(local, ((0, 0), (0, (-local.shape[1]) % (kv_block // KEYS))))
        expansion = (1, 1, 0)
        extra = dict(cp_size=DCP, cp_rank=jnp.int32(0), interleave_size=PAGE)
    seq = np.full((requests,), context + 4, np.int32)
    starts = np.arange(requests + 1, dtype=np.int32) * 4
    q = jnp.asarray(rng.standard_normal((rows, 4, 128)), jnp.bfloat16)
    w = jnp.full((rows, 4), 128**-0.5, jnp.float32)
    cache = pool.reshape(-1, KEYS // 2, 4, 256)
    score = functools.partial(
        sit.streamindex_topk,
        k=512,
        compression_ratio=RATIO,
        num_kv_pages_per_block=(8, kv_block // KEYS, kv_block // KEYS),
        num_queries_per_block=(1, 4, 4),
        decode_req_batch_size=8,
        components=qsa_indexing(
            RATIO - 1, per_token_scores=True, table_expansion=expansion,
            coalesce_entries=True, dense_step=bool(flag), tail_skip=tail_skip,
        ),
        decode_passes=False,
        topk_backend=_FirstColumns(),
        **extra,
    )
    args = (
        jnp.asarray(seq), jnp.asarray(pages.reshape(-1)), jnp.asarray(starts),
        jnp.asarray([0, 0, requests], jnp.int32),
    )

    @jax.jit
    def loop(q, cache, seq, pages, starts, dist):
        def body(i, acc):
            out = score(q + (acc * 0).astype(q.dtype), w, cache, seq, pages, starts, dist)
            out = out[0] if isinstance(out, tuple) else out
            return acc + out[0, 0].astype(jnp.float32)

        return jax.lax.fori_loop(0, ITERS, body, jnp.float32(0))

    jax.block_until_ready(loop(q, cache, *args))
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(loop(q, cache, *args))
        samples.append((time.perf_counter() - t0) / ITERS * 1e6)
    with jax.profiler.trace(str(trace_dir)):
        jax.block_until_ready(loop(q, cache, *args))
    blocks = np.maximum(1, -(-local_kv(seq, rank) // kv_block))
    return dict(
        mode=mode, context=context, requests=requests, flag=flag, kv_block=kv_block,
        null_tail=null_tail, tail_skip=tail_skip, case=trace_dir.name,
        us_per_call=float(np.median(samples)), samples=samples,
        steps=int(blocks.sum()), blocks_per_request=int(blocks[0]),
    )


def main():
    pool = make_pool()
    results = []
    grid = [
        ("dcpp", c, 32) for c in (2048, 32768, 53000, 104000, 155000, 210000)
    ] + [("dcpp", 104000, r) for r in (8, 16)] + [
        ("dcp", c, 32) for c in (2048, 32768, 104000, 210000)
    ] + [
        ("rep", c, 32) for c in (2048, 32768, 104000)
    ] + [("rep", 104000, r) for r in (8, 16)]
    out = OUT / "results.jsonl"
    with open(out, "w") as f:
        # A small third block (108k: rank 0 holds 6,784 local keys) with the
        # tail entries null as in served tables, against random tail pages.
        # VLLM_TPU_QSA_INDEXER_TAIL_SKIP: page groups of 8 and 26.
        for i, (context, null_tail, flag, tail) in enumerate(
            [(c, n, fl, 0) for c in (104000, 108000) for n in (True, False) for fl in (0, 1)]
            + [(c, True, fl, t) for c in (104000, 108000) for fl in (0, 1) for t in (8, 26)]
        ):
            rng = np.random.default_rng(200 + i)
            r = case(pool, "dcpp", context, 32, flag, rng,
                     OUT / "xprof" / f"dcpp-{context}-32-f{flag}-n{int(null_tail)}-t{tail}",
                     null_tail=null_tail, tail_skip=tail)
            results.append(r)
            print("STREAMIDX", json.dumps(r), flush=True)
            f.write(json.dumps(r) + "\n")
            f.flush()
        # Cost of a last block by fill, served tables (null tail entries):
        # every request's rank-0 keys are 2 full blocks plus `fill` of a
        # third; fill 0 is exactly 2 blocks.
        for i, (fill, flag, tail) in enumerate(
            (fl, d, t) for fl in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
            for d, t in ((0, 0), (0, 8), (0, 26), (1, 0), (1, 8))
        ):
            keys = 2 * KV_BLOCK + round(fill * KV_BLOCK)
            kv = keys // 128 * 512 + keys % 128
            context = 4 * kv - 4
            rng = np.random.default_rng(300 + i)
            r = case(pool, "dcpp", context, 32, flag, rng,
                     OUT / "xprof" / f"fill-{fill}-f{flag}-t{tail}",
                     null_tail=True, tail_skip=tail)
            r["fill"] = fill
            results.append(r)
            print("STREAMIDX", json.dumps(r), flush=True)
            f.write(json.dumps(r) + "\n")
            f.flush()
        for i, (mode, context, requests) in enumerate(grid):
            for flag in (0, 1):
                rng = np.random.default_rng(i)
                r = case(pool, mode, context, requests, flag, rng,
                         OUT / "xprof" / f"{mode}-{context}-{requests}-f{flag}")
                results.append(r)
                print("STREAMIDX", json.dumps(r), flush=True)
                f.write(json.dumps(r) + "\n")
                f.flush()
        # The per-token scores do not depend on the KV block: other sizes,
        # flag on, for the per-step overhead against the masked tail.
        for i, (context, kv_block) in enumerate(
            (c, b) for b in (1664, 6656) for c in (32768, 104000, 190000)
        ):
            rng = np.random.default_rng(100 + i)
            r = case(pool, "dcpp", context, 32, 1, rng,
                     OUT / "xprof" / f"dcpp-{context}-32-f1-b{kv_block}", kv_block)
            results.append(r)
            print("STREAMIDX", json.dumps(r), flush=True)
            f.write(json.dumps(r) + "\n")
            f.flush()
    for fill in (0.1, 0.25, 0.5, 0.75, 1.0):
        line = {"fill": fill}
        for flag, tail in ((0, 0), (0, 8), (0, 26), (1, 0), (1, 8)):
            pts = {r["fill"]: r["us_per_call"] for r in results
                   if "fill" in r and r["flag"] == flag and r["tail_skip"] == tail}
            line[f"f{flag}t{tail}_us_per_last_block"] = (pts[fill] - pts[0.0]) / 32
        print("STREAMIDX_FILL", json.dumps(line), flush=True)
    for mode in ("dcpp", "dcp", "rep"):
        for flag in (0, 1):
            pts = [r for r in results if r["mode"] == mode and r["flag"] == flag
                   and not r["null_tail"]
                   and r["requests"] == 32 and r["kv_block"] == KV_BLOCK
                   and not r["null_tail"] and "fill" not in r]
            x = np.array([r["steps"] for r in pts], float)
            y = np.array([r["us_per_call"] for r in pts], float)
            slope, icpt = np.polyfit(x, y, 1)
            line = dict(mode=mode, flag=flag, us_per_step=slope, us_fixed_per_call=icpt)
            print("STREAMIDX_FIT", json.dumps(line), flush=True)
    if os.environ.get("BENCH_LLO") == "1":
        with tarfile.open(OUT / "llo.tgz", "w:gz") as tar:
            for p in sorted(Path(DUMP).rglob("*StreamIdxTC*")):
                if any(s in p.name for s in ("final_bundles", "schedule-analysis_final")):
                    tar.add(p, arcname=p.name)
        print("llo", (OUT / "llo.tgz").stat().st_size, flush=True)


if __name__ == "__main__":
    main()
