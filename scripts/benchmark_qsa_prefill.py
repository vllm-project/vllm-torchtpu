# SPDX-License-Identifier: Apache-2.0
"""Compare QSA prefill with unchanged batched RPA on identical selected tokens.

Run on an allocated TPU with the repository's pinned dependencies:
  PYTHONPATH=src python scripts/benchmark_qsa_prefill.py --output /tmp/qsa-prefill

Includes selection metadata, attention and output gating; excludes the indexer,
cache writes, compilation and warmup. Does not benchmark decode or a model.
"""

import argparse
import functools
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16


# Benchmark-only baseline extracted from qsa_rpa.py at f7bd08a6e. It builds
# virtual pages and calls the existing batched RPA API without modifying it.
# Keeping the adapter here makes this kernel PR independent of model integration.
def virtual_page_metadata(blocks, lengths, table, *, ratio=4, page_size=128):
    """Map selected logical groups to physical micro-pages; no KV gathering."""
    batch, budget = blocks.shape
    complete = lengths // ratio
    count = jnp.minimum(complete, budget)
    columns = jnp.arange(budget + 1)[None, :]
    groups = jnp.concatenate((blocks, jnp.zeros((batch, 1), jnp.int32)), axis=1)
    groups = jnp.where(columns == count[:, None], complete[:, None], groups)
    valid = (columns < count[:, None]) | (
        (columns == count[:, None]) & ((lengths % ratio)[:, None] > 0)
    )
    safe = jnp.where(valid, groups, 0)
    # Reuse sparse MLA's vector lookup for small page tables; the generic
    # gather lowers to a slower indirect lookup on this path.
    from vllm_torchtpu.kernels.mla.sparse.kernel import gather_page_ids

    physical = gather_page_ids(
        table.reshape(-1),
        safe // (page_size // ratio),
        jnp.arange(batch, dtype=jnp.int32),
        batch,
    )
    pages = jnp.where(
        valid, physical * (page_size // ratio) + safe % (page_size // ratio), 0
    )
    kv_lens = count * ratio + lengths % ratio
    cu_q_lens = jnp.concatenate(
        (jnp.zeros((1,), jnp.int32), jnp.cumsum((lengths > 0).astype(jnp.int32)))
    )
    return pages, kv_lens, cu_q_lens, jnp.full((3,), batch, jnp.int32)


def _rpa_attention_batch(
    q,
    gate,
    blocks,
    lengths,
    table,
    cache,
    *,
    kv_heads,
    ratio,
    bkv_size=256,
    k_scale=1.0,
    v_scale=1.0,
):
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

    pages, kv_lens, cu_q_lens, distribution = virtual_page_metadata(
        blocks, lengths, table, ratio=ratio, page_size=cache.shape[1]
    )
    # Queries are packed in request order; zero-length requests have no query.
    order = jnp.argsort(lengths == 0, stable=True)
    query = jnp.pad(
        q[order].astype(jnp.float32),
        ((0, 0), (0, 0), (0, cache.shape[-1] - q.shape[-1])),
    )
    virtual_cache = cache.reshape(-1, ratio, *cache.shape[2:])
    dummy_k = jnp.zeros((q.shape[0], kv_heads, query.shape[-1]), cache.dtype)
    decode = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=bkv_size, batch_size=8)
    prefill = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=bkv_size, batch_size=1)
    out, next_cache = wrapper.ragged_paged_attention(
        query,
        dummy_k,
        dummy_k,
        virtual_cache,
        kv_lens,
        pages.reshape(-1),
        cu_q_lens,
        distribution,
        sm_scale=q.shape[-1] ** -0.5,
        q_scale=None,
        k_scale=k_scale,
        v_scale=v_scale,
        out_dtype=jnp.dtype(jnp.float32),
        skip_kv_update=True,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode,
        prefill_block_sizes=prefill,
    )
    out = out[jnp.minimum(cu_q_lens[:-1], len(q) - 1), :, : q.shape[-1]]
    out = bf16(
        bf16(out).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )
    out = jnp.where((lengths > 0)[:, None, None], out, 0)
    return out, next_cache.reshape(cache.shape)


def rpa_query_batch_size(page_columns):
    # RPA's minimum 128-step, batch8 schedule occupies about 815 KiB
    # of v7's 1 MiB SMEM. A 64-query, 513-column page table fits alongside
    # that schedule; keep larger tables within a 160 KiB scalar-data budget.
    max_queries = min(64, (160 * 1024) // (4 * page_columns))
    return 1 << (max_queries.bit_length() - 1)


def rpa_attention(
    q, gate, blocks, lengths, table, cache, *, kv_heads, ratio, k_scale=1.0, v_scale=1.0
):
    batch_size = rpa_query_batch_size(blocks.shape[1] + 1)
    attend = functools.partial(
        _rpa_attention_batch,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if len(q) <= batch_size:
        return attend(q, gate, blocks, lengths, table, cache)
    padding = (-len(q)) % batch_size

    def split(x):
        x = jnp.pad(x, ((0, padding),) + ((0, 0),) * (x.ndim - 1))
        return x.reshape(-1, batch_size, *x.shape[1:])

    def step(cache, inputs):
        query, gate, blocks, lengths, table = inputs
        output, cache = attend(query, gate, blocks, lengths, table, cache)
        return cache, output

    # Keep the small-step path fully unrolled, but bound the unroll for large
    # prefill steps. Fully expanding a 4K step repeats the RPA body 128 times
    # in each QSA layer and makes whole-model compilation impractical.
    num_chunks = (len(q) + batch_size - 1) // batch_size
    cache, output = jax.lax.scan(
        step,
        cache,
        tuple(map(split, (q, gate, blocks, lengths, table))),
        unroll=min(num_chunks, 4),
    )
    return output.reshape(-1, *q.shape[1:])[: len(q)], cache


def case(tokens: int, context: int, pattern: str) -> tuple[jax.Array, ...]:
    rng = np.random.default_rng(1964)
    page_size = 1664
    pages = (context + tokens + page_size - 1) // page_size
    query = jnp.asarray(rng.normal(0, 0.3, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, query.shape), jnp.bfloat16)
    cache = jnp.asarray(
        rng.normal(0, 0.3, (pages, page_size, 1, 4, 256)), jnp.float8_e4m3fn
    )
    lengths = context + np.arange(1, tokens + 1, dtype=np.int32)
    blocks = np.full((tokens, 512), -1, np.int32)
    for row, length in enumerate(lengths):
        complete = length // 4
        count = min(complete, 512)
        if pattern == "recent":
            selected = np.arange(complete - count, complete)
        else:
            selected = np.sort(rng.choice(complete, count, replace=False))
        blocks[row, :count] = selected
    table = np.broadcast_to(rng.permutation(pages), (tokens, pages)).copy()
    return (
        query,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table, jnp.int32),
        cache,
        jnp.array([0, tokens], jnp.int32),
    )


def baseline(query, gate, blocks, lengths, table, cache, starts):
    del starts
    return rpa_attention(
        query, gate, blocks, lengths, table, cache, kv_heads=2, ratio=4
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 8192])
    parser.add_argument("--contexts", type=int, nargs="+", default=[0, 8192])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["recent", "uniform"],
        default=["recent", "uniform"],
    )
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    assert args.iterations >= 3
    args.output.mkdir(parents=True, exist_ok=False)
    assert jax.default_backend() == "tpu"
    report = dict(
        device=str(jax.devices()[0]),
        versions={
            n: importlib.metadata.version(n) for n in ("jax", "jaxlib", "libtpu")
        },
        cases=[],
    )
    functions = {
        "rpa": jax.jit(baseline),
        "prefill": jax.jit(
            functools.partial(
                block_sparse_gqa_attention,
                kv_heads=2,
                ratio=4,
                query_block=256,
                kv_tile=256,
            )
        ),
    }
    for tokens in args.tokens:
        for context in args.contexts:
            for pattern in args.patterns:
                name = f"t{tokens}-c{context}-{pattern}"
                directory = args.output / name
                directory.mkdir()
                inputs = case(tokens, context, pattern)
                compiled = {}
                outputs = {}
                for arm, fn in functions.items():
                    compiled[arm] = fn.lower(*inputs).compile()
                    (directory / f"{arm}.hlo.txt").write_text(compiled[arm].as_text())
                    for _ in range(3):
                        outputs[arm] = jax.block_until_ready(compiled[arm](*inputs))[0]
                reference = np.asarray(outputs["rpa"], np.float32)
                actual = np.asarray(outputs["prefill"], np.float32)
                assert np.isfinite(reference).all() and np.isfinite(actual).all()
                relative_l2 = float(
                    np.linalg.norm(actual - reference) / np.linalg.norm(reference)
                )
                assert relative_l2 < 0.01, (name, relative_l2)
                samples = {arm: [] for arm in functions}
                for iteration in range(args.iterations):
                    order = (
                        list(functions)
                        if iteration % 2 == 0
                        else list(reversed(functions))
                    )
                    for arm in order:
                        start = time.perf_counter()
                        jax.block_until_ready(compiled[arm](*inputs))
                        samples[arm].append((time.perf_counter() - start) * 1000)
                with jax.profiler.trace(str(directory / "trace")):
                    for arm in functions:
                        with jax.profiler.TraceAnnotation(arm):
                            for _ in range(3):
                                jax.block_until_ready(compiled[arm](*inputs))
                medians = {
                    arm: statistics.median(values) for arm, values in samples.items()
                }
                row = dict(
                    tokens=tokens,
                    context=context,
                    pattern=pattern,
                    query_heads=24,
                    kv_heads=2,
                    head_dim=256,
                    ratio=4,
                    topk=512,
                    page_size=1664,
                    cache_dtype="float8_e4m3fn",
                    query_dtype="bfloat16",
                    relative_l2=relative_l2,
                    samples_ms=samples,
                    median_ms=medians,
                    speedup=medians["rpa"] / medians["prefill"],
                )
                report["cases"].append(row)
                (args.output / "results.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
