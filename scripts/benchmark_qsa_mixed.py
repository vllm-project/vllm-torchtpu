# SPDX-License-Identifier: Apache-2.0
"""QSA request-type dispatch A/B on matched inputs and padded token buckets.

Includes index metadata, attention, dispatch, shifts and output restoration;
excludes cache writes, projections, compilation and warmup. Timings include
host dispatch and synchronization. No full-model performance claim.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing
from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import select_blocks
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import (
    _prefill_attention,
    pack_rows,
    qsa_attention,
    rpa_attention,
)


def prefill_index_baseline(query, pool, lengths, table, req, starts, context):
    if len(query) <= len(context):
        return select_blocks(
            query, pool, lengths, table, req, starts, context, ratio=4, block_topk=512
        )
    # Prefill branch from pooled_qsa.select_blocks at 03acd2980.
    width = pool.shape[-1]
    physical_pages = np.prod(pool.shape[1:]) * pool.dtype.itemsize // (32 * width * 2)
    expanded = table[..., None] * physical_pages + jnp.arange(pool.shape[1] // 128)
    count = jnp.max(jnp.where(jnp.diff(starts) > 0, jnp.arange(len(context)) + 1, 0))
    return streamindex_topk(
        jnp.pad(query, ((0, 0), (0, 0), (0, width - query.shape[-1]))),
        jnp.full(query.shape[:2], query.shape[-1] ** -0.5, jnp.float32),
        pool.reshape(-1, 16, 4, width),
        context + jnp.diff(starts),
        expanded.reshape(-1),
        starts,
        jnp.asarray([0, 0, count], jnp.int32),
        k=512,
        compression_ratio=4,
        num_kv_pages_per_block=(8, 8, 8),
        num_queries_per_block=(1, 128, 128),
        decode_req_batch_size=8,
        enable_early_exit=True,
        chunk_tokens=512 if len(query) % 512 == 0 else None,
        components=qsa_indexing(3),
    )


def attention_baseline(query, gate, blocks, lengths, table, cache, starts):
    if len(query) <= len(starts) - 1:
        return rpa_attention(
            query, gate, blocks, lengths, table, cache, kv_heads=2, ratio=4
        )
    return _prefill_attention(
        query, gate, blocks, lengths, table, cache, starts, kv_heads=2, ratio=4
    )


def make_inputs(decodes, prefill_lengths, tokens, fp8):
    rng = np.random.default_rng(2055)
    request_lengths = [1] * decodes + prefill_lengths
    capacity = max(8, 1 << (len(request_lengths) - 1).bit_length())
    request_lengths += [0] * (capacity - len(request_lengths))
    starts = np.r_[0, np.cumsum(request_lengths)].astype(np.int32)
    assert starts[-1] <= tokens and tokens >= capacity
    context = np.full(capacity, 2303, np.int32)
    context[np.asarray(request_lengths) == 0] = 0
    req = np.minimum(
        np.searchsorted(starts[1:], np.arange(tokens), side="right"), capacity - 1
    )
    lengths = np.where(
        np.arange(tokens) < starts[-1],
        context[req] + np.arange(tokens) - starts[req] + 1,
        0,
    ).astype(np.int32)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.25, shape), jnp.bfloat16)

    query, gate, iq = (
        rand((tokens, 24, 256)),
        rand((tokens, 24, 256)),
        rand((tokens, 4, 128)),
    )
    pages_per_request = 32
    pages = capacity * pages_per_request
    table = jnp.asarray(
        rng.permutation(pages).reshape(capacity, pages_per_request), jnp.int32
    )
    key, value = rand((pages * 128, 2, 256)), rand((pages * 128, 2, 256))
    cache = pack_rows(key, value, fp8=fp8).reshape(pages, 128, -1, 4 if fp8 else 2, 256)
    index_pages = capacity * 2 + 1
    keys = np.asarray(rand((index_pages, 208, 2, 128))).view(np.uint16)
    raw = np.zeros((index_pages, 1664, 1, 4, 256), np.uint8)
    raw[:, :208, 0, 0::2, :128] = keys & 255
    raw[:, :208, 0, 1::2, :128] = keys >> 8
    pool = jax.device_put(raw.view(np.dtype(jnp.float8_e4m3fn)))
    index_table = jnp.asarray(
        rng.permutation(np.arange(1, index_pages)).reshape(capacity, 2), jnp.int32
    )
    index = (
        iq,
        pool,
        jnp.asarray(lengths),
        index_table,
        jnp.asarray(req),
        jnp.asarray(starts),
        jnp.asarray(context),
    )
    return index, (
        query,
        gate,
        jnp.asarray(lengths),
        table[req],
        cache,
        jnp.asarray(starts),
    )


def measure_pair(functions, args, trace_dir, rounds=5, repeats=20):
    compiled = [jax.jit(fn).lower(*args).compile() for fn in functions]
    outputs = [jax.block_until_ready(fn(*args)) for fn in compiled]
    for fn in compiled:
        for _ in range(3):
            jax.block_until_ready(fn(*args))
    times = [[], []]
    for round_id in range(rounds):
        for idx in [0, 1, 1, 0] if round_id % 2 == 0 else [1, 0, 0, 1]:
            start = time.perf_counter_ns()
            for _ in range(repeats):
                jax.block_until_ready(compiled[idx](*args))
            times[idx].append((time.perf_counter_ns() - start) / repeats / 1e3)
    with jax.profiler.trace(str(trace_dir)):
        for idx, label in enumerate(("baseline", "split")):
            with jax.profiler.TraceAnnotation(label):
                for _ in range(3):
                    jax.block_until_ready(compiled[idx](*args))
    return outputs, dict(
        baseline_us=times[0],
        split_us=times[1],
        median_us=[float(np.median(x)) for x in times],
    )


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    assert jax.default_backend() == "tpu"
    cases = [
        (8, [128], 256, True),
        (32, [512], 1024, True),
        (8, [128, 129], 512, False),
        (0, [128, 129], 512, True),
        (64, [], 64, True),
    ]
    records = []
    for case_id, (decodes, prefills, tokens, fp8) in enumerate(cases):
        index, attention = make_inputs(decodes, prefills, tokens, fp8)
        before_index = np.asarray(index[1]).view(np.uint8).copy()
        before_cache = np.asarray(attention[4]).view(np.uint8).copy()
        ids, index_times = measure_pair(
            [prefill_index_baseline, partial(select_blocks, ratio=4, block_topk=512)],
            index,
            output / f"case-{case_id}-index-trace",
        )
        np.testing.assert_array_equal(np.asarray(ids[0]), np.asarray(ids[1]))
        query, gate, lengths, table, cache, starts = attention
        args = (query, gate, ids[0], lengths, table, cache, starts)
        results, attention_times = measure_pair(
            [
                attention_baseline,
                partial(qsa_attention, kv_heads=2, ratio=4),
            ],
            args,
            output / f"case-{case_id}-attention-trace",
        )
        baseline, split = [np.asarray(x[0], np.float32) for x in results]
        assert np.isfinite(baseline).all() and np.isfinite(split).all()
        norm = np.linalg.norm((split - baseline).reshape(tokens, -1), axis=1)
        den = np.maximum(np.linalg.norm(baseline.reshape(tokens, -1), axis=1), 1e-6)
        relative = norm / den
        assert float(relative.max()) < 0.01, float(relative.max())
        np.testing.assert_array_equal(np.asarray(index[1]).view(np.uint8), before_index)
        for result in results:
            np.testing.assert_array_equal(
                np.asarray(result[1]).view(np.uint8), before_cache
            )
        record = dict(
            decodes=decodes,
            prefills=prefills,
            tokens=tokens,
            fp8=fp8,
            index=index_times,
            attention=attention_times,
            max_row_relative_l2=float(relative.max()),
            selected_ids_exact=True,
            cache_bytes_unchanged=True,
        )
        records.append(record)
        (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
