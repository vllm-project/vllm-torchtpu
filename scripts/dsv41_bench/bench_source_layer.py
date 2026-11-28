# SPDX-License-Identifier: Apache-2.0
"""One DSv4.1 source layer at the AgentX prefill shape: selection, KV gather, attention.

2,048 query rows of one 147K-token request (73,984 index keys, ratio 2), v7x TP1.
  unfused: the indexer op, the torch-side record mask, then the gather/attention op.
  fused: `_select_gather_attend`, each selection tile gathered and attended in place.
Checks all five outputs bitwise against the unfused path and reports device ms
per program, the top TC/SC ops, and trace+lower / compile seconds
(VARIANT=unfused|fused measures one in a fresh process; default runs both).
"""

import json
import os
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.profiler import ProfileData

jax.config.update("jax_enable_compilation_cache", False)

from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import (  # noqa: E402
    encode_csa2_kv,
    prepare_swa_metadata,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (  # noqa: E402
    PAGE_ROWS,
    PackedFP8Layout,
    quantize_index_rows,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter  # noqa: E402
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_fp8 import (  # noqa: E402
    select_fp8_indexer_jax,
)

OUT = pathlib.Path(os.environ.get("OUT", "."))
VARIANT = os.environ.get("VARIANT", "both")
T, TOPK, IPAGE, SCORE_WIDTH, HEADS = 2048, 512, 256, 524288, 64
MPAGE, SPAGE, RATIO = 64, 64, 2
PREFIX = 131072 + 14336
rng = np.random.default_rng(1)


def pack(keys, layout):
    values, scales, _ = quantize_index_rows(keys)
    bits = (
        np.asarray(jax.lax.bitcast_convert_type(values, jnp.uint8))
        .astype(np.uint32)
        .reshape(-1, 32, 4)
    )
    words = bits[..., 0] | bits[..., 1] << 8 | bits[..., 2] << 16 | bits[..., 3] << 24
    fields = np.concatenate(
        (words.view(np.int32), np.asarray(scales).view(np.int32)[:, None]), axis=1
    )
    records = np.arange(len(keys))
    at = (
        (records // IPAGE * layout.page_stride * 128)[:, None]
        + np.arange(PAGE_ROWS)[None, :] * IPAGE
        + (records % IPAGE)[:, None]
    )
    pool = np.zeros(layout.pages * layout.page_stride * 128, np.int32)
    pool[at] = fields
    return jnp.asarray(pool.reshape(-1, 128))


positions = PREFIX + np.arange(T, dtype=np.int32)
keys_total = (PREFIX + T) // RATIO + 256
ipages = keys_total // IPAGE + 1
layout = PackedFP8Layout(ipages, page_size=IPAGE)
index_pool = pack(
    jax.random.normal(jax.random.key(3), (layout.pages * IPAGE, 128), jnp.bfloat16),
    layout,
)
index_table = np.zeros((1, SCORE_WIDTH // IPAGE), np.int32)
index_table[0, :ipages] = np.arange(ipages)
lengths = (positions + 1) // RATIO

# Main CSA2 KV: two banks, logical pages scattered over physical pages.
records_total = 81920
words, scales = encode_csa2_kv(
    jnp.asarray(rng.standard_normal((records_total, 512)) * 0.2, jnp.bfloat16)
)
# The server's table: 8 request slots (MIN_NUM_SEQS) x 4,096 pages, request 0 live.
pages = np.zeros((8, 4096), np.int32)
pages[0, : records_total // MPAGE] = rng.permutation(records_total // MPAGE)

# SWA word pool: finite BF16 halves; one request's window pages.
swa_pages = 64
halves = (rng.standard_normal((swa_pages * 2 * SPAGE, 128, 2)) * 0.2).astype(
    jnp.bfloat16
)
swa = jnp.asarray(halves.view(np.uint16).astype(np.uint32)).astype(jnp.uint32)
swa = jax.lax.bitcast_convert_type(swa[..., 0] | swa[..., 1] << 16, jnp.int32)
swa_table = (np.arange((PREFIX + T) // SPAGE + 2) % (swa_pages - 1) + 1)[None, :]
starts = jnp.asarray([0, T], jnp.int32)
metadata, bounds, _ = prepare_swa_metadata(
    jnp.asarray(swa_table, jnp.int32),
    starts,
    jnp.asarray(positions),
    jnp.zeros((1,), jnp.int32),
    page_size=SPAGE,
    page_stride=SPAGE,
    row_offset=0,
)
angles = rng.standard_normal((PREFIX + T, 32)).astype(np.float32)
rope_rows = jnp.asarray(np.concatenate((np.cos(angles), np.sin(angles)), axis=1))[
    positions
]

args = dict(
    index_q=jnp.asarray(rng.normal(size=(T, 32, 128)), jnp.float8_e4m3fn),
    index_banks=(index_pool,),
    index_table=jnp.asarray(index_table),
    weights=jnp.asarray(rng.uniform(0.01, 0.1, (T, 32)), jnp.float32),
    index_lengths=jnp.asarray(lengths),
    candidates=jnp.empty((T, 0), jnp.int32),
    requests=jnp.zeros((T,), jnp.int32),
    batch_max=jnp.asarray(lengths.max()).reshape(1),
    index_starts=starts,
    lengths=jnp.asarray(lengths)[:, None],
    swa_valid=jnp.ones((T,), bool),
    q=jnp.asarray(rng.standard_normal((T, HEADS, 512)) * 0.2, jnp.bfloat16),
    raw=words[: records_total // 2],
    high=words[records_total // 2 :],
    scales=scales[: records_total // 2],
    high_scales=scales[records_total // 2 :],
    pages=jnp.asarray(pages),
    main_starts=starts,
    swa=swa,
    swa_high=None,
    starts=starts,
    positions=jnp.asarray(positions),
    sink=jnp.linspace(-1, 1, HEADS, dtype=jnp.float32),
    rope_rows=rope_rows,
    metadata=metadata,
    bounds=bounds,
)
layout_args = (layout.pages, layout.page_stride, layout.page_offset, IPAGE)
statics = (layout_args, TOPK, 8, SCORE_WIDTH, MPAGE, SPAGE, RATIO, 512**-0.5, False, 16)


def unfused(a):
    selected, _ = select_fp8_indexer_jax(
        a["index_q"],
        a["index_banks"],
        a["index_table"],
        a["weights"],
        a["index_lengths"],
        a["candidates"],
        a["requests"],
        a["batch_max"],
        a["index_starts"],
        layout_args,
        TOPK,
        8,
        False,
        False,
        True,
        SCORE_WIDTH,
    )
    valid = (selected >= 0) & (selected < a["lengths"]) & a["swa_valid"][:, None]
    selected = jnp.where(valid, selected, -1)
    output, gathered, gathered_scales, counts = csa2_adapter._gather_attend(
        a["q"],
        a["raw"],
        a["high"],
        a["scales"],
        a["high_scales"],
        selected,
        a["pages"],
        a["main_starts"],
        a["swa"],
        a["swa_high"],
        a["starts"],
        a["positions"],
        a["sink"],
        a["rope_rows"],
        a["metadata"],
        a["bounds"],
        MPAGE,
        SPAGE,
        RATIO,
        512**-0.5,
        False,
        16,
    )
    return output, selected, gathered, gathered_scales, counts


ORDER = list(args)  # jit rebuilds dicts with sorted keys


def fused(a):
    return csa2_adapter._select_gather_attend(*(a[k] for k in ORDER), *statics)


def device_ms(fn, name, reps=10):
    for _ in range(3):
        jax.block_until_ready(fn(args))
    d = OUT / name
    with jax.profiler.trace(str(d)):
        for _ in range(reps):
            r = fn(args)
        jax.block_until_ready(r)
    spans, ops = [], {}
    for p in d.rglob("*.xplane.pb"):
        for plane in ProfileData.from_file(str(p)).planes:
            if not plane.name.startswith("/device:TPU:0"):
                continue
            sc = "SparseCore" in plane.name
            for line in plane.lines:
                for e in line.events:
                    if line.name == "XLA Modules" and not sc:
                        spans.append(e.duration_ns / 1e6)
                    if (not sc and line.name == "XLA Ops") or (
                        sc and line.name in ("Sparse Core Ops", "TEC 0")
                    ):
                        key = ("SC " if sc else "TC ") + e.name.split(" = ")[0].split(
                            "."
                        )[0][:50]
                        ops[key] = ops.get(key, 0.0) + e.duration_ns / 1e3 / reps
    top = sorted(ops.items(), key=lambda kv: -kv[1])[:14]
    return round(float(np.median(spans)), 3), [(k, round(v, 1)) for k, v in top]


def build(fn):
    t0 = time.perf_counter()
    lowered = jax.jit(fn).lower(args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    return compiled, round(t1 - t0, 1), round(t2 - t1, 1)


variants = {"unfused": unfused, "fused": fused}
names = list(variants) if VARIANT == "both" else [VARIANT]
results, report = {}, {}
for name in names:
    compiled, lower_s, compile_s = build(variants[name])
    results[name] = jax.block_until_ready(compiled(args))
    ms, top = device_ms(compiled, name)
    report[name] = dict(
        device_ms=ms, lower_s=lower_s, compile_s=compile_s, top_ops_us=top
    )
    print(json.dumps({name: report[name]}), flush=True)
if len(results) == 2:
    labels = ("output", "records", "words", "scales", "counts")
    for label, a, b in zip(labels, results["unfused"], results["fused"]):
        a, b = np.asarray(a), np.asarray(b)
        same = a.shape == b.shape and np.array_equal(
            a.view(np.uint8) if a.dtype != bool else a,
            b.view(np.uint8) if b.dtype != bool else b,
        )
        report[f"bitwise_{label}"] = bool(same)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("bitwise")}))
(OUT / f"source_layer_{VARIANT}.json").write_text(json.dumps(report, indent=1))
sys.exit(0 if all(v for k, v in report.items() if k.startswith("bitwise")) else 3)
