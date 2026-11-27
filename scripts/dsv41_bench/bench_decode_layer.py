# SPDX-License-Identifier: Apache-2.0
"""One DSv4.1 dense-indexer layer at the AgentX v7x-16 decode shape, per rank.

12 requests x 6 rows (1 + 5 DSpark drafts) = 72 query rows, 147,456 cached
tokens each (73,728 index keys, ratio 2), v7x TP1.
  unfused: the indexer op, the record mask, then the gather/attention op.
  fused: `_select_gather_attend`, the model's source-layer op.
Checks all five outputs bitwise against the unfused path and reports device us
per program, the top TC/SC ops and one program's TC/SC timeline
(VARIANT=unfused|fused runs one).
"""

import json
import os
import pathlib
import sys

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
B, ROWS = int(os.environ.get("REQUESTS", "12")), 6
T, TOPK, IPAGE, HEADS, BQ = B * ROWS, 512, 256, 64, 6
MPAGE, SPAGE, RATIO = 128, 64, 2  # decode_preparation.CompressorLayout
CONTEXT = 147456
SCORE_WIDTH = (CONTEXT + 1024) // RATIO  # max_model_len = context + output
rng = np.random.default_rng(1)

positions = (CONTEXT - ROWS + np.arange(ROWS, dtype=np.int32))[None, :].repeat(B, 0)
positions = positions.reshape(-1)
requests = np.repeat(np.arange(B, dtype=np.int32), ROWS)
starts = jnp.asarray(np.arange(B + 1, dtype=np.int32) * ROWS)
lengths = (positions + 1) // RATIO

# Index K: every request owns its own pages, scattered over the pool.
ipages = SCORE_WIDTH // IPAGE
layout = PackedFP8Layout(B * ipages, page_size=IPAGE)
keys = jax.random.normal(jax.random.key(3), (layout.pages * IPAGE, 128), jnp.bfloat16)
values, key_scales, _ = quantize_index_rows(keys)
bits = np.asarray(jax.lax.bitcast_convert_type(values, jnp.uint8)).astype(np.uint32)
bits = bits.reshape(-1, 32, 4)
words = bits[..., 0] | bits[..., 1] << 8 | bits[..., 2] << 16 | bits[..., 3] << 24
fields = np.concatenate(
    (words.view(np.int32), np.asarray(key_scales).view(np.int32)[:, None]), axis=1
)
records = np.arange(len(fields))
at = (
    (records // IPAGE * layout.page_stride * 128)[:, None]
    + np.arange(PAGE_ROWS)[None, :] * IPAGE
    + (records % IPAGE)[:, None]
)
pool = np.zeros(layout.pages * layout.page_stride * 128, np.int32)
pool[at] = fields
index_pool = jnp.asarray(pool.reshape(-1, 128))
index_table = rng.permutation(layout.pages).astype(np.int32).reshape(B, ipages)

# Main CSA2 KV: two banks, logical pages scattered over physical pages.
mpages = -(-(CONTEXT // RATIO) // MPAGE)
records_total = B * mpages * MPAGE
latent = jax.random.normal(jax.random.key(4), (records_total, 512), jnp.bfloat16)
kv_words, kv_scales = jax.jit(encode_csa2_kv)(latent * 0.2)
del latent
pages = rng.permutation(B * mpages).astype(np.int32).reshape(B, mpages)

# SWA word pool: finite BF16 halves; each request's window pages.
swa_pages = 4 * B + 1
halves = (rng.standard_normal((swa_pages * 2 * SPAGE, 128, 2)) * 0.2).astype(
    jnp.bfloat16
)
swa = jnp.asarray(halves.view(np.uint16).astype(np.uint32)).astype(jnp.uint32)
swa = jax.lax.bitcast_convert_type(swa[..., 0] | swa[..., 1] << 16, jnp.int32)
columns = CONTEXT // SPAGE + 2
swa_table = (np.arange(B * columns) % (swa_pages - 1) + 1).reshape(B, columns)
metadata, bounds, _ = prepare_swa_metadata(
    jnp.asarray(swa_table, jnp.int32),
    starts,
    jnp.asarray(positions),
    jnp.zeros((B,), jnp.int32),
    page_size=SPAGE,
    page_stride=SPAGE,
    row_offset=0,
)
# The inverse-rotation rows are gathered outside the program: gathered inside
# from a [148480,64] parameter, XLA re-lays the table out on SparseCore per
# call (~30 us SparseCore, ~22 us TensorCore), which the model does not pay.
angles = rng.standard_normal((CONTEXT + 1024, 32)).astype(np.float32)
rope_rows = csa2_adapter._inverse_rope(
    jnp.asarray(np.concatenate((np.cos(angles), np.sin(angles)), axis=1))[positions]
)
csa2_adapter._inverse_rope = lambda rows: rows
half = records_total // 2

args = dict(
    index_q=jnp.asarray(rng.normal(size=(T, 32, 128)), jnp.float8_e4m3fn),
    index_banks=(index_pool,),
    index_table=jnp.asarray(index_table),
    weights=jnp.asarray(rng.uniform(0.01, 0.1, (T, 32)), jnp.float32),
    index_lengths=jnp.asarray(lengths),
    candidates=jnp.empty((T, 0), jnp.int32),
    requests=jnp.asarray(requests),
    batch_max=jnp.asarray(lengths.max()).reshape(1),
    index_starts=starts,
    lengths=jnp.asarray(lengths)[:, None],
    swa_valid=jnp.ones((T,), bool),
    q=jnp.asarray(rng.standard_normal((T, HEADS, 512)) * 0.2, jnp.bfloat16),
    raw=kv_words[:half],
    high=kv_words[half:],
    scales=kv_scales[:half],
    high_scales=kv_scales[half:],
    pages=jnp.asarray(pages),
    main_starts=starts,
    swa=swa,
    swa_high=None,
    starts=starts,
    positions=jnp.asarray(positions),
    sink=jnp.linspace(-1, 1, HEADS, dtype=jnp.float32),
    cos_sin=rope_rows,
    metadata=metadata,
    bounds=bounds,
)
layout_args = (layout.pages, layout.page_stride, layout.page_offset, IPAGE)
statics = (
    layout_args,
    TOPK,
    8,
    SCORE_WIDTH,
    MPAGE,
    SPAGE,
    RATIO,
    512**-0.5,
    False,
    BQ,
)


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
        a["cos_sin"],
        a["metadata"],
        a["bounds"],
        *statics[4:],
    )
    return output, selected, gathered, gathered_scales, counts


ORDER = list(args)  # jit rebuilds dicts with sorted keys


def fused(a):
    return csa2_adapter._select_gather_attend(*(a[k] for k in ORDER), *statics)


def device_us(fn, name, reps=20):
    for _ in range(3):
        jax.block_until_ready(fn(args))
    d = OUT / name
    with jax.profiler.trace(str(d)):
        for _ in range(reps):
            r = fn(args)
        jax.block_until_ready(r)
    spans, ops, events = [], {}, []
    for p in d.rglob("*.xplane.pb"):
        for plane in ProfileData.from_file(str(p)).planes:
            if not plane.name.startswith("/device:TPU:0"):
                continue
            sc = "SparseCore" in plane.name
            for line in plane.lines:
                for e in line.events:
                    if line.name == "XLA Modules" and not sc:
                        spans.append((e.start_ns, e.duration_ns))
                    if (not sc and line.name == "XLA Ops") or (
                        sc and line.name == "Sparse Core Ops"
                    ):
                        key = ("SC " if sc else "TC ") + e.name.split(" = ")[0].split(
                            "."
                        )[0][:50]
                        ops[key] = ops.get(key, 0.0) + e.duration_ns / 1e3 / reps
                        events.append((e.start_ns, e.duration_ns, key))
    top = sorted(ops.items(), key=lambda kv: -kv[1])[:16]
    # One program's timeline (TC ops and SparseCore ops >= 1 us), in us.
    start, length = sorted(spans)[len(spans) // 2]
    timeline = [
        (round((s - start) / 1e3, 1), round(n / 1e3, 1), k)
        for s, n, k in sorted(events)
        if start <= s < start + length and n >= 1000
    ]
    median = float(np.median([n for _, n in spans])) / 1e3
    return round(median, 2), [(k, round(v, 1)) for k, v in top], timeline


variants = {"unfused": unfused, "fused": fused}
names = list(variants) if VARIANT == "both" else [VARIANT]
results, report = {}, {}
for name in names:
    compiled = jax.jit(variants[name]).lower(args).compile()
    results[name] = jax.block_until_ready(compiled(args))
    us, top, timeline = device_us(compiled, name)
    report[name] = dict(device_us=us, top_ops_us=top)
    print(json.dumps({name: report[name]}), flush=True)
    for event in timeline:
        print(name, *event, flush=True)
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
(OUT / f"decode_layer_{VARIANT}.json").write_text(json.dumps(report, indent=1))
sys.exit(0 if all(v for k, v in report.items() if k.startswith("bitwise")) else 3)
