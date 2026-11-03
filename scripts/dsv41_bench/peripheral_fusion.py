# SPDX-License-Identifier: Apache-2.0
"""Standalone peripheral-region A/B; no production dispatch changes.

Native JAX baselines are imported or AST-extracted without mathematical edits
from this checkout. Engram alone is an explicitly labelled JAX mathematical
replay of the Torch forward tail, not the torch-tpu production executable.
"""

import argparse
import ast
import functools
import hashlib
import importlib.metadata
import json
import math
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import peripheral_references
import region_fusion as rf
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing import compress_rope, rope_gather
from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import prepare_main_records
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor


def extract_functions(relative, names, extra=None):
    """Load unmodified JAX helpers without importing the Torch model stack."""
    path = rf.ROOT / "src/vllm_torchtpu" / relative
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names)
    namespace = {"jax": jax, "jnp": jnp, "functools": functools, **(extra or {})}
    module = ast.Module(body=body, type_ignores=[])
    # Deferred annotations avoid importing Torch solely for type annotations.
    exec(compile(ast.unparse(module), str(path), "exec", flags=0x1000000), namespace)
    return namespace


CACHE = extract_functions(
    "models/vllm/deepseek_v41/v2/cache.py",
    [
        "_cache_num_rows",
        "_gather_raw_rows",
        "_gather_words",
        "_valid_records",
        "_gather_records",
        "_split_words",
        "gather_cache_pieces",
    ],
    {"ROW_WORDS": 128},
)
MHC = extract_functions(
    "models/vllm/deepseek_v41/common/mhc.py",
    [
        "_pre_gates",
        "delayed_pre_jax",
        "_truncate_bf16",
        "_pack_mix_weight",
        "_packed_mixes",
        "_flat_pre_sums",
        "_collapse_flat",
        "_delayed_pre_flat_small",
        "delayed_pre_flat_jax",
    ],
)

GATE = extract_functions(
    "kernels/deepseek_v41/mhc/shifted_post_pre_rms.py",
    ["_pre_gates"],
    {"math": math, "pl": pl, "pltpu": pltpu},
)
# Preserve the production collapse reduction tree when extending T=288.
_flat_tree = ast.parse(
    (rf.ROOT / "src/vllm_torchtpu/models/vllm/deepseek_v41/common/mhc.py").read_text()
)
_flat_node = next(
    n
    for n in _flat_tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_flat_pre_sums"
)
_flat_source = ast.unparse(_flat_node).replace(
    "functools.reduce(jnp.add, terms)",
    "((terms[0] + terms[2]) + (terms[1] + terms[3]))",
)
FLAT_CANDIDATE = dict(MHC)
exec(
    compile(_flat_source, "flat_pre_balanced", "exec", flags=0x1000000), FLAT_CANDIDATE
)


def quant_baseline(x):
    return quantize_tensor(x, jnp.float8_e4m3fn)


def quant_tiled(xs, tile=32):
    (x,) = xs
    t, d = x.shape

    def kernel(xr, qr, sr):
        q, s = quant_baseline(xr[...])
        qr[...], sr[...] = q, s

    return pl.pallas_call(
        kernel,
        grid=(t // tile,),
        in_specs=[pl.BlockSpec((tile, d), lambda i: (i, 0))],
        out_specs=(
            pl.BlockSpec((tile, d), lambda i: (i, 0)),
            pl.BlockSpec((tile, 1), lambda i: (i, 0)),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((t, 1), jnp.float32),
        ),
        name=f"dsv41_quantize_d{d}_t{tile}",
    )


def quant_candidate(xs):
    def body(x):
        q, scale = quant_baseline(x)
        return q, scale.T

    call = rf.fuse_function(body, xs, name="dsv41_quantize_full")

    def wrapped(x):
        q, scale = call(x)
        return q, scale.T

    return wrapped


def compact_baseline(records):
    return compress_rope._compact(records, 32768, records.size // 32, 32)


def compact_math(records):
    n = records.size
    r = records.reshape(1, n)
    valid = (r >= 0) & (r < 32768)
    rank = rf.prefix_sum(valid.astype(jnp.int32), axis=1) - 1
    out = jnp.arange(n, dtype=jnp.int32)[:, None]
    match = valid & (rank == out)
    token = jnp.sum(jnp.where(match, jnp.arange(n)[None, :], 0), axis=1)
    record = jnp.sum(jnp.where(match, r, 0), axis=1)
    pairs = jnp.stack((token, record), axis=1)
    boundary = (jnp.arange(n // 32 + 1) * 32)[:, None]
    offsets = jnp.sum(
        valid & (jnp.arange(n)[None, :] < boundary), axis=1, dtype=jnp.int32
    )
    return pairs, offsets


def compact_candidate(xs):
    call = rf.fuse_function(compact_math, xs, name="dsv41_compact_record_plan")

    def wrapped(records):
        pairs, offsets = call(records)
        return pairs.reshape(-1), offsets

    return wrapped


def cache_baseline(low, high, records):
    return CACHE["gather_cache_pieces"]((low, high), records, 512)


def unpack_candidate(n, tile=128):
    def kernel(wr, out):
        a, b = CACHE["_split_words"](wr[...])
        out[0, ...], out[1, ...] = a, b

    return pl.pallas_call(
        kernel,
        grid=(2, n // tile),
        in_specs=[pl.BlockSpec((1, tile, 128), lambda c, i: (c, i, 0))],
        out_specs=pl.BlockSpec((2, 1, tile, 128), lambda c, i: (c, 0, i, 0)),
        out_shape=jax.ShapeDtypeStruct((4, 1, n, 128), jnp.bfloat16),
        name="dsv41_cache_word_unpack",
    )


def cache_candidate(xs):
    n = xs[2].size
    tile = max(v for v in range(128, 4097, 128) if n % v == 0)
    unpack = unpack_candidate(n, tile)

    def call(low, high, records):
        cache = (low, high)
        ids = CACHE["_valid_records"](cache, records, 256) * 2
        rows = (ids[None] + jnp.arange(2)[:, None]).reshape(-1)
        words = CACHE["_gather_raw_rows"](cache, rows).reshape(2, n, 128)
        return unpack(words).reshape(4, n, 128)

    return call


def rope_candidate(xs):
    table, positions = xs
    n, width = positions.size, table.shape[1]

    def kernel(pos, src, dst, buf, sem):
        i = pl.program_id(0) * 8
        copies = []
        for j in range(8):
            row = jnp.clip(pos[i + j], 0, src.shape[0] - 1)
            copy = pltpu.make_async_copy(
                src.at[pl.ds(row, 1), :], buf.at[pl.ds(j, 1), :], sem.at[j]
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        dst[...] = buf[...]

    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(n // 8,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec((8, width), lambda i, p: (i, 0)),
            scratch_shapes=[
                pltpu.VMEM((8, width), jnp.float32),
                pltpu.SemaphoreType.DMA((8,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((n, width), jnp.float32),
        name="dsv41_rope_row_gather",
    )
    return lambda a, p: call(p, a)


def main_records_baseline(selected, pages, starts, *, page_size=256):
    return prepare_main_records(selected, pages, starts, page_size=page_size)


def main_records_window_candidate(xs, *, page_size=256):
    selected, pages, starts = xs
    t, k = selected.shape
    requests, pps = pages.shape
    padded_pps = (pps + 127) // 128 * 128

    def kernel(window, sel_ref, valid_ref, out_ref, counts_ref):
        sel = sel_ref[...]
        valid = (sel >= 0) & valid_ref[...]
        logical = jnp.maximum(sel, 0)
        page = jnp.clip(logical // page_size, 0, pps - 1)
        physical = jnp.zeros_like(page)
        for c in range(padded_pps // 128):
            local = page - c * 128
            chunk = window[:, c * 128 : (c + 1) * 128]
            value = rf.take_along(chunk, jnp.clip(local, 0, 127), axis=1)
            physical = jnp.where((local >= 0) & (local < 128), value, physical)
        out_ref[...] = jnp.where(valid, physical * page_size + logical % page_size, -1)
        counts_ref[...] = jnp.sum(valid, axis=1, keepdims=True, dtype=jnp.int32)

    call = pl.pallas_call(
        kernel,
        grid=(t // 8,),
        in_specs=[
            pl.BlockSpec((8, padded_pps), lambda i: (i, 0)),
            pl.BlockSpec((8, k), lambda i: (i, 0)),
            pl.BlockSpec((8, 1), lambda i: (i, 0)),
        ],
        out_specs=(
            pl.BlockSpec((8, k), lambda i: (i, 0)),
            pl.BlockSpec((8, 1), lambda i: (i, 0)),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((t, k), jnp.int32),
            jax.ShapeDtypeStruct((t, 1), jnp.int32),
        ),
        name="dsv41_main_record_lookup",
    )

    def wrapped(sel, table, boundaries):
        from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import _requests

        req = _requests(boundaries, t)
        table = jnp.pad(table, ((0, 0), (0, padded_pps - pps)))
        window = table[req]
        out, counts = call(window, sel, (jnp.arange(t) < boundaries[-1])[:, None])
        return out, counts.reshape(t)

    return wrapped


def main_records_candidate(xs, *, page_size=256):
    selected, pages, starts = xs
    t, k = selected.shape
    requests, pps = pages.shape
    padded_pps = (pps + 127) // 128 * 128

    def body(sel, table, boundaries):
        token = jnp.arange(t, dtype=jnp.int32)
        req = jnp.sum(token[:, None] >= boundaries[None, 1:], axis=1)
        req = jnp.minimum(req, requests - 1)
        onehot = (req[:, None] == jnp.arange(requests)[None, :]).astype(jnp.bfloat16)
        table = jnp.pad(table, ((0, 0), (0, padded_pps - pps)))
        window = jnp.zeros((t, padded_pps), jnp.int32)
        # Four exact byte limbs preserve every INT32 page id, including -1.
        # 4*2*T*requests*padded_pps = 56.6 MFLOP for this shape.
        for shift in (0, 8, 16, 24):
            limb = ((table >> shift) & 255).astype(jnp.bfloat16)
            gathered = jnp.dot(onehot, limb, preferred_element_type=jnp.float32).astype(
                jnp.int32
            )
            window = window | (gathered << shift)
        valid = (sel >= 0) & (token[:, None] < boundaries[-1])
        logical = jnp.maximum(sel, 0)
        page = jnp.clip(logical // page_size, 0, pps - 1)
        physical = jnp.zeros_like(page)
        for c in range(padded_pps // 128):
            local = page - c * 128
            chunk = window[:, c * 128 : (c + 1) * 128]
            value = rf.take_along(chunk, jnp.clip(local, 0, 127), axis=1)
            physical = jnp.where((local >= 0) & (local < 128), value, physical)
        records = jnp.where(valid, physical * page_size + logical % page_size, -1)
        return records, jnp.sum(valid, axis=1, keepdims=True, dtype=jnp.int32).T

    fused = rf.fuse_function(body, xs, name="dsv41_main_records_matrix")

    def call(*xs):
        records, counts = fused(*xs)
        return records, counts.reshape(t)

    return call


MHC_CONSTANTS = (1e-6, 1e-6, 1e-6, 2.0, 20)


def mhc_baseline(flat, fn, scale, base, pre):
    return MHC["delayed_pre_flat_jax"](flat, fn, scale, base, pre, *MHC_CONSTANTS)


def mhc_candidate(xs):
    flat, fn, scale, base, pre = xs
    t, width = flat.shape
    m, d = pre.shape[1], width // pre.shape[1]
    mix_count = m * (m + 2)

    def call(flat, fn, scale, base, pre):
        mix, square, value = FLAT_CANDIDATE["_flat_pre_sums"](
            flat, fn, pre, mix_count, 96
        )
        pad = (-t) % 128
        new_pre, post, comb = GATE["_pre_gates"](
            jnp.pad(mix.T, ((0, 0), (0, pad))),
            jnp.pad(square.T, ((0, 0), (0, pad))),
            scale,
            base,
            hidden_size=d,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
        )
        return post[:t, :, None], comb[:t], value, new_pre[:t]

    return call


def engram_baseline(hidden, kv, qw, kw, mask):
    """JAX math replay of common/engram.py forward tail, lines 969-984."""
    t, m, d = hidden.shape
    keys = kv[:, : m * d].reshape(t, m, d).astype(jnp.float32)
    values = kv[:, m * d :].astype(jnp.float32)
    h = hidden.astype(jnp.float32)
    dot = jnp.sum(h * qw * kw * keys, axis=-1)
    dot *= lax.rsqrt(jnp.mean(h * h, axis=-1) + 1e-6)
    dot *= lax.rsqrt(jnp.mean(keys * keys, axis=-1) + 1e-6) * d**-0.5
    gate_input = jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6))
    gate = jax.nn.sigmoid(jnp.where(dot < 0, -gate_input, gate_input))
    gate *= mask[:, None]
    return (h + gate[:, :, None] * values[:, None, :]).astype(hidden.dtype)


def engram_candidate(xs, tile=32):
    h, kv, qw, kw, mask = xs
    t, m, d = h.shape

    def kernel(hr, kr, qr, wr, mr, out):
        out[...] = engram_baseline(hr[...], kr[...], qr[...], wr[...], mr[:, 0])

    call = pl.pallas_call(
        kernel,
        grid=(t // tile,),
        in_specs=[
            pl.BlockSpec((tile, m, d), lambda i: (i, 0, 0)),
            pl.BlockSpec((tile, (m + 1) * d), lambda i: (i, 0)),
            pl.BlockSpec(qw.shape, lambda i: (0, 0)),
            pl.BlockSpec(kw.shape, lambda i: (0, 0)),
            pl.BlockSpec((tile, 1), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tile, m, d), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct(h.shape, h.dtype),
        name="dsv41_engram_gate_residual",
    )
    return lambda h, k, q, w, m: call(h, k, q, w, m[:, None])


def router_baseline(scores, gate_l, topk_idx, topk_weights, row_scale_l):
    from vllm_torchtpu.kernels.experimental.push_fused_moe import layer
    from vllm_torchtpu.kernels.fused_moe.v2.layer import _row_routes

    row_routes = _row_routes(scores, gate_l)
    denom = jnp.maximum(topk_weights.sum(axis=-1, keepdims=True), 1e-20)
    weights = topk_weights / jnp.where(row_routes, denom, 1.0)
    weights = jnp.where(row_routes, weights, 0.0)
    idx = jnp.where(row_routes, topk_idx, jnp.int32(384))
    blocks = idx.reshape(27, 64)
    hist = rf.host.pair_block_hist(blocks, 384)
    bits = lax.bitcast_convert_type(row_scale_l[:, 0].astype(jnp.float32), jnp.int32)
    packed = layer._pack_routing_exchange(
        blocks.reshape(-1),
        bits,
        hist.sum(axis=0),
        layer._exchange_layout(288, 6, 384, True),
    )
    return weights, packed, blocks, hist, row_routes


def router_candidate(xs):
    from vllm_torchtpu.kernels.experimental.push_fused_moe import layer

    def body(scores, gate, idx, weights, scale):
        valid = (
            jnp.any(jnp.isfinite(scores), axis=1, keepdims=True).T
            & jnp.any(jnp.isfinite(gate), axis=1, keepdims=True).T
        )
        weights, idx = weights.T, idx.T
        denom = jnp.maximum(jnp.sum(weights, axis=0, keepdims=True), 1e-20)
        weights = jnp.where(valid, weights / jnp.where(valid, denom, 1.0), 0.0)
        idx = jnp.where(valid, idx, 384)
        bits = lax.bitcast_convert_type(scale.T, jnp.int32)
        return weights, idx, bits, valid

    fused = rf.fuse_function(body, xs, name="dsv41_router_normalize_mask")

    def call(*xs):
        weights, idx, bits, valid = fused(*xs)
        blocks = idx.T.reshape(27, 64)
        hist = rf.host.pair_block_hist(blocks, 384)
        packed = layer._pack_routing_exchange(
            blocks.reshape(-1),
            bits.reshape(-1),
            hist.sum(axis=0),
            layer._exchange_layout(288, 6, 384, True),
        )
        return weights.T, packed, blocks, hist, valid.T

    return call


def fixtures(seed=0, case="regular", *, page_size=256, pages_per_seq=385):
    rng = np.random.default_rng(seed)
    bf = ml_dtypes.bfloat16

    def normal(shape):
        return rng.normal(size=shape).astype(bf)

    cases = {}
    for d in (1280, 2304, 5120, 6144):
        x = normal((288, d))
        if d == 6144:
            x = rng.normal(size=x.shape).astype(np.float32)
        if case == "edge":
            x[:4] = 0
            x[4:8] = np.asarray(1e-20, x.dtype)
            x[8:12] = -x[12:16]
        if case == "empty":
            x[:] = 0
        cases[f"quant{d}"] = (quant_baseline, quant_candidate, (x,), "production-jax")
    records = rng.integers(-4096, 36000, 288, dtype=np.int32)
    if case == "edge":
        records[:] = -1
        records[[0, 31, 32, 287]] = [0, 32767, 32768, 13]
    if case == "empty":
        records[:] = -1
    cases["compact"] = (
        compact_baseline,
        compact_candidate,
        (records,),
        "production-jax",
    )
    table = rng.normal(size=(98816, 64)).astype(np.float32)
    pos = (np.repeat(np.arange(48) * 1024, 6) + np.tile(np.arange(6), 48)).astype(
        np.int32
    )
    if case == "edge":
        pos[:4] = [-1, 98816, 0, 98815]
    cases["rope"] = (
        rope_gather.gather_rows,
        rope_candidate,
        (table, pos),
        "production-jax",
    )
    pages = rng.integers(1, 1024, (48, pages_per_seq), dtype=np.int32)
    selected = rng.integers(
        0, min(98304, pages_per_seq * page_size), (288, 512), dtype=np.int32
    )
    selected[:, 400:] = -1
    starts = np.arange(49, dtype=np.int32) * 6
    if case == "edge":
        selected[:6] = -1
        starts[1] = 0
        starts[-3:] = starts[-3]
        pages[::2, ::3] = 65539
    if case == "empty":
        starts[:] = 0
        selected[:] = -1
    cases["main_records"] = (
        functools.partial(main_records_baseline, page_size=page_size),
        functools.partial(main_records_candidate, page_size=page_size),
        (selected, pages, starts),
        "production-jax",
    )
    # Smaller owners, same 288x128 selected records and BF16 piece geometry.
    n = 288 * 128
    raw = normal((2, 8192, 256)).view(np.uint16)
    packed = (
        raw[..., :128].astype(np.uint32) | (raw[..., 128:].astype(np.uint32) << 16)
    ).view(np.int32)
    ids = (
        (
            np.repeat(np.arange(48) * 128, 6)[:, None]
            + np.tile(np.arange(6), 48)[:, None]
            + np.arange(128)[None, :]
        )
        .reshape(-1)
        .astype(np.int32)
    )
    if case == "edge":
        ids = rng.integers(-100, 8192, n, dtype=np.int32)
    cases["cache_unpack"] = (
        cache_baseline,
        cache_candidate,
        (packed[0], packed[1], ids),
        "production-jax",
    )
    flat = normal((288, 20480))
    fn = MHC["_pack_mix_weight"](
        jnp.asarray(rng.normal(0, 0.01, (24, 20480)), jnp.float32)
    )
    pre = rng.random((288, 4), dtype=np.float32)
    if case == "empty":
        flat[:] = 0
    cases["mhc_pre"] = (
        mhc_baseline,
        mhc_candidate,
        (flat, np.asarray(fn), np.ones(3, np.float32), np.zeros(24, np.float32), pre),
        "production-jax",
    )
    hidden, kv = normal((288, 4, 5120)), normal((288, 25600))
    qw, kw = normal((4, 5120)).astype(np.float32), normal((4, 5120)).astype(np.float32)
    mask = np.ones(288, np.float32)
    if case == "edge":
        hidden[:4] = 0
        kv[4:8] = 0
        mask[::3] = 0
    if case == "empty":
        hidden[:] = 0
        kv[:] = 0
        mask[:] = 0
    cases["engram"] = (
        engram_baseline,
        engram_candidate,
        (hidden, kv, qw, kw, mask),
        "jax-math-replay",
    )
    logits = rng.normal(size=(288, 384)).astype(np.float32)
    scores = 1 / (1 + np.exp(-logits))
    indices = np.argsort(scores, axis=-1)[:, -6:].astype(np.int32)
    weights = np.take_along_axis(scores, indices, axis=1)
    if case == "edge":
        logits[:32] = -np.inf
        scores[:32] = 0
        weights[:32] = 0
    if case == "empty":
        logits[:] = -np.inf
        scores[:] = 0
        weights[:] = 0
    cases["router"] = (
        router_baseline,
        router_candidate,
        (scores, logits, indices, weights, rng.random((288, 1), dtype=np.float32)),
        "production-jax-tail-replay",
    )
    return cases


def full_cache_fixture(case):
    """Original trace owner shapes; generate on TPU, reference only read rows."""
    rows = (28652288, 33423360)

    def words(xp, row, phase):
        lane = xp.arange(128, dtype=xp.uint32)[None, :]
        row = row.astype(xp.uint32)[:, None]
        lo = (row + lane + phase) % 256 + 0x3F00
        hi = (row * 3 + lane * 7 + phase) % 512 + 0xBE00
        return (lo | (hi << 16)).view(xp.int32)

    banks = []
    for size, phase in zip(rows, (13, 97), strict=True):

        def initialize(phase, _size=size):
            return words(jnp, jnp.arange(_size, dtype=jnp.uint32), phase)

        banks.append(jax.block_until_ready(jax.jit(initialize)(np.uint32(phase))))
    base = np.arange(48, dtype=np.int32) * 32768
    base[24:] = rows[0] // 2 + np.arange(24) * 32768
    ids = (
        (
            np.repeat(base, 6)[:, None]
            + np.tile(np.arange(6), 48)[:, None]
            + np.arange(128)[None, :]
        )
        .reshape(-1)
        .astype(np.int32)
    )
    if case == "edge":
        ids[:6] = [-1, 0, rows[0] // 2 - 1, rows[0] // 2, sum(rows) // 2 - 1, -1]
    valid = np.where(ids >= 0, ids, np.arange(ids.size) % (sum(rows) // 2))
    addresses = valid[None, :] * 2 + np.arange(2)[:, None]
    high = addresses >= rows[0]
    local = np.where(high, addresses - rows[0], addresses).reshape(-1)
    phase = np.where(high, 97, 13).reshape(-1, 1)
    lane = np.arange(128, dtype=np.uint32)[None, :]
    lo = ((local[:, None] + lane + phase) % 256 + 0x3F00).astype(np.uint16)
    hi = ((local[:, None] * 3 + lane * 7 + phase) % 512 + 0xBE00).astype(np.uint16)
    expected = np.stack(
        (
            lo.view(ml_dtypes.bfloat16).reshape(2, ids.size, 128),
            hi.view(ml_dtypes.bfloat16).reshape(2, ids.size, 128),
        ),
        axis=1,
    )
    return (*banks, ids), expected.reshape(4, ids.size, 128)


def compare(a, b, region):
    rows = []
    for want, got in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        want, got = np.ascontiguousarray(want), np.ascontiguousarray(got)
        exact = np.array_equal(want.view(np.uint8), got.view(np.uint8))
        if region not in ("mhc_pre", "engram"):
            np.testing.assert_array_equal(want.view(np.uint8), got.view(np.uint8))
        else:
            # No dtype changes or approximate nonlinear functions. Account only
            # for FP32 parallel reduction ordering, followed by BF16 rounding.
            tol = (2e-5, 2e-6) if want.dtype == np.float32 else (0.0078125, 2e-6)
            np.testing.assert_allclose(
                got.astype(np.float32),
                want.astype(np.float32),
                rtol=tol[0],
                atol=tol[1],
            )
        delta = np.abs(want.astype(np.float32) - got.astype(np.float32))
        rows.append(
            {
                "exact": exact,
                "max_abs": float(np.nanmax(delta)),
                "different_elements": int(np.count_nonzero(want != got)),
            }
        )
    return rows


def check_reference(expected, actual, name, inputs):
    if name.startswith("quant"):
        return peripheral_references.check_quantization(inputs, actual)
    if name == "router":
        want, got = np.asarray(expected[0]), np.asarray(actual[0])
        np.testing.assert_allclose(want, got, rtol=3e-7, atol=1e-8)
        result = compare(expected[1:], actual[1:], name)
        return {
            "weights_max_abs": float(np.max(np.abs(want - got))),
            "integer_outputs": result,
        }
    return compare(expected, actual, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--regions",
        nargs="+",
        default=[
            "quant1280",
            "quant2304",
            "quant5120",
            "quant6144",
            "compact",
            "rope",
            "main_records",
            "cache_unpack",
            "mhc_pre",
            "engram",
            "router",
        ],
    )
    ap.add_argument("--case", choices=["regular", "edge", "empty"], default="regular")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--full-cache", action="store_true")
    ap.add_argument("--page-size", type=int, default=256)
    ap.add_argument("--pages-per-seq", type=int, default=385)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = fixtures(
        args.seed, args.case, page_size=args.page_size, pages_per_seq=args.pages_per_seq
    )
    full_cache_reference = None
    if args.full_cache:
        assert args.regions == ["cache_unpack"]
        cache_xs, full_cache_reference = full_cache_fixture(args.case)
        cases["cache_unpack"] = (
            cache_baseline,
            cache_candidate,
            cache_xs,
            "production-jax-original-owner-shapes",
        )
    report = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {
            p: importlib.metadata.version(p)
            for p in ("jax", "jaxlib", "numpy", "ml_dtypes")
        },
        "device": str(jax.devices()[0]),
        "case": args.case,
        "seed": args.seed,
        "main_records_geometry": {
            "page_size": args.page_size,
            "pages_per_seq": args.pages_per_seq,
        },
        "regions": {},
    }
    for name in args.regions:
        baseline, build, host_xs, provenance = cases[name]
        xs = jax.device_put(host_xs, jax.devices()[0])
        row = {
            "provenance": provenance,
            "inputs": [
                {"shape": list(x.shape), "dtype": str(x.dtype)} for x in host_xs
            ],
            "variants": {},
        }
        compiled, expected = {}, None
        independent = (
            full_cache_reference
            if args.full_cache
            else peripheral_references.reference(
                name, host_xs, page_size=args.page_size
            )
        )
        for variant in ("baseline", "fused"):
            try:
                fn = baseline if variant == "baseline" else build(host_xs)

                def call(*args, _fn=fn):
                    return _fn(*args)

                call.__name__ = f"{name}_{variant}"
                exe = jax.jit(call).lower(*xs).compile()
                (args.out / f"{name}-{variant}.hlo").write_text(exe.as_text())
                actual = jax.block_until_ready(exe(*xs))
                if expected is None:
                    expected = jax.tree.map(np.asarray, actual)
                check = compare(expected, actual, name)
                compiled[variant] = exe
                for _ in range(5):
                    jax.block_until_ready(exe(*xs))
                row["variants"][variant] = {
                    "baseline_comparison": check,
                    "numpy_reference": check_reference(
                        independent, actual, name, host_xs
                    ),
                    "host_us": [],
                }
            except Exception as exc:
                compiled.pop(variant, None)
                row["variants"][variant] = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                print(name, variant, traceback.format_exc(), flush=True)
        for i in range(args.repeat):
            for variant in list(compiled) if i % 2 == 0 else list(compiled)[::-1]:
                begin = time.perf_counter_ns()
                jax.block_until_ready(compiled[variant](*xs))
                row["variants"][variant]["host_us"].append(
                    (time.perf_counter_ns() - begin) / 1000
                )
        if args.profile:
            with jax.profiler.trace(str(args.out / f"profile-{name}")):
                for _ in range(12):
                    for exe in compiled.values():
                        jax.block_until_ready(exe(*xs))
        for v in row["variants"].values():
            if "host_us" in v:
                v["host_median_us"] = float(np.median(v["host_us"]))
        report["regions"][name] = row
        (args.out / "results.json").write_text(json.dumps(report, indent=2))
        print(
            name,
            {
                k: {a: b for a, b in v.items() if a not in ("host_us", "traceback")}
                for k, v in row["variants"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
