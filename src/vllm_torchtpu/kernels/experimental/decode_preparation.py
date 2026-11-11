# SPDX-License-Identifier: Apache-2.0
"""Experimental decode preparation boundaries for measured decode buckets.

These functions keep communication and SparseCore producers outside the
TensorCore preparation call. Callers retain their general-shape paths.

Every fused region asks `admit` before taking its fused path. Token buckets
below are the counts whose fused lowering and native parity were measured on
v7; layout contracts are checked by the region and passed in as a problem
string. `configure` selects the policy once, before the model is traced.
"""

import dataclasses
import enum
import logging

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import get_tpu_version

# Stdlib logging under vLLM's namespace: kernels/ must not import vLLM
# (R-2.1), yet vLLM's handlers still format these records when present.
logger = logging.getLogger("vllm." + __name__)


class FusionMode(enum.Enum):
    AUTO = "auto"  # fuse where admitted, otherwise fall back and log once
    OFF = "off"  # always take the native path (A/B baseline)
    REQUIRE = "require"  # raise when a decode-sized call cannot fuse


# Measured buckets per region. A count above a region's largest bucket is a
# prefill-sized call and falls back silently, even under REQUIRE.
ROUTING_TOKENS = frozenset({96, 192, 256, 288, 384})  # per-shard t_local
PRESORT_TOKENS = ROUTING_TOKENS
COMPACT_TOKENS = frozenset({96, 192, 240, 256, 288, 384})
COMPRESSOR_STATE_TOKENS = frozenset({96, 192, 240, 256, 288})
COMPRESSOR_ADDRESS_TOKENS = frozenset({288})
INDEXER_TOKENS = frozenset({128, 256, 288, 384, 512})
MHC_TOKENS = frozenset({288})
CACHE_UNPACK_RECORDS = frozenset({12288, 36864})
SWA_PACK_TOKENS = frozenset({96, 192, 288})
# (tokens, requests): the region builds token x request matrices.
MAIN_RECORD_BATCHES = frozenset({(96, 16), (192, 32), (288, 32), (288, 48)})
# The main-record kernel's one-hot gather costs O(T * requests * columns);
# smaller page tables than the largest measured one are strictly cheaper.
MAIN_RECORD_MAX_COLUMNS = 772
MEASURED_TPU_VERSION = 7


@dataclasses.dataclass(frozen=True)
class CompressorLayout:
    """Cache geometry the compressor address kernel is specialized for."""

    state_capacity: int = 16  # rows per compressor state ring page
    state_stride: int = 16
    state_offset: int = 0
    window: int = 128  # SWA bounded-replay window
    main_capacity: int = 128  # compressed records per main KV page
    main_stride: int = 128
    main_offset: int = 0
    index_capacity: int = 256  # records per FP8 index page

    @property
    def tag(self) -> str:
        return "_".join(str(v) for v in dataclasses.astuple(self))


DEFAULT_COMPRESSOR_LAYOUT = CompressorLayout()

_mode = FusionMode.AUTO
_mode_used = False
_reported: set[tuple[str, str]] = set()


def configure(mode: str | FusionMode) -> None:
    """Select the policy; it must not change once a region has been traced."""
    global _mode
    mode = FusionMode(mode.lower() if isinstance(mode, str) else mode)
    if _mode_used and mode is not _mode:
        raise RuntimeError(
            f"Decode fusion mode is already {_mode.value} in traced graphs; set "
            "TPU_DEEPSEEK_V41_DECODE_FUSION before the model is compiled"
        )
    _mode = mode


def admit(
    region: str,
    problem: str | None = None,
    *,
    tokens: int | tuple[int, ...] | None = None,
    buckets: frozenset | None = None,
    backend: bool = True,
    report: bool = True,
) -> bool:
    """Whether ``region`` takes its fused path.

    ``problem`` names a violated layout contract. ``tokens`` is checked
    against ``buckets``; any hashable bucket key works, with ``tokens`` an
    int or a tuple whose first item is the token count. ``backend`` requires
    a TPU. Under AUTO a decode-sized miss is logged once per reason.
    ``report=False`` marks a region whose miss takes another tuned path, so
    it neither logs nor raises.
    """
    global _mode_used
    _mode_used = True
    if _mode is FusionMode.OFF:
        return False
    if backend and jax.default_backend() != "tpu":
        return False  # interpret/CPU runs always take the native path
    if problem is None and buckets is not None and tokens not in buckets:
        count = tokens[0] if isinstance(tokens, tuple) else tokens
        largest = max(b[0] if isinstance(b, tuple) else b for b in buckets)
        if count is not None and count > largest:
            return False
        problem = f"{tokens} is not a measured bucket {sorted(buckets)}"
    if problem is None:
        return True
    if not report:
        return False
    if _mode is FusionMode.REQUIRE:
        raise NotImplementedError(
            f"DSv4.1 decode fusion '{region}' is required but unavailable: "
            f"{problem}. Add a measured bucket or set "
            "TPU_DEEPSEEK_V41_DECODE_FUSION=auto to allow the native path."
        )
    if (region, problem) not in _reported:
        _reported.add((region, problem))
        logger.warning("DSv4.1 decode fusion '%s' falls back: %s", region, problem)
    return False


def first_problem(*checks: tuple[bool, str]) -> str | None:
    """Description of the first failed ``(ok, description)`` layout check."""
    return next((description for ok, description in checks if not ok), None)


def tpu_version_problem(version: int = MEASURED_TPU_VERSION) -> str | None:
    """Buckets were measured on one TPU generation; others keep native paths."""
    if jax.default_backend() != "tpu":
        return None  # `admit` reports the backend itself, silently
    actual = get_tpu_version()
    return None if actual == version else f"TPU v{actual} (measured on v{version})"


def compressor_layout_problem(
    actual: CompressorLayout, expected: CompressorLayout = DEFAULT_COMPRESSOR_LAYOUT
) -> str | None:
    """Fields where the bound cache differs from the registered kernel."""
    diffs = [
        f"{f.name}={getattr(actual, f.name)} (kernel {getattr(expected, f.name)})"
        # Pass an instance: Dynamo cannot trace ``fields`` on the class.
        for f in dataclasses.fields(expected)
        if getattr(actual, f.name) != getattr(expected, f.name)
    ]
    return "cache layout " + ", ".join(diffs) if diffs else None


def take_along(values, indices, axis):
    """Explicit batched gather; retain singleton dimensions for Mosaic."""
    batch = tuple(i for i in range(values.ndim) if i != axis)
    return lax.gather(
        values,
        indices[..., None],
        lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(axis,),
            start_index_map=(axis,),
            operand_batching_dims=batch,
            start_indices_batching_dims=batch,
        ),
        slice_sizes=(1,) * values.ndim,
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def prefix_sum(a, axis=None, dtype=None, out=None):
    """Integer-only inclusive scan with explicit shift/add dependencies."""
    assert out is None
    if axis is None:
        a, axis = a.reshape(-1), 0
    if dtype is not None:
        a = a.astype(dtype)
    assert jnp.issubdtype(a.dtype, jnp.integer)
    distance = 1
    while distance < a.shape[axis]:
        tail = lax.slice_in_dim(a, 0, a.shape[axis] - distance, axis=axis)
        pad_shape = list(a.shape)
        pad_shape[axis] = distance
        a = a + jnp.concatenate((jnp.zeros(pad_shape, a.dtype), tail), axis=axis)
        distance *= 2
    return a


def fuse_function(fn, args, *, name, interpret=False):
    """Real Pallas boundary, with the same externally visible array outputs."""
    out_shape = jax.eval_shape(fn, *args)
    leaves, tree = jax.tree.flatten(out_shape)

    def kernel(*refs):
        outputs = fn(*(r[...] for r in refs[: len(args)]))
        for ref, value in zip(refs[len(args) :], jax.tree.leaves(outputs)):
            ref[...] = value

    call = pl.pallas_call(
        kernel,
        out_shape=tuple(leaves),
        name=name,
        interpret=interpret,
        compiler_params=None if interpret else _half_vmem(),
    )
    return lambda *xs: jax.tree.unflatten(tree, call(*xs))


def _half_vmem() -> pltpu.CompilerParams:
    return pltpu.CompilerParams(
        vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes // 2
    )


def pack_matrix(records, values, *, base, capacity, padded):
    n = values.shape[0]
    local = jnp.where(
        (max(base, 64) <= records.T) & (base + capacity > records.T),
        records.T - base,
        capacity,
    )
    idx = jnp.arange(n, dtype=jnp.int32)[:, None]
    ranks = jnp.sum(
        (local > local.T) | ((local == local.T) & (idx > idx.T)), axis=1, keepdims=True
    )
    permutation = idx == ranks.T
    ids = jnp.sum(jnp.where(permutation, local.T, 0), axis=1, keepdims=True)
    next_ids = jnp.concatenate((ids[1:], jnp.full((1, 1), capacity, jnp.int32)), axis=0)
    pairs = ids // 2
    boundary = (ids < capacity) & (
        jnp.concatenate(
            (jnp.ones((1, 1), jnp.int32), (pairs[1:] != pairs[:-1]).astype(jnp.int32)),
            axis=0,
        )
        != 0
    )
    even = ids % 2 == 0
    odd = (~even) | (next_ids // 2 == pairs)
    value_bits = lax.bitcast_convert_type(values, jnp.uint16).astype(jnp.uint32)
    bits = jnp.dot(
        permutation.astype(jnp.float32),
        value_bits.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
    ).astype(jnp.uint32)
    next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)), axis=0)
    low = jnp.where(even, bits, jnp.uint32(0))
    high = jnp.where(odd, jnp.where(even, next_bits, bits), jnp.uint32(0))
    words = lax.bitcast_convert_type(low | (high << 16), jnp.int32)
    masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
        odd, jnp.int32(-65536), jnp.int32(0)
    )
    return (
        jnp.pad(
            jnp.where(boundary, pairs, -1),
            ((0, padded - n), (0, 0)),
            constant_values=-1,
        ),
        jnp.broadcast_to(jnp.pad(masks, ((0, padded - n), (0, 0))), (padded, 512)),
        jnp.pad(words, ((0, padded - n), (0, 0))),
    )


def swa_word_pool_pack(records, values, *, capacity, padded, interpret=False):
    """Sort and pair SWA records for the word-pool publisher in one TC region.

    Returns 128-word physical row ids (four per token pair, -1 when skipped),
    the packed words and their BF16-half masks, padded to ``4 * padded`` rows.
    """

    def body(records, values):
        return pack_matrix(records, values, base=0, capacity=capacity, padded=padded)

    inputs = (records[None, :], values)
    pairs, masks, words = fuse_function(
        body, inputs, name="decode_swa_sort_pack", interpret=interpret
    )(*inputs)
    physical = jnp.where(
        pairs >= 0, pairs * 4 + jnp.arange(4, dtype=jnp.int32)[None, :], -1
    )
    return physical.reshape(-1), words.reshape(-1, 128), masks.reshape(-1, 128)


def main_records_call(xs, *, page_size, page_stride=None, row_offset=0):
    selected, pages, starts = xs
    t, k = selected.shape
    requests, pps = pages.shape
    padded_pps = (pps + 127) // 128 * 128
    stride = page_size if page_stride is None else page_stride

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
            value = take_along(chunk, jnp.clip(local, 0, 127), axis=1)
            physical = jnp.where((local >= 0) & (local < 128), value, physical)
        records = jnp.where(
            valid, physical * stride + row_offset + logical % page_size, -1
        )
        return records, jnp.sum(valid, axis=1, keepdims=True, dtype=jnp.int32).T

    fused = fuse_function(body, xs, name="dsv41_main_records_matrix")

    def call(*xs):
        records, counts = fused(*xs)
        return records, counts.reshape(t)

    return call


def compact_records(records: jax.Array, total_records: int, tb: int):
    n = records.size

    def body(records):
        r = records.reshape(1, n)
        valid = (r >= 0) & (r < total_records)
        rank = prefix_sum(valid.astype(jnp.int32), axis=1) - 1
        match = valid & (rank == jnp.arange(n, dtype=jnp.int32)[:, None])
        token = jnp.sum(jnp.where(match, jnp.arange(n)[None, :], 0), axis=1)
        record = jnp.sum(jnp.where(match, r, 0), axis=1)
        boundary = (jnp.arange(n // tb + 1) * tb)[:, None]
        offsets = jnp.sum(
            valid & (jnp.arange(n)[None, :] < boundary), axis=1, dtype=jnp.int32
        )
        return jnp.stack((token, record), axis=1), offsets

    pairs, offsets = fuse_function(body, (records,), name="decode_compact_records")(
        records
    )
    return pairs.reshape(-1), offsets


def unpack_cache_words(words: jax.Array) -> jax.Array:
    chunks, n, width = words.shape
    tile = max(v for v in range(128, 4097, 128) if n % v == 0)

    def kernel(wr, out):
        value = wr[...]
        lo = lax.bitcast_convert_type(value << 16, jnp.float32).astype(jnp.bfloat16)
        hi = lax.bitcast_convert_type(value & jnp.int32(-65536), jnp.float32).astype(
            jnp.bfloat16
        )
        out[0, ...], out[1, ...] = lo, hi

    return pl.pallas_call(
        kernel,
        grid=(chunks, n // tile),
        in_specs=[pl.BlockSpec((1, tile, width), lambda c, i: (c, i, 0))],
        out_specs=pl.BlockSpec((2, 1, tile, width), lambda c, i: (c, 0, i, 0)),
        out_shape=jax.ShapeDtypeStruct((2 * chunks, 1, n, width), jnp.bfloat16),
        name="decode_cache_word_unpack",
    )(words).reshape(2 * chunks, n, width)


def compressor_inputs_problem(
    table, starts, kv_starts, positions, minimum, *page_tables
) -> str | None:
    """Shape/dtype contract shared by both compressor metadata kernels.

    Request and token counts come from the arguments; nothing is fixed to one
    batch. Every input is INT32: the kernels never cast metadata.
    """
    requests = table.shape[0] if table.ndim == 2 else -1
    expected = (
        (table, "table", (requests, max(table.shape[-1], 1))),
        (starts, "query_start_loc", (requests + 1,)),
        (kv_starts, "kv query_start_loc", (requests + 1,)),
        (positions, "positions", (positions.shape[0],) if positions.ndim else ()),
        (minimum, "window_starts", (requests,)),
        *((t, "page table", (requests, max(t.shape[-1], 1))) for t in page_tables),
    )
    for value, name, shape in expected:
        if value.shape != shape or not shape or value.dtype != jnp.int32:
            return (
                f"{name} is {value.dtype}{list(value.shape)}, needs int32{list(shape)}"
            )
    if requests < 1:
        return "no requests"
    return None


def _compressor_state_addresses(
    table, starts, kv_starts, positions, minimum, layout: CompressorLayout
):
    capacity, stride, offset = (
        layout.state_capacity,
        layout.state_stride,
        layout.state_offset,
    )
    tokens, requests = positions.size, table.shape[0]
    rows = jnp.arange(tokens, dtype=jnp.int32)
    r = jnp.arange(requests, dtype=jnp.int32)[:, None]

    def map_requests(boundaries):
        req = jnp.minimum(
            jnp.sum(rows[None, :] >= boundaries[1:, None], axis=0), requests - 1
        )
        return req[None, :] == r

    selected = map_requests(starts)

    def select(values):
        return jnp.sum(jnp.where(selected, values[:, None], 0), axis=0)

    first, end, blocks = select(starts[:-1]), select(starts[1:]), select(table[:, 0])
    has_prev = rows != first
    active = rows < starts[-1]
    slots = jnp.where(active, blocks * capacity + positions % capacity, -1)
    previous = jnp.where(active, blocks * capacity + (positions - 1) % capacity, -1)
    stored = (slots >= 0) & (rows >= end - capacity)
    replay_start = jnp.sum(
        jnp.where(map_requests(kv_starts), minimum[:, None], 0), axis=0
    )
    replay = (
        (replay_start > 0)
        & (positions >= replay_start)
        & (positions < replay_start + layout.window)
    )
    valid = (rows < kv_starts[-1]) & ((positions + 1) % 2 == 0) & ~replay

    def physical(slot):
        return offset + slot // capacity * stride + slot % capacity

    return (
        has_prev.astype(jnp.int32),
        valid.astype(jnp.int32),
        jnp.where(stored, physical(slots), -1),
        jnp.where(has_prev | (slots < 0), -1, physical(previous)),
    )


def _compact_ring(records, total_records):
    """Stable ``(token, record)`` pairs and ``[0, count)`` for one whole tile."""
    n = records.size
    r = records.reshape(1, n)
    valid = (r >= 0) & (r < total_records)
    rank = prefix_sum(valid.astype(jnp.int32), axis=1) - 1
    match = valid & (rank == jnp.arange(n, dtype=jnp.int32)[:, None])
    token = jnp.sum(jnp.where(match, jnp.arange(n)[None, :], 0), axis=1)
    record = jnp.sum(jnp.where(match, r, 0), axis=1)
    count = jnp.sum(valid, dtype=jnp.int32)
    return jnp.stack((token, record), axis=1), jnp.stack((jnp.int32(0), count))


def _check_compressor_inputs(*args):
    problem = compressor_inputs_problem(*args)
    if problem is not None:
        raise ValueError(f"Packed compressor metadata: {problem}")


def compressor_state_packed(
    table: jax.Array,
    starts: jax.Array,
    kv_starts: jax.Array,
    positions: jax.Array,
    minimum: jax.Array,
    *,
    total_records: int,
    layout: CompressorLayout = DEFAULT_COMPRESSOR_LAYOUT,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Decode flags and stable ring tables for one whole compressor tile."""
    args = (table, starts, kv_starts, positions, minimum)
    _check_compressor_inputs(*args)

    def kernel(*refs):
        state, ss, ks, pos, mins = (r[...] for r in refs[:5])
        prev, valid, write, read = _compressor_state_addresses(
            state, ss, ks, pos, mins, layout
        )
        outputs = (
            prev,
            valid,
            *_compact_ring(write, total_records),
            *_compact_ring(read, total_records),
        )
        for ref, value in zip(refs[5:], outputs, strict=True):
            ref[...] = value

    n = positions.shape[0]
    outputs = pl.pallas_call(
        kernel,
        out_shape=tuple(
            jax.ShapeDtypeStruct(s, jnp.int32)
            for s in ((n,), (n,), (n, 2), (2,), (n, 2), (2,))
        ),
        name="dsv41_compressor_state_packed",
        interpret=interpret,
        compiler_params=None if interpret else _half_vmem(),
    )(*args)
    return (
        outputs[0][:, None],
        outputs[1][:, None],
        outputs[2].reshape(-1),
        outputs[3],
        outputs[4].reshape(-1),
        outputs[5],
    )


def compressor_addresses_packed(
    table: jax.Array,
    starts: jax.Array,
    kv_starts: jax.Array,
    positions: jax.Array,
    minimum: jax.Array,
    main: jax.Array,
    index: jax.Array,
    *,
    total_records: int,
    layout: CompressorLayout = DEFAULT_COMPRESSOR_LAYOUT,
    interpret: bool = False,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Decode state, stable ring tables and ratio-2 main/index addresses.

    Main records are physical rows (``main_offset + page * main_stride +
    slot``), matching ``PagedCacheRows.row_indices``; index slots stay
    logical, matching the native ``_cache_slots``.
    """
    args = (table, starts, kv_starts, positions, minimum, main, index)
    _check_compressor_inputs(*args)
    n, requests = positions.shape[0], table.shape[0]

    def kernel(*refs):
        state, ss, ks, pos, mins, mt, it = (r[...] for r in refs[:7])
        prev, valid, write, read = _compressor_state_addresses(
            state, ss, ks, pos, mins, layout
        )
        rows = jnp.arange(n, dtype=jnp.int32)
        request = jnp.minimum(
            jnp.sum(rows[None, :] >= ks[1:, None], axis=0), requests - 1
        )
        selected = request[None, :] == jnp.arange(requests)[:, None]

        def slots(page_table, capacity):
            # Returns (page, slot within page), -1 page past the last request.
            logical = pos // 2
            col = jnp.minimum(logical // capacity, page_table.shape[1] - 1)
            col = jnp.where(col < 0, col + page_table.shape[1], col)
            indices = jnp.broadcast_to((col % 128)[None, :], (requests, n))
            chunks = (page_table.shape[1] + 127) // 128
            padded = jnp.pad(
                page_table, ((0, 0), (0, chunks * 128 - page_table.shape[1]))
            )
            per_request = jnp.zeros((requests, n), jnp.int32)
            for chunk in range(chunks):
                tile = padded[:, chunk * 128 : (chunk + 1) * 128]
                gathered = take_along(tile, indices, axis=1)
                per_request += jnp.where(col[None, :] // 128 == chunk, gathered, 0)
            blocks = jnp.sum(jnp.where(selected, per_request, 0), axis=0)
            return rows < ks[-1], blocks, logical % capacity

        live, block, slot = slots(mt, layout.main_capacity)
        main_rows = layout.main_offset + block * layout.main_stride + slot
        records = jnp.where(live & (block >= 0) & (valid != 0), main_rows, -1)
        live, block, slot = slots(it, layout.index_capacity)
        index_slots = jnp.where(live, block * layout.index_capacity + slot, -1)
        outputs = (
            prev,
            valid,
            *_compact_ring(write, total_records),
            *_compact_ring(read, total_records),
            records,
            index_slots,
        )
        for ref, value in zip(refs[7:], outputs, strict=True):
            ref[...] = value

    outputs = pl.pallas_call(
        kernel,
        out_shape=tuple(
            jax.ShapeDtypeStruct(s, jnp.int32)
            for s in ((n,), (n,), (n, 2), (2,), (n, 2), (2,), (n,), (n,))
        ),
        name=(
            "dsv41_compressor_addresses_packed"
            f"_c{layout.main_capacity}_i{layout.index_capacity}"
        ),
        interpret=interpret,
        compiler_params=None if interpret else _half_vmem(),
    )(*args)
    return tuple(
        x.reshape(s)
        for x, s in zip(
            outputs,
            ((n, 1), (n, 1), (2 * n,), (2,), (2 * n,), (2,), (n,), (n,)),
            strict=True,
        )
    )
