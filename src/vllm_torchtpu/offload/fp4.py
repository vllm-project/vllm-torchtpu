# SPDX-License-Identifier: Apache-2.0
"""FP4 host copy of the main KV (VLLM_TPU_OFFLOAD_FP4).

With the host tier classes and the victim policy, the ``main`` class (the QSA
layers' FP8 main KV) keeps MXFP4 on the host instead of the pool's FP8 bytes:
e2m1 values (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives) with one
power-of-two (e8m0) scale per 32 consecutive values. A token row of 1,024 FP8
values (4 head slots x 256 lanes) becomes 512 packed bytes and 32 scale bytes,
so a host entry is 544 / 1024 = 53% of the FP8 page.

Raiden copies bytes between registered device rows and host slots and never
runs device code, so the conversion happens on each worker around the DMA, in
small HBM staging buffers that the class registers in place of the pool:

- store (a victim block HBM evicts): the scheduler gives each main-class key a
  staging slot and ships ``(pool block, slot)`` in the step's metadata; every
  worker quantizes those pool pages into its staging slots at ``start_load_kv``,
  ahead of the step's forward on the same stream (which may already write the
  block's new owner), records an event and acknowledges it once retired. The
  scheduler then saves the slots (``submit_store_job`` with the slots as device
  rows); the pool block is free as soon as the quantize ran, so the main class
  needs no drain.
- load: the class's child job lands in staging slots; when the whole job has
  finished, the scheduler ships ``(slot, pool block)`` with the job's completion
  and every worker dequantizes into the pool block at ``start_load_kv``, before
  the request is scheduled again. The slots return to the free list once every
  worker acknowledges that the dequantize retired.

Staging layout per region: ``[slots, stage_rows, 4, 256]`` uint8, the pool's
own 1,024-byte row shape. Per slot, the packed values (``rows * 4`` runs of
128 bytes: one head slot of 256 lanes) are followed by the scales (``rows *
32`` bytes). A 3,200-token page packs into 1,700 rows (1,740,800 B, 53% of
the FP8 page). Writes into the pool and staging are whole-page DMAs into the
aliased target (``page_scatter``), so no buffer is ever rebound; pool pages are
read as FP8 values and only converted to float (never bitcast FP8 -> integer
in XLA, which canonicalised NaN encodings).
"""

# No `from __future__ import annotations`: pallas.jax_op reads the kernels' argument
# types from their annotations (offload/tier_check.py does the same).

import functools
import math
import threading
from collections.abc import Sequence

import numpy as np
import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Token row geometry of a main-KV page: 4 head slots x 256 lanes of FP8.
ROW_SLOTS, ROW_LANES = 4, 256
ROW_VALUES = ROW_SLOTS * ROW_LANES
GROUP = 32  # values per scale
PACKED_LANES = ROW_LANES // 2  # two e2m1 nibbles a byte
SCALES_PER_ROW = ROW_VALUES // GROUP  # 32
FP8_MAX = 448.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between grid values: a magnitude above threshold i has code > i. A tie
# rounds to the even mantissa (OCP MX round-to-nearest-even): up at 0.75, 1.75 and
# 3.5 (to 1, 2, 4), down at 0.25, 1.25, 2.5 and 5 (to 0, 1, 2, 4).
E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_TIE_UP = (False, True, False, True, False, True, False)
SCALE_BIAS = 127
# Batch sizes compiled for each direction (a batch is padded up to the next one by
# repeating its last pair, which rewrites the same bytes; bigger batches are split).
BUCKETS = (1, 4, 16, 32, 64)


def page_rows(page_bytes: int) -> int:
    if page_bytes % ROW_VALUES:
        raise ValueError(
            f"VLLM_TPU_OFFLOAD_FP4: a main-KV page of {page_bytes} B is not whole "
            f"{ROW_VALUES}-byte token rows"
        )
    rows = page_bytes // ROW_VALUES
    if rows % 32:
        raise ValueError(f"VLLM_TPU_OFFLOAD_FP4: {rows} rows a page is not a multiple of 32")
    return rows


def packed_rows(rows: int) -> int:
    """1,024-byte rows the MXFP4 bytes of a ``rows``-row page fill: 544 B a token row."""
    return rows * (ROW_VALUES // 2 + SCALES_PER_ROW) // ROW_VALUES


def stage_rows(rows: int) -> int:
    """Rows of one staging slot (the page writer DMAs whole slots: no tile padding)."""
    return packed_rows(rows)


def fp4_page_bytes(page_bytes: int) -> int:
    """Host bytes of one main-KV page in MXFP4 (values + scales)."""
    return stage_rows(page_rows(page_bytes)) * ROW_VALUES


def staging_shape(page_bytes: int, slots: int) -> tuple[int, int, int, int]:
    """``[slots, stage_rows, 4, 256]`` uint8, in the pool's 1,024-byte row shape."""
    return (slots, stage_rows(page_rows(page_bytes)), ROW_SLOTS, ROW_LANES)


# -- GDN state pages (VLLM_TPU_OFFLOAD_FP4_GDN) -----------------------------------
# A GDN page is rows of 1,024 bytes holding FP32 words in the Mosaic typed view
# (pd_staging._typed: one word a lane, its four bytes over the row's four sublanes):
# 256 FP32 values a row (the SSM state, then the conv state, then padding). MXFP4 of
# a row: 128 packed bytes + 8 scales = 136 B, 13.3% of the row.
GDN_ROW_VALUES = ROW_LANES  # FP32 words a row
GDN_ROW_FP4 = GDN_ROW_VALUES // 2 + GDN_ROW_VALUES // GROUP  # 136


def gdn_packed_rows(rows: int) -> int:
    """1,024-byte rows the MXFP4 bytes of a ``rows``-row GDN page fill."""
    if (rows * GDN_ROW_FP4) % ROW_VALUES:
        raise ValueError(
            f"VLLM_TPU_OFFLOAD_FP4_GDN: {rows} rows a page do not pack into whole rows"
        )
    return rows * GDN_ROW_FP4 // ROW_VALUES


def fp4_gdn_page_bytes(page_bytes: int) -> int:
    """Host bytes of one GDN state page in MXFP4."""
    return gdn_packed_rows(page_rows(page_bytes)) * ROW_VALUES


def slot_layout(main_slots: int, gdn_slots: int) -> dict[str, tuple[int, int]]:
    """Kind -> (first global slot id, slots). One id space over both kinds, so the
    scheduler's pairs and the workers' metadata carry plain ints; a class hands
    Raiden its local slot (id - first)."""
    out = {"main": (0, int(main_slots))}
    if gdn_slots:
        out["gdn"] = (int(main_slots), int(gdn_slots))
    return out


def configured_slot_layout() -> dict[str, tuple[int, int]]:
    from vllm_torchtpu import envs

    main = int(envs.VLLM_TPU_OFFLOAD_FP4_SLOTS)
    gdn = 0
    if envs.VLLM_TPU_OFFLOAD_FP4_GDN:
        gdn = int(envs.VLLM_TPU_OFFLOAD_FP4_GDN_SLOTS) or main
    return slot_layout(main, gdn)


def page_gather_u8(source, ids, tile: int = 64):
    """``source[ids]`` (kernel pages ``[P, R, *row]`` of the FP8 carrier) as uint8 rows
    ``[n * R, 4, 256]``: each tile of ``tile`` rows lands by DMA in a VMEM block of the
    carrier dtype and is stored through a Mosaic vreg bitcast (``pltpu.bitcast``), as
    ``pd_staging.dma_row_gather`` does, so no XLA op ever holds the bytes as FP8 (an
    XLA FP8 -> integer bitcast canonicalised NaN encodings, which FP32 state words
    contain). One SMEM index a page."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = int(source.shape[1])
    if rows % tile:
        raise ValueError(f"page_gather_u8: {rows} rows a page, not tiles of {tile}")
    steps = rows // tile
    stage = source.reshape(-1, ROW_SLOTS, ROW_LANES)

    def kernel(ids_ref, src_ref, out_ref, sem, scratch):
        i, c = pl.program_id(0), pl.program_id(1)
        pltpu.async_copy(
            src_ref.at[pl.ds(ids_ref[i] * rows + c * tile, tile)], scratch, sem
        ).wait()
        out_ref[...] = pltpu.bitcast(scratch[...], jnp.uint8)

    n = ids.shape[0]
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(n, steps),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(
                (tile, ROW_SLOTS, ROW_LANES), lambda i, c, *_: (i * steps + c, 0, 0)
            ),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((tile, ROW_SLOTS, ROW_LANES), stage.dtype),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((n * rows, ROW_SLOTS, ROW_LANES), jnp.uint8),
        name="qwen4_offload_fp4_page_gather_u8",
    )(ids.astype(jnp.int32), stage)


def _words(rows_u8, jnp, lax):
    """``[..., 4, 256]`` uint8 rows -> ``[..., 256]`` FP32 (pd_staging._typed)."""
    w = jnp.swapaxes(rows_u8, -1, -2)  # [..., 256, 4]: one word a lane
    return lax.bitcast_convert_type(lax.bitcast_convert_type(w, jnp.uint32), jnp.float32)


def _unwords(x, jnp, lax):
    """Inverse of ``_words``: ``[..., 256]`` FP32 -> ``[..., 4, 256]`` uint8."""
    u = lax.bitcast_convert_type(lax.bitcast_convert_type(x, jnp.uint32), jnp.uint8)
    return jnp.swapaxes(u, -1, -2)


def _pow2(e, jnp):
    """Exact ``2.0 ** e`` for integer ``e`` in [-126, 127], from the exponent bits
    (``exp2`` is not exact on every backend)."""
    from jax import lax

    bits = ((e.astype(jnp.int32) + SCALE_BIAS) << 23).astype(jnp.uint32)
    return lax.bitcast_convert_type(bits, jnp.float32)


def _mx_encode(g, jnp):
    """``[..., 32]`` float groups -> (nibbles ``[..., 32]`` uint8, exponents ``[..., 1]``)."""
    amax = jnp.max(jnp.abs(g), axis=-1, keepdims=True)
    mant, expo = jnp.frexp(amax / 6.0)
    e = expo - (mant == 0.5).astype(expo.dtype)
    e = jnp.where(amax > 0, e, 0)
    e = jnp.clip(e, -SCALE_BIAS + 1, SCALE_BIAS - 1)
    y = jnp.abs(g) * _pow2(-e, jnp)
    code = jnp.zeros(y.shape, jnp.uint8)
    for t, up in zip(E2M1_THRESHOLDS, E2M1_TIE_UP):
        code = code + ((y >= t) if up else (y > t)).astype(jnp.uint8)
    return code | ((g < 0).astype(jnp.uint8) << 3), e


def _mx_decode(nib, scales_u8, jnp):
    # Compares and selects, not a table lookup: on the TPU ``grid[codes]`` is an XLA
    # gather per value (the main class's 16-page dequant spent 442 of 448 ms in it,
    # j-84136a6b-98f3-48a8-9dba).
    code = nib & 7
    mag = jnp.zeros(code.shape, jnp.float32)
    for k, v in enumerate(E2M1_GRID):
        if v:
            mag = jnp.where(code == k, jnp.float32(v), mag)
    val = jnp.where((nib >> 3) == 1, -mag, mag)
    return val * _pow2(scales_u8.astype(jnp.int32) - SCALE_BIAS, jnp)


def quantize_gdn_pages(rows_u8, jnp, lax):
    """``[n, rows, 4, 256]`` uint8 GDN rows -> ``[n, gdn_packed_rows, 4, 256]`` uint8:
    per page the packed nibbles (``rows * 128`` B) then the scales (``rows * 8``)."""
    n, rows = rows_u8.shape[0], rows_u8.shape[1]
    g = _words(rows_u8, jnp, lax).reshape(n, rows, GDN_ROW_VALUES // GROUP, GROUP)
    nib, e = _mx_encode(g, jnp)
    nib = nib.reshape(n, rows, GDN_ROW_VALUES // 2, 2)
    packed = (nib[..., 0] | (nib[..., 1] << 4)).reshape(n, -1)
    scales = (e + SCALE_BIAS).astype(jnp.uint8).reshape(n, -1)
    out = jnp.concatenate([packed, scales], axis=1)
    return out.reshape(n, gdn_packed_rows(rows), ROW_SLOTS, ROW_LANES)


def dequantize_gdn_pages(staged_u8, rows: int, jnp, lax):
    """Inverse of ``quantize_gdn_pages``: -> ``[n, rows, 4, 256]`` uint8 GDN rows."""
    n = staged_u8.shape[0]
    flat = staged_u8.reshape(n, -1)
    packed = flat[:, : rows * GDN_ROW_VALUES // 2].reshape(n, rows, GDN_ROW_VALUES // 2)
    scales = flat[:, rows * GDN_ROW_VALUES // 2 :].reshape(
        n, rows, GDN_ROW_VALUES // GROUP, 1
    )
    nib = jnp.stack([packed & 0xF, packed >> 4], axis=-1).reshape(
        n, rows, GDN_ROW_VALUES // GROUP, GROUP
    )
    x = _mx_decode(nib, scales, jnp).reshape(n, rows, GDN_ROW_VALUES)
    return _unwords(x, jnp, lax)


def gdn_rows_from_f32(values: np.ndarray) -> np.ndarray:
    """NumPy, tests and self-test: ``[..., rows, 256]`` FP32 -> ``[..., rows, 4, 256]``
    uint8 rows in the typed view (little-endian words over the sublanes)."""
    u = np.ascontiguousarray(values, dtype="<f4").view(np.uint8)
    u = u.reshape(*values.shape, 4)
    return np.ascontiguousarray(np.swapaxes(u, -1, -2))


def gdn_f32_from_rows(rows: np.ndarray) -> np.ndarray:
    w = np.ascontiguousarray(np.swapaxes(rows, -1, -2))
    return w.view("<f4").reshape(rows.shape[:-2] + (rows.shape[-1],))


def page_scatter(target, values, dst):
    """``target[dst[i]] = values[i]`` in place: one HBM-to-HBM DMA per page, the
    target aliased to the output (donated by the caller). One SMEM index per page;
    the PD row writer's four per-row index vectors overflowed SMEM at 64 pages of
    3,200 rows (j-de0946ff-9209-4465-8784: 1.27M of 1.00M). Repeated destinations
    must carry identical bytes (a padded bucket repeats its last pair)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def kernel(dst_ref, src_ref, _old_ref, out_ref, sem):
        i = pl.program_id(0)
        pltpu.async_copy(src_ref.at[i], out_ref.at[dst_ref[i]], sem).wait()

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(values.shape[0],),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(target.shape, target.dtype),
        input_output_aliases={2: 0},
        name="qwen4_offload_fp4_page_scatter",
    )(dst.astype(jnp.int32), values, target)


def page_gather(source, ids):
    """``source[ids]`` as one HBM-to-HBM DMA per page, bytes untouched. An XLA gather
    of the FP8 region widened the whole operand first (j-7fec99d7-f70c-4a0d-a96a
    reserved 60 GiB, the 30 GiB FP8 region as bf16)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def kernel(ids_ref, src_ref, out_ref, sem):
        i = pl.program_id(0)
        pltpu.async_copy(src_ref.at[ids_ref[i]], out_ref.at[i], sem).wait()

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(ids.shape[0],),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct((ids.shape[0], *source.shape[1:]), source.dtype),
        name="qwen4_offload_fp4_page_gather",
    )(ids.astype(jnp.int32), source)


# -- the two transforms (JAX; traced inside pallas.jax_op) ---------------------


def quantize_pages(pages, jnp, lax):
    """``[n, rows, 4, 256]`` FP8 values (or their uint8 bytes) -> ``[n, rows*4 +
    rows//4, 128]`` uint8 MXFP4.

    The page-sized math runs in bfloat16: FP8 e4m3 values, their powers-of-two
    rescaling and the e2m1 thresholds are all exact in bf16, so the codes match the
    float32 formulation bit for bit while the page-sized temporaries are half the
    bytes. Only the per-group scale (1/32 of the page) is computed in float32."""
    n, rows = pages.shape[0], pages.shape[1]
    if pages.dtype == jnp.uint8:
        pages = lax.bitcast_convert_type(pages, jnp.float8_e4m3fn)
    x = pages.astype(jnp.bfloat16)
    g = x.reshape(n, rows, ROW_SLOTS, ROW_LANES // GROUP, GROUP)
    amax = jnp.max(jnp.abs(g), axis=-1, keepdims=True).astype(jnp.float32)
    # Smallest power of two s with amax / s <= 6: s = 2^ceil(log2(amax / 6)).
    mant, expo = jnp.frexp(amax / 6.0)  # amax/6 = mant * 2^expo, mant in [0.5, 1)
    e = expo - (mant == 0.5).astype(expo.dtype)
    e = jnp.where(amax > 0, e, 0)
    e = jnp.clip(e, -SCALE_BIAS + 1, SCALE_BIAS)
    y = jnp.abs(g) * jnp.exp2(-e.astype(jnp.float32)).astype(jnp.bfloat16)
    code = jnp.zeros(y.shape, jnp.uint8)
    for t, up in zip(E2M1_THRESHOLDS, E2M1_TIE_UP):
        code = code + ((y >= t) if up else (y > t)).astype(jnp.uint8)
    nib = code | ((g < 0).astype(jnp.uint8) << 3)
    nib = nib.reshape(n, rows, ROW_SLOTS, PACKED_LANES, 2)
    packed = nib[..., 0] | (nib[..., 1] << 4)  # [n, rows, 4, 128]
    scales = (e + SCALE_BIAS).astype(jnp.uint8).reshape(n, rows * SCALES_PER_ROW // 128, 128)
    return jnp.concatenate([packed.reshape(n, rows * ROW_SLOTS, 128), scales], axis=1)


def dequantize_pages(staged_u8, rows: int, jnp, lax, as_fp8: bool = False):
    """``[n, rows*4 + rows//4, 128]`` uint8 MXFP4 -> ``[n, rows, 4, 256]`` uint8 FP8
    bytes (``as_fp8``: the FP8 values themselves, with no FP8 -> integer bitcast).

    The e2m1 magnitude is decoded with compares and selects, not a table lookup (an
    XLA gather per value on the TPU), and the page-sized math runs in bfloat16:
    every e2m1 value times a power of two in FP8 range is exact in bf16, so the FP8
    bytes match the float32 formulation."""
    n = staged_u8.shape[0]
    packed = staged_u8[:, : rows * ROW_SLOTS].reshape(n, rows, ROW_SLOTS, PACKED_LANES)
    scales = staged_u8[:, rows * ROW_SLOTS :].reshape(
        n, rows, ROW_SLOTS, ROW_LANES // GROUP, 1
    )
    nib = jnp.stack([packed & 0xF, packed >> 4], axis=-1).reshape(
        n, rows, ROW_SLOTS, ROW_LANES // GROUP, GROUP
    )
    code = nib & 7
    mag = jnp.zeros(code.shape, jnp.bfloat16)
    for k, v in enumerate(E2M1_GRID):
        if v:
            mag = jnp.where(code == k, jnp.bfloat16(v), mag)
    val = jnp.where((nib >> 3) == 1, -mag, mag)
    # 2^(e - 127) is exact in bf16 (float32's exponent range), and so is its product
    # with an e2m1 value (at most 2 significant bits); above FP8_MAX the clip saturates.
    scale = jnp.exp2(scales.astype(jnp.float32) - SCALE_BIAS).astype(jnp.bfloat16)
    val = val * scale
    val = jnp.clip(val, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
    if as_fp8:
        return val.reshape(n, rows, ROW_SLOTS, ROW_LANES)
    return lax.bitcast_convert_type(val, jnp.uint8).reshape(n, rows, ROW_SLOTS, ROW_LANES)


def quantize_reference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference for tests: float values ``[..., 32]`` -> (codes, exponents)."""
    amax = np.max(np.abs(values), axis=-1, keepdims=True)
    mant, expo = np.frexp(amax / 6.0)
    e = np.where(amax > 0, expo - (mant == 0.5), 0).clip(-SCALE_BIAS + 1, SCALE_BIAS)
    y = np.abs(values) * np.exp2(-e.astype(np.float64))
    code = sum(
        ((y >= t) if up else (y > t)).astype(np.uint8)
        for t, up in zip(E2M1_THRESHOLDS, E2M1_TIE_UP)
    )
    return (code | ((values < 0) << 3)).astype(np.uint8), e


# -- worker side -----------------------------------------------------------------


def _bucket(n: int) -> int:
    for b in BUCKETS:
        if n <= b:
            return b
    return BUCKETS[-1]


class Fp4Transcoder:
    """One worker's main-class regions, their staging buffers and the two programs.

    ``pools`` are the model's own typed FP8 region tensors (``[pages, kernel rows,
    *row]``, one 1,024-byte token row per kernel row), never reinterpreted views:
    the raiden managers (this connector's and the PD connector's) pin each
    storage's current device buffer, and a program that wrote a view of another
    dtype rebound the storage to a new buffer, which the PD admission then found
    without its TPU layout (j-fdf67c82-4a51-4f6f-a226). Writes into the pool and
    into staging go through ``page_scatter`` (whole-page DMAs, aliased, donated,
    with the ``copy_`` writeback inside ``torch.compile``). Pool pages are read
    as FP8 values and converted to float, never bitcast to an
    integer type (main-KV pages hold FP8 KV values, not reinterpreted bytes);
    pages are read with ``page_gather`` (whole-page DMAs), never an XLA gather.
    """

    def __init__(
        self,
        pools: dict[int, torch.Tensor],
        page_bytes: int,
        slots: int,
        kind: str = "main",
        offset: int = 0,
    ):
        # kind "main": FP8 main-KV pages (1,700 staged rows a page); "gdn": GDN state
        # pages of FP32 words (425 staged rows a page). ``offset``: this kind's first
        # global slot id (slot_layout); pairs carry global ids.
        assert kind in ("main", "gdn"), kind
        self.kind = kind
        self.offset = int(offset)
        self.pools = dict(pools)
        self.page_bytes = page_bytes
        self.rows = page_rows(page_bytes)
        self.slots = slots
        self.srows = stage_rows(self.rows) if kind == "main" else gdn_packed_rows(self.rows)
        for r, p in self.pools.items():
            if kind == "gdn" and int(p.shape[1]) % 64:
                raise ValueError(
                    f"VLLM_TPU_OFFLOAD_FP4_GDN: region {r} kernel pages of {p.shape[1]} "
                    "rows are not whole 64-row tiles"
                )
            if (
                p.dim() < 3
                or p.dtype != torch.float8_e4m3fn
                or math.prod(p.shape[2:]) * p.element_size() != ROW_VALUES
                or self.rows % int(p.shape[1])
            ):
                raise ValueError(
                    f"VLLM_TPU_OFFLOAD_FP4: region {r} tensor {tuple(p.shape)} {p.dtype} is not "
                    f"FP8 [pages, rows, 1,024-byte row] dividing a {self.rows}-row page"
                )
        self.staging: dict[int, torch.Tensor] = {
            r: torch.zeros(self.staging_shape(slots), dtype=torch.uint8, device=p.device)
            for r, p in self.pools.items()
        }
        self._quant: dict[tuple, object] = {}
        self._dequant: dict[tuple, object] = {}
        self._ops: dict[str, object] = {}

    def staging_shape(self, slots: int) -> tuple[int, int, int, int]:
        return (slots, self.srows, ROW_SLOTS, ROW_LANES)

    @property
    def row_bytes(self) -> int:
        """Host bytes of one entry a region (a staging slot)."""
        return self.srows * ROW_VALUES

    @property
    def staging_bytes(self) -> int:
        return sum(t.numel() for t in self.staging.values())

    def _op(self, kind: str):
        op = self._ops.get(kind)
        if op is not None:
            return op
        import jax
        import jax.numpy as jnp
        from jax import lax
        from torch_tpu._internal import pallas

        rows, srow = self.rows, self.srows
        gdn = self.kind == "gdn"
        tag = "_gdn" if gdn else ""

        def kernel_pages(pool, blocks):
            # Scheduler block b -> its kernel pages b*split .. b*split+split-1.
            split = rows // pool.shape[1]
            ids = blocks.astype(jnp.int32)
            if split == 1:
                return ids
            return (ids[:, None] * split + jnp.arange(split, dtype=jnp.int32)).reshape(-1)

        if kind == "quant":
            # pallas.jax_op reads the argument types from the annotations.
            def quant_fn(
                pool: jax.Array, staging: jax.Array, blocks: jax.Array, slots: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                n = blocks.shape[0]
                if gdn:
                    # FP32 words: read as uint8 rows by DMA, never through an XLA
                    # FP8 op (page_gather_u8).
                    u8 = page_gather_u8(pool, kernel_pages(pool, blocks))
                    q = quantize_gdn_pages(u8.reshape(n, rows, ROW_SLOTS, ROW_LANES), jnp, lax)
                else:
                    pages = page_gather(pool, kernel_pages(pool, blocks)).reshape(
                        n, rows, ROW_SLOTS, ROW_LANES
                    )
                    q = quantize_pages(pages, jnp, lax).reshape(n, srow, ROW_SLOTS, ROW_LANES)
                return page_scatter(staging, q, slots), slots[0]

            op = pallas.jax_op(
                f"pallas::qwen4_offload_fp4_quant{tag}", quant_fn, donate_argnums=(1,)
            )
            op.register_fake(
                lambda pool, staging, blocks, slots: (
                    torch.empty_like(staging),
                    torch.empty((), dtype=slots.dtype, device=slots.device),
                )
            )
        else:

            def dequant_fn(
                pool: jax.Array, staging: jax.Array, slots: jax.Array, blocks: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                n = slots.shape[0]
                staged = staging[slots.astype(jnp.int32)]  # uint8: a plain gather is exact
                if gdn:
                    u8 = dequantize_gdn_pages(staged, rows, jnp, lax)
                    # uint8 -> carrier dtype, as the PD repack writes its rows.
                    values = lax.bitcast_convert_type(u8, pool.dtype).reshape(
                        -1, *pool.shape[1:]
                    )
                else:
                    staged = staged.reshape(
                        n, rows * ROW_SLOTS + rows * SCALES_PER_ROW // 128, 128
                    )
                    vals = dequantize_pages(staged, rows, jnp, lax, as_fp8=True).astype(
                        pool.dtype
                    )
                    values = vals.reshape(-1, *pool.shape[1:])
                return page_scatter(pool, values, kernel_pages(pool, blocks)), blocks[0]

            op = pallas.jax_op(
                f"pallas::qwen4_offload_fp4_dequant{tag}", dequant_fn, donate_argnums=(0,)
            )
            op.register_fake(
                lambda pool, staging, slots, blocks: (
                    torch.empty_like(pool),
                    torch.empty((), dtype=blocks.dtype, device=blocks.device),
                )
            )
        self._ops[kind] = op
        return op

    def _program(self, kind: str, bucket: int):
        cache = self._quant if kind == "quant" else self._dequant
        run = cache.get(bucket)
        if run is None:
            op = self._op(kind)
            if kind == "quant":

                @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
                def run(pool, staging, blocks, slots):
                    new_staging, marker = op(pool, staging, blocks, slots)
                    staging.copy_(new_staging)
                    return marker

            else:

                @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
                def run(pool, staging, slots, blocks):
                    new_pool, marker = op(pool, staging, slots, blocks)
                    pool.copy_(new_pool)
                    return marker

            cache[bucket] = run
        return run

    def _batches(self, pairs: Sequence[tuple[int, int]]):
        pairs = list(pairs)
        top = BUCKETS[-1]
        for i in range(0, len(pairs), top):
            chunk = pairs[i : i + top]
            b = _bucket(len(chunk))
            chunk = chunk + [chunk[-1]] * (b - len(chunk))
            yield b, np.asarray(chunk, dtype=np.int32)

    def owns(self, slot: int) -> bool:
        return self.offset <= slot < self.offset + self.slots

    def _local(self, pairs, slot_at: int) -> list[tuple[int, int]]:
        """This kind's pairs (global slot ids) with local slot ids."""
        out = []
        for p in pairs:
            if self.owns(p[slot_at]):
                q = list(p)
                q[slot_at] -= self.offset
                out.append((q[0], q[1]))
        return out

    def quantize(self, pairs: Sequence[tuple[int, int]]) -> None:
        """``(pool block, global slot)`` pairs: every region's page -> its staging slot.
        Pairs of the other kind are skipped."""
        pairs = self._local(pairs, 1)
        if not pairs or not self.pools:
            return
        for b, arr in self._batches(pairs):
            run = self._program("quant", b)
            for r, pool in self.pools.items():
                dev = pool.device
                blocks = torch.from_numpy(np.ascontiguousarray(arr[:, 0])).to(dev)
                slots = torch.from_numpy(np.ascontiguousarray(arr[:, 1])).to(dev)
                run(pool, self.staging[r], blocks, slots)

    def dequantize(self, pairs: Sequence[tuple[int, int]]) -> None:
        """``(global slot, pool block)`` pairs: every region's staging slot -> its page."""
        pairs = self._local(pairs, 0)
        if not pairs or not self.pools:
            return
        for b, arr in self._batches(pairs):
            run = self._program("dequant", b)
            for r, pool in self.pools.items():
                dev = pool.device
                slots = torch.from_numpy(np.ascontiguousarray(arr[:, 0])).to(dev)
                blocks = torch.from_numpy(np.ascontiguousarray(arr[:, 1])).to(dev)
                run(pool, self.staging[r], slots, blocks)

    def warmup_and_selftest(self) -> dict:
        """Compiles every bucket of both programs on block 0 (vLLM's null block, never
        read) and checks the round trip on scratch tensors of random FP8 values."""
        o = self.offset
        for b in BUCKETS:
            self.quantize([(0, o + i) for i in range(b)])
            self.dequantize([(o + b - 1, 0)] * b)
            # Run them now (execution is deferred): a program that cannot load fails
            # here, attributed to its bucket. Sync the whole tensors (as
            # pooled_cache does), never a slice: a slice read joined the deferred
            # writeback into a program that kept the pool alive, and the copy_ became
            # a 30 GiB copy (j-b32f9066-e0ee-48cc-91cc).
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors(list(self.pools.values()) + list(self.staging.values()))
            logger.info("offload fp4 %s: bucket %d programs ran", self.kind, b)
        return self.selftest()

    def selftest(self) -> dict:
        """Round trip of one page through both programs on the device, on a scratch pool
        of the region tensor's dtype and row shape: random FP8 values (main), or
        random FP32 words over six decades of magnitude (gdn)."""
        if not self.pools:
            return {}
        pool = next(iter(self.pools.values()))
        dev = pool.device
        kr = int(pool.shape[1])
        rng = np.random.default_rng(0)
        if self.kind == "gdn":
            f32 = rng.standard_normal((2, self.rows, GDN_ROW_VALUES)).astype(np.float32)
            # Per-row magnitudes from 1e-4 to 1e2, as state rows differ by head.
            f32 *= np.float32(10.0) ** rng.uniform(-4, 2, (2, self.rows, 1)).astype(np.float32)
            raw = gdn_rows_from_f32(f32)  # [2, rows, 4, 256] uint8
            ref = f32
        else:
            vals = (rng.standard_normal((2, self.rows, ROW_SLOTS, ROW_LANES)) * 4).astype(
                np.float32
            )
            vals[0, :, :, :32] *= 50.0  # a group near the FP8 range
            f8 = torch.from_numpy(vals).to(torch.float8_e4m3fn)
            ref = f8.to(torch.float32).numpy()
            raw = f8.view(torch.uint8).numpy()
        scratch_pool = (
            torch.from_numpy(np.ascontiguousarray(raw))
            .reshape(2 * self.rows // kr, kr, *pool.shape[2:])
            .view(pool.dtype)
            .to(dev)
        )
        scratch_stage = torch.zeros(self.staging_shape(2), dtype=torch.uint8, device=dev)
        q = self._program("quant", 1)
        d = self._program("dequant", 1)
        one = lambda v: torch.tensor([v], dtype=torch.int32).to(dev)
        q(scratch_pool, scratch_stage, one(0), one(0))
        d(scratch_pool, scratch_stage, one(0), one(1))  # slot 0 -> block 1
        out_u8 = scratch_pool.cpu().view(torch.uint8).numpy()
        if self.kind == "gdn":
            back = gdn_f32_from_rows(out_u8.reshape(2, self.rows, ROW_SLOTS, ROW_LANES))[1]
            src = ref[0]
            # Relative error per row (rows differ by orders of magnitude).
            num = np.sqrt(np.mean((back - src) ** 2, axis=-1))
            den = np.maximum(np.sqrt(np.mean(src**2, axis=-1)), 1e-30)
            row_rel = num / den
            out = {
                "rel_rms_err": round(float(np.mean(row_rel)), 4),
                "max_row_rel_err": round(float(np.max(row_rel)), 4),
                "finite": bool(np.isfinite(back).all()),
                "staging_gb": round(self.staging_bytes / 1e9, 2),
                "regions": len(self.pools),
                "pool": f"{tuple(pool.shape)} {pool.dtype}",
            }
            logger.info("offload fp4 gdn: transcode self-test %s", out)
            if not out["finite"] or out["rel_rms_err"] > 0.2 or out["max_row_rel_err"] > 0.3:
                raise RuntimeError(f"VLLM_TPU_OFFLOAD_FP4_GDN: transcode self-test failed {out}")
            return out
        back = (
            torch.from_numpy(out_u8).view(torch.float8_e4m3fn)
            .to(torch.float32).numpy().reshape(2, self.rows, ROW_SLOTS, ROW_LANES)[1]
        )
        src = ref[0]
        err = np.abs(back - src)
        rel = float(np.sqrt(np.mean(err**2)) / max(np.sqrt(np.mean(src**2)), 1e-12))
        out = {
            "rel_rms_err": round(rel, 4),
            "max_abs_err": float(err.max()),
            "finite": bool(np.isfinite(back).all()),
            "staging_gb": round(self.staging_bytes / 1e9, 2),
            "regions": len(self.pools),
            "pool": f"{tuple(pool.shape)} {pool.dtype}",
        }
        logger.info("offload fp4: transcode self-test %s", out)
        if not out["finite"] or rel > 0.2:
            raise RuntimeError(f"VLLM_TPU_OFFLOAD_FP4: transcode self-test failed {out}")
        return out


# Per process: the worker's transcoder, registered by the connector before the class
# workers are built (raiden_store._tier_worker reads the staging buffers from it).
_TRANSCODERS: dict[str, Fp4Transcoder] = {}
_LOCK = threading.Lock()


def set_transcoder(t: Fp4Transcoder | None, kind: str = "main") -> None:
    with _LOCK:
        if t is None:
            _TRANSCODERS.pop(kind, None)
        else:
            assert t.kind == kind, (t.kind, kind)
            _TRANSCODERS[kind] = t


def transcoder(kind: str = "main") -> Fp4Transcoder | None:
    return _TRANSCODERS.get(kind)


def transcoders() -> list[Fp4Transcoder]:
    return list(_TRANSCODERS.values())


def quantize_all(pairs: Sequence[tuple[int, int]]) -> None:
    """``(pool block, global slot)``: each kind's transcoder takes its own slots."""
    for t in transcoders():
        t.quantize(pairs)


def dequantize_all(pairs: Sequence[tuple[int, int]]) -> None:
    for t in transcoders():
        t.dequantize(pairs)


# -- scheduler side --------------------------------------------------------------


class SlotPool:
    """Free list of staging slots (one id space shared by every worker and region)."""

    def __init__(self, slots: int):
        self.total = slots
        self._free = list(range(slots - 1, -1, -1))

    @property
    def free(self) -> int:
        return len(self._free)

    def take(self, n: int) -> list[int] | None:
        if n > len(self._free):
            return None
        out = [self._free.pop() for _ in range(n)]
        return out

    def give(self, slots: Sequence[int]) -> None:
        self._free.extend(slots)
        assert len(self._free) <= self.total, (len(self._free), self.total)
