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
BUCKETS = (1, 4, 16, 64)


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


# -- the two transforms (JAX; traced inside pallas.jax_op) ---------------------


def quantize_pages(pages, jnp, lax):
    """``[n, rows, 4, 256]`` FP8 values (or their uint8 bytes) -> ``[n, rows*4 +
    rows//4, 128]`` uint8 MXFP4."""
    n, rows = pages.shape[0], pages.shape[1]
    if pages.dtype == jnp.uint8:
        pages = lax.bitcast_convert_type(pages, jnp.float8_e4m3fn)
    x = pages.astype(jnp.float32)
    g = x.reshape(n, rows, ROW_SLOTS, ROW_LANES // GROUP, GROUP)
    amax = jnp.max(jnp.abs(g), axis=-1, keepdims=True)
    # Smallest power of two s with amax / s <= 6: s = 2^ceil(log2(amax / 6)).
    mant, expo = jnp.frexp(amax / 6.0)  # amax/6 = mant * 2^expo, mant in [0.5, 1)
    e = expo - (mant == 0.5).astype(expo.dtype)
    e = jnp.where(amax > 0, e, 0)
    e = jnp.clip(e, -SCALE_BIAS + 1, SCALE_BIAS)
    y = jnp.abs(g) * jnp.exp2(-e.astype(jnp.float32))
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
    bytes (``as_fp8``: the FP8 values themselves, with no FP8 -> integer bitcast)."""
    n = staged_u8.shape[0]
    packed = staged_u8[:, : rows * ROW_SLOTS].reshape(n, rows, ROW_SLOTS, PACKED_LANES)
    scales = staged_u8[:, rows * ROW_SLOTS :].reshape(
        n, rows, ROW_SLOTS, ROW_LANES // GROUP, 1
    )
    nib = jnp.stack([packed & 0xF, packed >> 4], axis=-1).reshape(
        n, rows, ROW_SLOTS, ROW_LANES // GROUP, GROUP
    )
    grid = jnp.asarray(E2M1_GRID, jnp.float32)
    mag = grid[(nib & 7).astype(jnp.int32)]
    val = jnp.where((nib >> 3) == 1, -mag, mag)
    val = val * jnp.exp2(scales.astype(jnp.float32) - SCALE_BIAS)
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
    with a plain gather as FP8 values and converted to float, never bitcast to an
    integer type (main-KV pages hold FP8 KV values, not reinterpreted bytes).
    """

    def __init__(self, pools: dict[int, torch.Tensor], page_bytes: int, slots: int):
        self.pools = dict(pools)
        self.page_bytes = page_bytes
        self.rows = page_rows(page_bytes)
        self.slots = slots
        for r, p in self.pools.items():
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
            r: torch.zeros(staging_shape(page_bytes, slots), dtype=torch.uint8, device=p.device)
            for r, p in self.pools.items()
        }
        self._quant: dict[tuple, object] = {}
        self._dequant: dict[tuple, object] = {}
        self._ops: dict[str, object] = {}

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

        rows, srow = self.rows, stage_rows(self.rows)

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
                pages = pool[kernel_pages(pool, blocks)].reshape(n, rows, ROW_SLOTS, ROW_LANES)
                q = quantize_pages(pages, jnp, lax).reshape(n, srow, ROW_SLOTS, ROW_LANES)
                return page_scatter(staging, q, slots), slots[0]

            op = pallas.jax_op("pallas::qwen4_offload_fp4_quant", quant_fn, donate_argnums=(1,))
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
                staged = staged.reshape(n, rows * ROW_SLOTS + rows * SCALES_PER_ROW // 128, 128)
                vals = dequantize_pages(staged, rows, jnp, lax, as_fp8=True).astype(pool.dtype)
                values = vals.reshape(-1, *pool.shape[1:])
                return page_scatter(pool, values, kernel_pages(pool, blocks)), blocks[0]

            op = pallas.jax_op("pallas::qwen4_offload_fp4_dequant", dequant_fn, donate_argnums=(0,))
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

    def quantize(self, pairs: Sequence[tuple[int, int]]) -> None:
        """``(pool block, slot)`` pairs: every region's page -> its staging slot."""
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
        """``(slot, pool block)`` pairs: every region's staging slot -> its page."""
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
        for b in BUCKETS:
            self.quantize([(0, i) for i in range(b)])
            self.dequantize([(b - 1, 0)] * b)
        return self.selftest()

    def selftest(self) -> dict:
        """Round trip of one page of random FP8 values through both programs on the
        device, on a scratch pool of the region tensor's dtype and row shape."""
        if not self.pools:
            return {}
        pool = next(iter(self.pools.values()))
        dev = pool.device
        kr = int(pool.shape[1])
        rng = np.random.default_rng(0)
        vals = (rng.standard_normal((2, self.rows, ROW_SLOTS, ROW_LANES)) * 4).astype(np.float32)
        vals[0, :, :, :32] *= 50.0  # a group near the FP8 range
        f8 = torch.from_numpy(vals).to(torch.float8_e4m3fn)
        ref = f8.to(torch.float32).numpy()
        scratch_pool = (
            f8.view(torch.uint8).reshape(2 * self.rows // kr, kr, *pool.shape[2:])
            .view(pool.dtype).to(dev)
        )
        scratch_stage = torch.zeros(staging_shape(self.page_bytes, 2), dtype=torch.uint8, device=dev)
        q = self._program("quant", 1)
        d = self._program("dequant", 1)
        one = lambda v: torch.tensor([v], dtype=torch.int32).to(dev)
        q(scratch_pool, scratch_stage, one(0), one(0))
        d(scratch_pool, scratch_stage, one(0), one(1))  # slot 0 -> block 1
        back = (
            scratch_pool.cpu().view(torch.uint8).view(torch.float8_e4m3fn)
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
_TRANSCODER: Fp4Transcoder | None = None
_LOCK = threading.Lock()


def set_transcoder(t: Fp4Transcoder | None) -> None:
    global _TRANSCODER
    with _LOCK:
        _TRANSCODER = t


def transcoder() -> Fp4Transcoder | None:
    return _TRANSCODER


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
