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

Staging layout per region: ``[slots, rows * 4 + rows // 4, 128]`` uint8, i.e.
per slot the packed values (``rows * 4`` sublane rows of 128 bytes: one head
slot of 256 lanes per row) followed by the scales (``rows * 32`` bytes =
``rows // 4`` rows of 128). A 3,200-token page: 12,800 + 800 rows of 128 B =
1,740,800 B, no tile padding. Pages are read and written as uint8 views (the
PD repack's rule: XLA must never handle FP8 KV bytes as FP8 in a gather or a
bitcast, which canonicalised NaN encodings); the FP8 interpretation happens
only on values about to be quantized or just dequantized.
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
    if rows % 4:
        raise ValueError(f"VLLM_TPU_OFFLOAD_FP4: {rows} rows a page is not a multiple of 4")
    return rows


def fp4_page_bytes(page_bytes: int) -> int:
    """Host bytes of one main-KV page in MXFP4 (values + scales)."""
    rows = page_rows(page_bytes)
    return rows * (ROW_VALUES // 2 + SCALES_PER_ROW)


def staging_shape(page_bytes: int, slots: int) -> tuple[int, int, int]:
    rows = page_rows(page_bytes)
    return (slots, rows * ROW_SLOTS + rows * SCALES_PER_ROW // 128, 128)


# -- the two transforms (JAX; traced inside pallas.jax_op) ---------------------


def quantize_pages(pages_u8, jnp, lax):
    """``[n, rows, 4, 256]`` uint8 FP8 bytes -> ``[n, rows*4 + rows//4, 128]`` uint8 MXFP4."""
    n, rows = pages_u8.shape[0], pages_u8.shape[1]
    x = lax.bitcast_convert_type(pages_u8, jnp.float8_e4m3fn).astype(jnp.float32)
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


def dequantize_pages(staged_u8, rows: int, jnp, lax):
    """``[n, rows*4 + rows//4, 128]`` uint8 MXFP4 -> ``[n, rows, 4, 256]`` uint8 FP8 bytes."""
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
    """One worker's main-class regions, their staging buffers and the two programs."""

    def __init__(self, pools: dict[int, torch.Tensor], page_bytes: int, slots: int):
        # region -> the pool region as a uint8 view [num_blocks, rows, 4, 256]
        self.pools = dict(pools)
        self.page_bytes = page_bytes
        self.rows = page_rows(page_bytes)
        self.slots = slots
        self.staging: dict[int, torch.Tensor] = {
            r: torch.zeros(staging_shape(page_bytes, slots), dtype=torch.uint8, device=p.device)
            for r, p in self.pools.items()
        }
        self._quant: dict[int, object] = {}
        self._dequant: dict[int, object] = {}
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

        rows = self.rows
        if kind == "quant":
            # pallas.jax_op reads the argument types from the annotations.
            def quant_fn(
                pool: jax.Array, staging: jax.Array, blocks: jax.Array, slots: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                pages = pool[blocks.astype(jnp.int32)]
                return staging.at[slots.astype(jnp.int32)].set(
                    quantize_pages(pages, jnp, lax)
                ), slots[0]

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
                pages = dequantize_pages(staging[slots.astype(jnp.int32)], rows, jnp, lax)
                return pool.at[blocks.astype(jnp.int32)].set(pages), blocks[0]

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
        read) and checks the round trip on a scratch region of random FP8 values."""
        for b in BUCKETS:
            self.quantize([(0, i) for i in range(b)])
            self.dequantize([(b - 1, 0)] * b)
        return self.selftest()

    def selftest(self) -> dict:
        """Round trip of one page of random FP8 values through both transforms, on
        the device, on scratch tensors of the same programs' shapes (bucket 1)."""
        if not self.pools:
            return {}
        r0 = next(iter(self.pools))
        dev = self.pools[r0].device
        rng = np.random.default_rng(0)
        vals = (rng.standard_normal((2, self.rows, ROW_SLOTS, ROW_LANES)) * 4).astype(np.float32)
        vals[0, :, :, :32] *= 50.0  # a group near the FP8 range
        f8 = torch.from_numpy(vals).to(torch.float8_e4m3fn)
        ref = f8.to(torch.float32).numpy()
        scratch_pool = f8.view(torch.uint8).to(dev)
        scratch_stage = torch.zeros(staging_shape(self.page_bytes, 2), dtype=torch.uint8, device=dev)
        q = self._program("quant", 1)
        d = self._program("dequant", 1)
        one = lambda v: torch.tensor([v], dtype=torch.int32).to(dev)
        q(scratch_pool, scratch_stage, one(0), one(0))
        d(scratch_pool, scratch_stage, one(0), one(1))  # slot 0 -> block 1
        back = scratch_pool.cpu().view(torch.float8_e4m3fn).to(torch.float32).numpy()[1]
        src = ref[0]
        err = np.abs(back - src)
        rel = float(np.sqrt(np.mean(err**2)) / max(np.sqrt(np.mean(src**2)), 1e-12))
        out = {
            "rel_rms_err": round(rel, 4),
            "max_abs_err": float(err.max()),
            "finite": bool(np.isfinite(back).all()),
            "staging_gb": round(self.staging_bytes / 1e9, 2),
            "regions": len(self.pools),
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
