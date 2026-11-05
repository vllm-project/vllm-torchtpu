# SPDX-License-Identifier: Apache-2.0
"""LM-head logits at decode row counts, weight streamed from HBM (TPU_LM_HEAD_HBM).

The vocab-parallel LM head is `x [m, k] bf16 @ w [n, k]^T bf16 -> [m, n] bf16` with the
weight N-major, row-major `[vocab/tp, hidden]` (vLLM's `F.linear`; GLM-5.2 TP16:
n = 9680, k = 6144, 119 MB per rank). XLA's dot streams that layout at about 2.6 TB/s
on tpu7x (44-47 us per call at m = 4..64), and torch_tpu/XLA will not keep the weight
K-major. This kernel takes the weight as an HBM operand -- constrained to HBM with
`pltpu.with_memory_space_constraint` -- and walks it through a
3-slot VMEM ring of contiguous row tiles `w[s:s+t, :]` in one grid step, computing each
tile as it lands with one dot that contracts k on both operands (the weight tile is the
MXU's transposed right-hand side; Mosaic absorbs the transpose into the push).

Measured on tpu7x, one TensorCore, jax 0.11.2 (write-tpu-kernels t6-lmhead probes
01-04, j-b8dd8d2e / j-f27ab5bc / j-7de87885 / j-9b4627be; chained calls with a dot
between them): this kernel 35.7-37.6 us at m <= 16 and 40.0-41.6 at m = 64, vs XLA's
dot 43.8-48.1 in the same jobs. The dots bind, not HBM: the weight push alone (DMA
removed) runs 35.8-37.2 us, the DMA ring alone 33.1-34.9. Row tiles of 512 are the
measured best: 256-row tiles serialize the DMA with the dot (66-70 us), 640/768/1280
rows lose 4-10 us, and a 2-slot ring loses 8 us.

Numerics: one dot per tile over the full k, f32 accumulation, cast once to bf16. Not
bit-identical to XLA's dot: about 1-2 in 10^4 logits differ, by 1-4 bf16 ulp of
max(|logit|, 2^-6) (no k-chunking or summation order tried reproduces XLA's bits); the
error against the f64 product matches XLA's. Draft-token use is lossless for the
target distribution; the target sampler's use needs an accuracy check.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MAX_M = 96
TILE = 512
NBUF = 3
MAX_K = 8192  # ring = NBUF x TILE x k x 2 B <= 24 MiB
SUBLANE_BF16 = 16
LANE = 128
VMEM_LIMIT = 32 * 1024 * 1024  # default config: ring 18.9 MB + x / out blocks <= 4.1 MB
VMEM_LIMIT_MAX = 56 * 1024 * 1024

# Timing-only and fault-injection variants for the kernel's own tests and benchmark.
_VARIANTS = (
    None,
    "dma_only",
    "compute_only",
    "drop_tail",
    "off",
    "stale",
    "xrow",
    "kdrop",
)


def vmem_need(
    m: int, k: int, n: int, t: int = TILE, nbuf: int = NBUF, item: int = 2
) -> int:
    """Scoped VMEM bytes of one call: the ring plus the double-buffered x and out
    blocks."""
    t = min(t, n)
    return nbuf * t * k * item + 2 * m * k * item + 2 * m * n * item


def supports(m: int, x: jax.Array, w: jax.Array) -> bool:
    """Decode row counts (m <= MAX_M), bf16 operands, k a lane multiple <= MAX_K,
    n a bf16 sublane-tile multiple, and the default config's VMEM within the cap (a
    very wide vocab shard keeps XLA's dot instead of failing to compile)."""
    return (
        1 <= m <= MAX_M
        and x.ndim == 2
        and w.ndim == 2
        and x.shape[0] == m
        and x.dtype == jnp.bfloat16
        and w.dtype == jnp.bfloat16
        and x.shape[1] == w.shape[1]
        and w.shape[1] % LANE == 0
        and w.shape[1] <= MAX_K
        and w.shape[0] % SUBLANE_BF16 == 0
        and vmem_need(m, w.shape[1], w.shape[0]) + (4 << 20) <= VMEM_LIMIT_MAX
    )


def tiles_of(n: int, t: int) -> tuple[tuple[int, int], ...]:
    """(first row, rows) of each tile: full tiles of t, then the n % t tail."""
    out = [(j * t, t) for j in range(n // t)]
    if n % t:
        out.append(((n // t) * t, n % t))
    return tuple(out)


def _kernel(x_ref, w_hbm, o_ref, wbuf, sem, *, tiles, nbuf, variant):
    n_tiles = len(tiles)
    last = n_tiles - 1

    def copy(j):
        s, r = tiles[j]
        if variant == "off" and j == 0:
            # fault: the first tile is fetched 16 rows late (its first 16 rows dropped,
            # its last 16 slot rows never written)
            s, r = s + SUBLANE_BF16, r - SUBLANE_BF16
        return pltpu.make_async_copy(
            w_hbm.at[pl.ds(s, r), :],
            wbuf.at[j % nbuf, pl.ds(0, r), :],
            sem.at[j % nbuf],
        )

    compute_only = variant == "compute_only"
    for j in range(min(nbuf, n_tiles)):
        copy(j).start()
    if (
        compute_only
    ):  # timing only: fetch the first slots once, compute every tile from them
        for j in range(min(nbuf, n_tiles)):
            copy(j).wait()

    m, k = x_ref.shape
    dn = (((1,), (1,)), ((), ()))
    for j in range(n_tiles):
        s, r = tiles[j]
        if not compute_only:
            copy(j).wait()
        slot = (j + 1) % nbuf if variant == "stale" else j % nbuf
        if (
            variant == "dma_only"
        ):  # timing only: no dot; keep a live sink on the last tile
            if j == last:
                rows = min(r, m)
                o_ref[0:rows, 0:LANE] = wbuf[slot, 0:rows, 0:LANE].astype(o_ref.dtype)
        else:
            # x is re-read from its VMEM block for every tile (holding the [m, k] value
            # live across the unrolled loop measured +2.3 us at m = 4, probe 03).
            xx = x_ref[...]
            if variant == "xrow" and m > 1:  # fault: rows of x rotated by one
                xx = jnp.concatenate([xx[1:], xx[:1]], axis=0)
            w = wbuf[slot, 0:r, :]
            if variant == "kdrop":  # fault: the last lane-tile of k is not contracted
                xx, w = xx[:, : k - LANE], w[:, : k - LANE]
            y = jax.lax.dot_general(xx, w, dn, preferred_element_type=jnp.float32)
            if not (variant == "drop_tail" and j == last):
                o_ref[:, s : s + r] = y.astype(o_ref.dtype)
        if not compute_only and j + nbuf < n_tiles:
            copy(j + nbuf).start()


@functools.cache
def _build(m: int, k: int, n: int, dtype, t: int, nbuf: int, variant=None):
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    t = min(t, n)
    if t % SUBLANE_BF16 or n % SUBLANE_BF16:
        raise ValueError(f"tile {t} / rows {n} must be multiples of {SUBLANE_BF16}")
    tiles = tiles_of(n, t)
    nbuf = max(1, min(nbuf, len(tiles)))
    # a non-default config may need more than the default scoped limit (t=1024 x 3
    # slots: 37.7 MB), never more than 56 MiB
    need = vmem_need(m, k, n, t, nbuf, jnp.dtype(dtype).itemsize)
    vmem_limit = max(VMEM_LIMIT, need + (4 << 20))
    if vmem_limit > VMEM_LIMIT_MAX:
        raise ValueError(f"config t={t} nbuf={nbuf} needs {need} B of VMEM")
    kern = functools.partial(_kernel, tiles=tiles, nbuf=nbuf, variant=variant)
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m, k), lambda g: (0, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((m, n), lambda g: (0, 0)),
        scratch_shapes=[
            pltpu.VMEM((nbuf, t, k), dtype),
            pltpu.SemaphoreType.DMA((nbuf,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=vmem_limit
        ),
        name=(
            f"lm_head_hbm_t{t}_b{nbuf}{'_' + variant if variant else ''}_m{m}_k{k}_n{n}"
        ),
    )


@functools.partial(jax.jit, static_argnums=0)
def _call_hbm_pinned(call, x, w):
    # The constraint sets the operand aval's memory space, which pallas_call lowers
    # to the custom call's `input_memory_space_colors` (HBM = 0); without it XLA may
    # stage the operand into VMEM first. It only binds a pallas_call input and cannot
    # run eagerly, hence the jit around both.
    return call(x, pltpu.with_memory_space_constraint(w, pltpu.HBM))


def lm_head_hbm(
    x: jax.Array,
    w: jax.Array,
    *,
    config: tuple[int, int] | None = None,
    _variant: str | None = None,
) -> jax.Array:
    """x [m <= MAX_M, k] bf16, w [n, k] bf16 -> x @ w.T [m, n] bf16, weight read from
    HBM. `config` = (tile rows, ring depth). `_variant`: tests and benchmarks only."""
    m, k = x.shape
    n = w.shape[0]
    t, nbuf = config or (TILE, NBUF)
    return _call_hbm_pinned(_build(m, k, n, x.dtype, t, nbuf, _variant), x, w)


def lm_head_logits(x: jax.Array, w: jax.Array) -> jax.Array:
    """`F.linear(x, w)` for a bias-free bf16 LM head: the HBM-streaming kernel at decode
    row counts, XLA's dot otherwise (unchanged arithmetic)."""
    if supports(x.shape[0], x, w):
        return lm_head_hbm(x, w)
    return jax.lax.dot_general(x, w, (((1,), (1,)), ((), ())))
