# SPDX-License-Identifier: Apache-2.0
"""PLE-INTO-D14 (q38 L1-PLE, ``VLLM_TPU_PLE_D14_FOLD``): the speculative PLE conv of
a megakernel (D14) decode step as one Pallas program with the pool I/O inside it.

On a D14 step the per-layer embedding's dilated conv runs
``pool.pooled_short_conv_speculative`` on D07-b2's bounded request rows
(``VLLM_TPU_PLE_DECODE_BOUNDED`` >= 1): a Pallas history read, a relayout copy,
per-token tap and checkpoint-window gathers (mode 1) or static slices (mode 2)
that materialise T x state_len x C checkpoint values in HBM, then the Pallas
checkpoint writer (one grid step and one waited DMA per token). Here each live
request's window

    E[r] = concat(history[r] masked by initial[r], x[r * window : (r + 1) * window])

lives in VMEM in the pool's own byte layout. One DMA reads the history into it
(every read is waited before any write, as ``read_region`` runs before
``write_region``), the conv taps are static slices of it, and each checkpoint is
one DMA of a contiguous slice of it (``E[r, j + 1 : j + 1 + state_len]``).

The math is ``pool.uniform_window_conv``'s: the same f32 products in the same tap
order starting from zeros, the same BF16 rounding of the sum, and the same masks
and (target, value) pairs. The kernel returns the BF16 pre-activation; silu, its
rounding and the output mask stay in XLA with the same jnp ops, so on TPU only the
f32 tap sum moves from an XLA fusion into Mosaic. The checkpoint targets carry
the same precondition as ``write_region``: unique positive ids.

Taken only when the bucket is exactly ``rows * window`` tokens (D14 steps), the
history is ``(taps - 1) * dilation`` positions and the pool geometry holds whole
tokens per channel row (``fold_geometry``); otherwise the caller keeps the D07-b2
path.
"""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.qwen4_exp.ple import _bf16

# VMEM budget of the fold program (E buffers + double-buffered x / pre / weights);
# a larger bucket keeps the D07-b2 path. The D14 buckets (<= 64 tokens) need <= ~10 MiB.
VMEM_BUDGET = 24 * 1024 * 1024
VMEM_LIMIT = 32 * 1024 * 1024  # the fold program's scoped VMEM limit (Mosaic refuses to compile above it)
_SEEN: set = set()


def fold_geometry(pool, channels: int, state_len: int, window: int):
    """(pool tokens per conv position, bf16 rows per position) or None.

    A position (``channels`` BF16 values) must fill whole pool tokens and whole
    lane rows, so every history position, window token and checkpoint is a
    token range of the pool's own layout (``region_rows`` then has no padding).
    """
    from vllm_torchtpu.kernels import pool_adapters

    if pool.ndim < 4 or pool_adapters.is_seq_along_lane_pool():
        return None
    lanes = pool.shape[-1]
    itemsize = jnp.dtype(pool.dtype).itemsize
    token_bytes = math.prod(pool.shape[2:]) * itemsize
    if (pool.shape[-2] * itemsize) % 2 or channels % lanes or (2 * channels) % token_bytes:
        return None
    tpp = 2 * channels // token_bytes
    if state_len * tpp > pool.shape[1]:
        return None
    return tpp, channels // lanes


def vmem_bytes(pool, rows: int, window: int, taps: int, state_len: int, channels: int) -> int:
    """VMEM of the fold program: E buffers plus double-buffered x, pre and weights."""
    tpp = 2 * channels // (math.prod(pool.shape[2:]) * jnp.dtype(pool.dtype).itemsize)
    token_bytes = math.prod(pool.shape[2:]) * jnp.dtype(pool.dtype).itemsize
    e = rows * (state_len + window) * tpp * token_bytes
    return e + 2 * (2 * rows * window * channels * 2) + 2 * taps * channels * 4


def fold_applies(pool, x, weight, ckpt_ids, *, dilation: int, state_len: int) -> bool:
    """The fold's static preconditions for a bounded bucket (see the module docstring)."""
    count, window = ckpt_ids.shape
    taps = weight.shape[-1]
    return (
        x.shape[0] == count * window
        and state_len == (taps - 1) * dilation
        and fold_geometry(pool, x.shape[-1], state_len, window) is not None
        and vmem_bytes(pool, count, window, taps, state_len, x.shape[-1]) <= VMEM_BUDGET
    )


def _note(token_count: int, rows: int, window: int, tpp: int) -> None:
    """Trace-time engagement line, once per process (a compiled program that comes
    from a cache is not traced again; the bind-time line in pooled_ple always logs)."""
    if _SEEN:
        return
    _SEEN.add(True)
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(
        "q38 PLE-D14-FOLD traced: %d tokens, %d request rows x window %d, conv + checkpoints "
        "in one Pallas program (%d pool tokens per position)",
        token_count,
        rows,
        window,
        tpp,
    )


def _fold_kernel(
    src_ref, tgt_ref, x_ref, w_ref, pool_in_ref, pool_ref, pre_ref, enc, rsem, wsem,
    *, rows, window, taps, dilation, state_len, tpp,
):
    """src_ref[r]: >= 0 read the history from that pool block; -1 live without
    initial state (zero history); -2 inactive (no compute, no writes).
    tgt_ref[r * window + j]: checkpoint block of token (r, j), 0 = no write."""
    ntok = state_len * tpp
    zeros = jnp.zeros(x_ref.shape[2:], jnp.bfloat16)

    def read(r):
        return pltpu.make_async_copy(
            pool_in_ref.at[src_ref[r], pl.ds(0, ntok)], enc.at[r, pl.ds(0, ntok)], rsem.at[r]
        )

    def write(r, j):
        return pltpu.make_async_copy(
            enc.at[r, pl.ds((j + 1) * tpp, ntok)],
            pool_ref.at[tgt_ref[r * window + j], pl.ds(0, ntok)],
            wsem.at[r * window + j],
        )

    def position(r, p):
        return enc.at[r, pl.ds(p * tpp, tpp)]

    # Every history read is waited before any checkpoint write.
    for r in range(rows):
        @pl.when(src_ref[r] >= 0)
        def _start():
            read(r).start()

    for r in range(rows):
        @pl.when(src_ref[r] >= 0)
        def _wait():
            read(r).wait()

        @pl.when(src_ref[r] == -1)
        def _zero():  # the masked `past` of a live request without initial state
            for p in range(state_len):
                typed_ldst.store_typed(position(r, p), zeros)

    for r in range(rows):
        @pl.when(src_ref[r] != -2)
        def _compute():
            for j in range(window):
                typed_ldst.store_typed(position(r, state_len + j), x_ref[r, j])
            for j in range(window):
                acc = jnp.zeros(x_ref.shape[2:], jnp.float32)
                for k in range(taps):
                    v = typed_ldst.load_typed(position(r, k * dilation + j), view_dtype=jnp.bfloat16)
                    acc = acc + v.astype(jnp.float32) * w_ref[k]
                pre_ref[r, j] = acc.astype(jnp.bfloat16)
            for j in range(window):
                @pl.when(tgt_ref[r * window + j] > 0)
                def _ckpt():
                    write(r, j).start()

        @pl.when(src_ref[r] == -2)
        def _idle():
            for j in range(window):
                pre_ref[r, j] = zeros

    for r in range(rows):
        for j in range(window):
            @pl.when((src_ref[r] != -2) & (tgt_ref[r * window + j] > 0))
            def _drain():
                write(r, j).wait()


def uniform_window_conv_fold(
    x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed, *, dilation: int
):
    """``(pool, output)`` of ``pool.pooled_short_conv_speculative`` on a bounded
    uniform-window bucket (``x.shape[0] == count * window``); see the module docstring."""
    token_count, channels = x.shape
    count, window = ckpt_ids.shape
    taps = weight.shape[-1]
    state_len = (taps - 1) * dilation
    geometry = fold_geometry(pool, channels, state_len, window)
    assert token_count == count * window and geometry is not None, (x.shape, ckpt_ids.shape, pool.shape)
    tpp, crow = geometry
    lanes = pool.shape[-1]
    _note(token_count, count, window, tpp)
    # Masks and (target, value) pairs exactly as uniform_window_conv.
    lengths = starts[1:] - starts[:-1]
    active = (ids > 0) & (lengths > 0)
    t = jnp.arange(token_count, dtype=jnp.int32)
    req = t // window
    rel = t % window
    valid = (t < starts[-1]) & active[req]
    win = windowed[req]
    write = valid & jnp.where(win, rel < window, rel == lengths[req] - 1)
    column = jnp.where(win, jnp.clip(rel, 0, window - 1), 0)
    targets = jnp.where(write, ckpt_ids[req, column], 0).astype(jnp.int32)
    # read_region reads row max(id, 0); `past` is zero without initial state.
    source = jnp.where(
        active, jnp.where(initial, jnp.maximum(read_ids, 0), -1), -2
    ).astype(jnp.int32)
    x4 = x.reshape(count, window, crow, lanes)
    w4 = jnp.transpose(weight).astype(jnp.float32).reshape(taps, crow, lanes)
    kernel = _kernel_partial(count, window, taps, dilation, state_len, tpp)
    pool, pre = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[
                pl.BlockSpec((count, window, crow, lanes), lambda i, s, g: (0, 0, 0, 0)),
                pl.BlockSpec((taps, crow, lanes), lambda i, s, g: (0, 0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((count, window, crow, lanes), lambda i, s, g: (0, 0, 0, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((count, (state_len + window) * tpp) + tuple(pool.shape[2:]), pool.dtype),
                pltpu.SemaphoreType.DMA((count,)),
                pltpu.SemaphoreType.DMA((count * window,)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
            jax.ShapeDtypeStruct((count, window, crow, lanes), jnp.bfloat16),
        ],
        input_output_aliases={4: 0},
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
        name="q38_ple_d14_fold",
    )(source, targets, x4, w4, pool)
    pre = pre.reshape(token_count, channels)
    output = _bf16(jax.nn.silu(pre.astype(jnp.float32)))
    return pool, jnp.where(valid[:, None], output, 0)


def _kernel_partial(rows, window, taps, dilation, state_len, tpp):
    import functools

    return functools.partial(
        _fold_kernel, rows=rows, window=window, taps=taps, dilation=dilation,
        state_len=state_len, tpp=tpp,
    )
